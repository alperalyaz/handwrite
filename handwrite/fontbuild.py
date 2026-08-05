"""OpenType (TTF) font üretimi.

Glif kütüphanesini alıp kurulabilir bir font dosyasına çevirir. İki karar bu
modülün çıktısını Calligraphr benzeri araçlardan ayırır:

**Varyant döngüsü.** Her karakterin birden çok gerçek örneği fonta konur ve
OpenType `calt` (contextual alternates) özelliğiyle sırayla kullanılır. El
yazısı fontlarının sahte görünmesinin bir numaralı sebebi her "a"nın piksel
piksel aynı olmasıdır; iki kısa kural bunu çözer:

    sub @BASE @BASE' by @ALT1;    # bir temel harften sonra gelen -> 1. varyant
    sub @ALT1 @BASE' by @ALT2;    # 1. varyanttan sonra gelen   -> 2. varyant

Üçüncü bir kurala gerek yok: ALT2'den sonraki harf hiçbir kurala uymaz ve temel
biçimde kalır, böylece döngü kendiliğinden BASE→ALT1→ALT2→BASE olur.

**Ölçülmüş metrikler.** İlerleme genişlikleri ve yan boşluklar tahmin edilmez;
kullanıcının kağıt üzerindeki gerçek harf aralıklarından ölçülür. Kişinin kendi
yazı ritmi fonta böyle geçer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fontTools import agl
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

from .config import FontConfig, GlyphConfig
from .glyph import CharacterSet, Glyph, GlyphLibrary
from .vectorize import Contour, trace_mask

NOTDEF = ".notdef"
SPACE = "space"


# --------------------------------------------------------------------------
# Glif adları
# --------------------------------------------------------------------------


def glyph_name(char: str) -> str:
    """Karakter için üretim glif adı üretir.

    Mümkünse standart AGL adı ("a", "zero", "ccedilla") kullanılır; font
    editöründe açıldığında okunabilir olur. Kalanlar için "uniXXXX" biçimi
    kullanılır ki ad çakışması ve belirsizlik olmasın.
    """
    code = ord(char)
    name = agl.UV2AGL.get(code)
    if name and name.isidentifier():
        return name
    return f"uni{code:04X}"


def variant_name(base: str, index: int) -> str:
    return base if index == 0 else f"{base}.alt{index}"


# --------------------------------------------------------------------------
# Kontur → font birimi
# --------------------------------------------------------------------------


def _signed_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _flatten_for_area(contour: Contour) -> list[tuple[float, float]]:
    """Alan işareti için konturun düğüm noktaları — eğri örneklemesine gerek yok."""
    return contour.points()


def glyph_outline(
    glyph: Glyph,
    font: FontConfig,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> list[list[tuple]]:
    """Bir glifi font birimlerinde kontur listesine çevirir.

    Piksel uzayı y-aşağı, font uzayı y-yukarıdır; çevirme sırasında bütün
    konturların yönü tersine döner. TrueType dolgu kuralı yön duyarlı olduğu
    için, çevirmeden sonra toplam işaretli alana bakıp gerekiyorsa hepsini
    birden ters çeviriyoruz. "Hepsini birden" olması önemli: dış kontur ile
    deliklerin *birbirine göre* yönü korunmalı, yoksa "a"nın gözü dolar.
    """
    contours = trace_mask(glyph.mask, font)
    if not contours:
        return []

    scale = glyph.scale
    origin_x = glyph.bearing_px
    origin_y = glyph.ascent_px

    def to_font(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            (origin_x + x) * scale + shift_x,
            (origin_y - y) * scale + shift_y,
        )

    converted: list[list[tuple]] = []
    total_area = 0.0
    for contour in contours:
        path: list[tuple] = [("move", to_font(contour.start))]
        for segment in contour.segments:
            if segment[0] == "line":
                path.append(("line", to_font(segment[1])))
            else:
                path.append(
                    ("curve", to_font(segment[1]), to_font(segment[2]), to_font(segment[3]))
                )
        converted.append(path)
        total_area += _signed_area([to_font(p) for p in _flatten_for_area(contour)])

    if total_area > 0:
        converted = [_reverse_path(path) for path in converted]
    return converted


def _reverse_path(path: list[tuple]) -> list[tuple]:
    """Bir konturun yönünü tersine çevirir (kontrol noktaları dahil)."""
    nodes: list[tuple[float, float]] = [path[0][1]]
    kinds: list[tuple] = []
    for segment in path[1:]:
        if segment[0] == "line":
            kinds.append(("line",))
            nodes.append(segment[1])
        else:
            kinds.append(("curve", segment[1], segment[2]))
            nodes.append(segment[3])

    reversed_path: list[tuple] = [("move", nodes[-1])]
    for i in range(len(kinds) - 1, -1, -1):
        target = nodes[i]
        kind = kinds[i]
        if kind[0] == "line":
            reversed_path.append(("line", target))
        else:
            reversed_path.append(("curve", kind[2], kind[1], target))
    return reversed_path


def draw_outline(pen, path_list: list[list[tuple]]) -> None:
    """Kontur listesini bir kaleme çizer."""
    for path in path_list:
        pen.moveTo(path[0][1])
        for segment in path[1:]:
            if segment[0] == "line":
                pen.lineTo(segment[1])
            else:
                pen.curveTo(segment[1], segment[2], segment[3])
        pen.closePath()


# --------------------------------------------------------------------------
# Font kurulumu
# --------------------------------------------------------------------------


@dataclass
class BuildReport:
    """Font üretiminin özeti."""

    glyph_count: int = 0
    characters: int = 0
    missing: list[str] = field(default_factory=list)
    variants_used: dict[str, int] = field(default_factory=dict)
    empty_outlines: list[str] = field(default_factory=list)


def _variant_placement(
    entry: CharacterSet, variant: Glyph, cfg: GlyphConfig
) -> tuple[float, float]:
    """Bir varyantın karakter medyanına göre kaydırılması (font birimi).

    Ham örnekler olduğu gibi yerleştirilirse harfler taban çizgisinde zıplar ve
    yan boşluklar düzensiz olur; tamamen hizalanırsa el yazısı hissi kaybolur.
    Kaydırma, ikisi arasında ayarlanabilir bir noktada durur.
    """
    shift_x = cfg.horizontal_regularize * (entry.bearing - variant.bearing)
    shift_y = cfg.vertical_regularize * (variant.descent - entry.descent)
    return shift_x, shift_y


def build_font(
    library: GlyphLibrary,
    font_cfg: FontConfig,
    glyph_cfg: GlyphConfig,
) -> tuple[object, BuildReport]:
    """Glif kütüphanesinden bir TTFont nesnesi üretir."""
    report = BuildReport()

    order: list[str] = [NOTDEF, SPACE]
    outlines: dict[str, list[list[tuple]]] = {NOTDEF: [], SPACE: []}
    advances: dict[str, float] = {
        NOTDEF: font_cfg.default_space_width,
        SPACE: library.space_advance or font_cfg.default_space_width,
    }
    cmap: dict[int, str] = {ord(" "): SPACE}

    base_names: list[str] = []
    alt_names: list[list[str]] = [[] for _ in range(max(0, glyph_cfg.variants_per_char - 1))]

    for char, entry in sorted(library.characters.items()):
        if not entry.variants:
            continue
        base = glyph_name(char)
        drawn: list[str] = []

        for index in range(glyph_cfg.variants_per_char):
            # Yeterli farklı örnek yoksa varyantlar başa dönerek tekrarlanır.
            # Ayrı glif olarak yazılmaları, `calt` sınıflarının aynı uzunlukta
            # ve birebir eşlenik kalmasını garanti eder.
            variant = entry.variants[index % len(entry.variants)]
            name = variant_name(base, index)
            shift_x, shift_y = _variant_placement(entry, variant, glyph_cfg)
            path = glyph_outline(variant, font_cfg, shift_x, shift_y)
            if not path:
                report.empty_outlines.append(name)
            outlines[name] = path
            advances[name] = entry.advance
            order.append(name)
            drawn.append(name)

        cmap[ord(char)] = base
        base_names.append(base)
        for i in range(1, glyph_cfg.variants_per_char):
            alt_names[i - 1].append(drawn[i])

        report.characters += 1
        report.variants_used[char] = len(entry.variants)

    report.glyph_count = len(order)

    builder = FontBuilder(font_cfg.units_per_em, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(cmap)

    tolerance = font_cfg.cu2qu_tolerance * font_cfg.units_per_em
    glyf: dict[str, object] = {}
    for name in order:
        pen = TTGlyphPen(None)
        draw_outline(Cu2QuPen(pen, tolerance), outlines.get(name, []))
        glyf[name] = pen.glyph()
    builder.setupGlyf(glyf)

    metrics = {}
    for name in order:
        glyph_obj = glyf[name]
        left = 0
        if hasattr(glyph_obj, "numberOfContours") and glyph_obj.numberOfContours > 0:
            glyph_obj.recalcBounds(builder.font["glyf"])
            left = glyph_obj.xMin
        metrics[name] = (int(round(advances[name])), int(round(left)))
    builder.setupHorizontalMetrics(metrics)

    builder.setupHorizontalHeader(
        ascent=font_cfg.ascender, descent=font_cfg.descender, lineGap=font_cfg.line_gap
    )
    builder.setupNameTable(_name_records(font_cfg))
    builder.setupOS2(
        sTypoAscender=font_cfg.ascender,
        sTypoDescender=font_cfg.descender,
        sTypoLineGap=font_cfg.line_gap,
        usWinAscent=font_cfg.ascender,
        usWinDescent=abs(font_cfg.descender),
        sxHeight=font_cfg.x_height,
        sCapHeight=font_cfg.cap_height,
        achVendID="HNDW",
        fsType=0,
    )
    builder.setupPost(isFixedPitch=0)

    feature = _calt_feature(base_names, alt_names)
    if feature:
        addOpenTypeFeaturesFromString(builder.font, feature)

    return builder.font, report


def _name_records(cfg: FontConfig) -> dict[str, str]:
    full = f"{cfg.family_name} {cfg.style_name}"
    postscript = f"{cfg.family_name}-{cfg.style_name}".replace(" ", "")
    return {
        "familyName": cfg.family_name,
        "styleName": cfg.style_name,
        "uniqueFontIdentifier": f"{postscript}; handwrite",
        "fullName": full,
        "version": "Version 1.000",
        "psName": postscript,
        "manufacturer": "handwrite",
    }


def _calt_feature(base_names: list[str], alt_names: list[list[str]]) -> str:
    """Varyant döngüsünü kuran `calt` özelliğini üretir.

    Sınıflar birebir eşlenik olmalıdır: @BASE'in i. glifi ile @ALT1'in i. glifi
    aynı karakterin iki biçimidir. Sınıf tabanlı yer değiştirme eşlemeyi
    konumdan okur, isimden değil.
    """
    usable = [names for names in alt_names if len(names) == len(base_names) and names]
    if not base_names or not usable:
        return ""

    lines = [
        "languagesystem DFLT dflt;",
        "languagesystem latn dflt;",
        "",
        f"@BASE = [{' '.join(base_names)}];",
    ]
    for i, names in enumerate(usable, start=1):
        lines.append(f"@ALT{i} = [{' '.join(names)}];")

    lines += ["", "feature calt {"]
    lines.append("    sub @BASE @BASE' by @ALT1;")
    for i in range(1, len(usable)):
        lines.append(f"    sub @ALT{i} @BASE' by @ALT{i + 1};")
    lines += ["} calt;", ""]
    return "\n".join(lines)


def save_font(font, path: str | Path) -> None:
    """Fontu diske yazar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(path))
