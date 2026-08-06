"""Uçtan uca boru hattı: fotoğraflardan font dosyasına.

    fotoğraf → kayıt → mürekkep → satır → eğim → karakter → glif → kontur → TTF

Her adım kendi modülünde; burada yapılan iş onları sırayla çağırmak, sayfalar
arası ortak kararları (eğim gibi) tek bir yerden vermek ve kullanıcıya
gösterilecek teşhisi toplamaktır.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import Config
from .fontbuild import BuildReport, build_font, save_font
from .glyph import GlyphLibrary, apply_xheights, build_library, measure_xheights, normalize_all
from .lines import Line, deslant_line, detect_lines, lines_from_bands, page_slant
from .preprocess import RegistrationError, detect_markers, load_gray, prepare_page
from .segment import DocumentSegmentation, segment_document
from .template import SheetSet, SheetSpec


@dataclass
class PageResult:
    """Tek bir sayfanın işlenme sonucu."""

    source: str
    page_index: int | None
    lines: int
    ink_pixels: int
    error: str | None = None


@dataclass
class Diagnostics:
    """Kullanıcıya gösterilecek teşhis bilgisi."""

    pages: list[PageResult] = field(default_factory=list)
    slant: float = 0.0
    stroke_width: float = 0.0
    #: Ölçülen belge x-yüksekliği (piksel).
    xheight_px: float = 0.0
    #: (sayfa, satır) → eleme sebebi.
    rejected_lines: dict[tuple[int, int], str] = field(default_factory=dict)
    #: Hiç örnek bulunamayan, dolayısıyla fonta giremeyen karakterler.
    missing_characters: list[str] = field(default_factory=list)
    #: Örneklerinin çoğu elenmiş, gözden geçirilmesi iyi olur.
    weak_characters: list[tuple[str, int, int]] = field(default_factory=list)
    total_samples: int = 0
    #: Serbest modda modelin okuduğu satırlar: (indeks, metin, güven).
    transcriptions: list[tuple[int, str, float]] = field(default_factory=list)
    #: Hizalama maliyeti yüksek olduğu için atılan glif sayısı.
    dropped_glyphs: int = 0

    def summary_lines(self) -> list[str]:
        rows = [
            f"sayfa: {len([p for p in self.pages if p.error is None])}/{len(self.pages)} okundu",
            f"eğim: {self.slant:.1f}°   kalem: {self.stroke_width:.1f} px   x-yüksekliği: {self.xheight_px:.0f} px",
            f"toplam karakter örneği: {self.total_samples}",
        ]
        if self.transcriptions:
            rows.append(f"okunan satır: {len(self.transcriptions)}")
        if self.rejected_lines:
            rows.append(f"elenen satır: {len(self.rejected_lines)}")
        if self.dropped_glyphs:
            rows.append(f"hizalaması şüpheli bulunup atılan glif: {self.dropped_glyphs}")
        if self.missing_characters:
            rows.append("fontta olmayan karakter: " + " ".join(self.missing_characters))
        if self.weak_characters:
            weak = ", ".join(f"{c} ({k}/{s})" for c, s, k in self.weak_characters[:10])
            rows.append(f"az örnekli karakter: {weak}")
        return rows


@dataclass
class PipelineResult:
    """Boru hattının tam çıktısı."""

    font: object
    library: GlyphLibrary
    build: BuildReport
    diagnostics: Diagnostics
    segmentation: DocumentSegmentation | None = None


def identify_page(gray: np.ndarray, sheets: SheetSet) -> SheetSpec | None:
    """Fotoğraftaki köşe işaretlerinden hangi sayfa olduğunu bulur.

    Her sayfa farklı işaret kimlikleri taşıdığı için fotoğrafların sırası ya da
    dosya adı önemli değildir; kullanıcı hepsini bir kerede, karışık atabilir.
    """
    found = set(detect_markers(gray))
    if not found:
        return None
    best, best_hits = None, 0
    for spec in sheets.sheets:
        hits = len(found & set(spec.marker_ids))
        if hits > best_hits:
            best, best_hits = spec, hits
    return best if best_hits >= 3 else None


def collect_lines(
    images: list[tuple[str, np.ndarray]],
    sheets: SheetSet,
    cfg: Config,
) -> tuple[list[Line], dict[int, tuple[int, int]], Diagnostics]:
    """Bütün sayfalardan satırları toplar.

    Satırlar sayfalar arasında yeniden numaralandırılır: her sayfa kendi
    bantlarını 0'dan saydığı için, birleştirilmeden önce numaralar çakışırdı.
    Dönen eşleme, teşhiste satırı tekrar (sayfa, bant) olarak gösterebilmek
    içindir.
    """
    diagnostics = Diagnostics()
    lines: list[Line] = []
    origin: dict[int, tuple[int, int]] = {}
    strokes: list[float] = []

    for name, image in images:
        spec = identify_page(image, sheets)
        if spec is None:
            diagnostics.pages.append(
                PageResult(
                    source=name,
                    page_index=None,
                    lines=0,
                    ink_pixels=0,
                    error="sayfa köşe işaretleri okunamadı — dört köşenin de "
                    "kadrajda ve net olduğundan emin olun",
                )
            )
            continue

        try:
            page = prepare_page(image, spec, cfg.preprocess)
        except RegistrationError as exc:
            diagnostics.pages.append(
                PageResult(source=name, page_index=spec.page_index, lines=0, ink_pixels=0, error=str(exc))
            )
            continue

        page_lines = lines_from_bands(page, cfg.line)
        for line in page_lines:
            index = len(lines)
            origin[index] = (spec.page_index, line.index)
            lines.append(replace(line, index=index))

        strokes.append(page.stroke_width)
        diagnostics.pages.append(
            PageResult(
                source=name,
                page_index=spec.page_index,
                lines=len(page_lines),
                ink_pixels=int(page.ink.sum()),
            )
        )

    diagnostics.stroke_width = float(np.median(strokes)) if strokes else 1.0
    return lines, origin, diagnostics


def build_from_freeform(
    images: list[tuple[str, np.ndarray]],
    transcriber,
    cfg: Config | None = None,
    min_confidence: float = 0.35,
) -> PipelineResult:
    """Herhangi bir el yazısı sayfasından font üretir — şablon gerekmeden.

    Çalışma sayfası yolundan tek farkı, metnin nereden geldiğidir. Orada metni
    biz dayatıyorduk; burada sayfada ne yazdığını bir görsel model okuyor.
    Geri kalan her şey aynı deterministik zincirden geçer.

    Modelin hata payı boru hattının içinde soğurulur: okunan metin yanlışsa o
    kelimenin hizalama maliyeti yükselir ve glifleri atılır (bkz.
    `segment.drop_misaligned_words`). Yani model yanılabilir, font bozulmaz.
    """
    cfg = cfg or Config()
    diagnostics = Diagnostics()

    lines: list[Line] = []
    origin: dict[int, tuple[int, int]] = {}
    strokes: list[float] = []

    for page_index, (name, image) in enumerate(images):
        page = prepare_page(image, None, cfg.preprocess)
        page_lines = detect_lines(page, cfg.line)
        for line in page_lines:
            index = len(lines)
            origin[index] = (page_index, line.index)
            lines.append(replace(line, index=index))
        strokes.append(page.stroke_width)
        diagnostics.pages.append(
            PageResult(
                source=name,
                page_index=page_index,
                lines=len(page_lines),
                ink_pixels=int(page.ink.sum()),
            )
        )

    if not lines:
        raise ValueError(
            "Sayfalarda yazı satırı bulunamadı. Fotoğrafın net ve yazının "
            "kağıttan belirgin şekilde koyu olduğundan emin olun."
        )

    diagnostics.stroke_width = float(np.median(strokes)) if strokes else 1.0

    # Okuma, eğim düzeltmesinden *önce* yapılır: model yazıyı doğal haliyle
    # daha iyi okur, dikleştirilmiş hali ona tanıdık gelmez.
    readings = transcriber.transcribe_lines([line.ink for line in lines])
    diagnostics.transcriptions = [(r.index, r.text, r.confidence) for r in readings]

    usable: list[Line] = []
    for line, reading in zip(lines, readings):
        if not reading.usable or reading.confidence < min_confidence:
            diagnostics.rejected_lines[origin.get(line.index, (0, line.index))] = (
                "satır okunamadı" if not reading.usable
                else f"okuma güveni düşük ({reading.confidence:.2f})"
            )
            continue
        line.text = reading.text
        usable.append(line)

    if not usable:
        raise ValueError("Hiçbir satır okunamadı; fotoğrafın okunaklı olduğundan emin olun.")

    return _finish(usable, origin, diagnostics, cfg)


def build_from_images(
    images: list[tuple[str, np.ndarray]],
    sheets: SheetSet,
    cfg: Config | None = None,
) -> PipelineResult:
    """Doldurulmuş çalışma sayfası fotoğraflarından font üretir."""
    cfg = cfg or Config()

    lines, origin, diagnostics = collect_lines(images, sheets, cfg)
    if not lines:
        raise ValueError(
            "Hiçbir sayfadan satır çıkarılamadı. Fotoğraflarda köşe işaretleri "
            "görünüyor mu ve yazı kılavuz çizgilerinin üstünde mi?"
        )
    return _finish(lines, origin, diagnostics, cfg)


def _finish(
    lines: list[Line],
    origin: dict[int, tuple[int, int]],
    diagnostics: Diagnostics,
    cfg: Config,
) -> PipelineResult:
    """Satırlardan fonta giden ortak kuyruk.

    Metnin nereden geldiği (şablondan mı, modelden mi) buradan itibaren önemsiz
    olur; iki yol da aynı deterministik zinciri kullanır.
    """
    # Eğim bütün sayfalardan ortak hesaplanır: aynı elin yazısı olduğu için
    # sayfa başına ayrı düzeltmek tutarsızlık üretirdi.
    diagnostics.slant = page_slant(lines, limit=cfg.line.slant_limit, coarse=cfg.line.slant_step)
    straight = [deslant_line(line, diagnostics.slant) for line in lines]

    segmentation = segment_document(straight, diagnostics.stroke_width, cfg.segment)
    diagnostics.rejected_lines.update(
        {origin.get(index, (-1, index)): reason for index, reason in segmentation.rejected.items()}
    )
    diagnostics.dropped_glyphs = segmentation.dropped_words

    # x-yüksekliği, segmentasyondan sonra gerçek gliflerden ölçülür; satırın
    # mürekkep profilinden yapılan ilk tahmin salt büyük harfli satırlarda
    # yanılır ve o satırların glifleri küçük kalırdı.
    measured, document_xheight = measure_xheights(segmentation.boxes)
    apply_xheights(segmentation.boxes, measured, document_xheight)
    diagnostics.xheight_px = document_xheight

    glyphs = normalize_all(segmentation.boxes, cfg.font)
    diagnostics.total_samples = len(glyphs)

    space = _measure_space(segmentation, measured, document_xheight, cfg)
    library = build_library(glyphs, cfg.glyph, cfg.font, space_advance=space)

    from .config import CHARSET

    present = set(library.characters)
    diagnostics.missing_characters = [ch for ch in CHARSET if ch not in present]
    diagnostics.weak_characters = [
        (char, entry.seen, entry.kept)
        for char, entry in sorted(library.characters.items())
        if entry.kept < 3
    ]

    font, build = build_font(library, cfg.font, cfg.glyph)
    return PipelineResult(
        font=font,
        library=library,
        build=build,
        diagnostics=diagnostics,
        segmentation=segmentation,
    )


def _measure_space(
    segmentation: DocumentSegmentation,
    xheights: dict[int, float],
    document_xheight: float,
    cfg: Config,
) -> float | None:
    """Boşluk karakterinin genişliğini ölçülen değerlerden em birimine çevirir.

    Boşluk, kişinin yazı ritminin en görünür parçalarından biri: kimi sıkışık
    kimi ferah yazar. Varsayılan bir değer koymak yerine ölçmek, fontla yazılan
    metnin el yazısına benzemesini belirgin şekilde artırır.

    Ölçek olarak gliflerin normalize edildiği x-yüksekliğinin *aynısı*
    kullanılır; başka bir ölçek boşlukları harflerle tutarsız hale getirirdi.
    """
    values: list[float] = []
    for index, scale in segmentation.scales.items():
        if scale <= 0 or index in segmentation.rejected:
            continue
        xheight = xheights.get(index, document_xheight)
        if xheight <= 0:
            continue
        space_px = segmentation.priors.space * scale
        values.append(space_px * (cfg.font.x_height / xheight))
    return float(np.median(values)) if values else None


# --------------------------------------------------------------------------
# Dosya tabanlı kolaylık sarmalayıcıları
# --------------------------------------------------------------------------


def build_from_paths(
    paths: list[str | Path],
    sheets_path: str | Path,
    cfg: Config | None = None,
) -> PipelineResult:
    """Dosya yollarından font üretir."""
    sheets = SheetSet.load(sheets_path)
    images = [(str(Path(p).name), load_gray(p)) for p in paths]
    return build_from_images(images, sheets, cfg)


def write_font(result: PipelineResult, path: str | Path) -> Path:
    save_font(result.font, path)
    return Path(path)
