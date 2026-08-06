"""Sentetik el yazısı üreteci — boru hattını doğrulamak için.

Gerçek bir el yazısı örneği olmadan segmentasyonun doğru çalıştığını iddia
edemeyiz. Bu modül, doldurulmuş bir çalışma sayfasını *ve her karakterin gerçek
konumunu* üretir. Böylece "gözle bakınca iyi görünüyor" yerine ölçülebilir bir
doğruluk sayısı elde ederiz.

Üretilen yazı bilerek zorlaştırılmıştır:

* harfler arası bağlantılar rastgele kurulur — yani bir bağlantılı bileşen
  birden çok karakter içerebilir (karışık bitişik/ayrık yazı),
* her glif kendi başına döndürülüp ölçeklenir ve elastik olarak bozulur,
* taban çizgisi satır boyunca yavaşça kayar,
* kalem kalınlığı değişir.

Ardından `simulate_camera` sayfayı bir telefon fotoğrafına dönüştürür:
perspektif, düzensiz aydınlatma, bulanıklık ve gürültü.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .template import Band, SheetSpec, find_reference_font, mm_to_px, render_template


@dataclass
class HandStyle:
    """Sentetik el yazısının karakterini belirleyen parametreler."""

    #: Genel eğim (derece). Gerçek el yazısında kişiye özgü ve tutarlıdır.
    slant_deg: float = 9.0
    #: Basılı x-yüksekliğine göre yazı boyutu.
    #:
    #: DejaVu Sans, aynı x-yüksekliğindeki tipik el yazısından belirgin şekilde
    #: geniştir; 1.0'da sentetik satırlar yazı alanına sığmaz ve gerçekte
    #: olmayan bir "satırı bitirememe" sorunu üretir. 0.85 ile ortalama harf
    #: ilerlemesi ~4,3 mm olur, bu da 5 mm x-yüksekliğindeki gerçek el yazısının
    #: ölçüsüdür.
    size_ratio: float = 0.85
    #: Kalem kalınlığı, x-yüksekliğinin oranı.
    pen_ratio: float = 0.075
    #: Kalem kalınlığındaki rastgele değişim oranı.
    pen_jitter: float = 0.30
    #: Glif başına döndürme genliği (derece).
    rotate_deg: float = 3.5
    #: Glif başına ölçek değişimi.
    scale_jitter: float = 0.07
    #: Glif başına dikey kayma, x-yüksekliğinin oranı.
    baseline_jitter: float = 0.05
    #: Harf ilerlemesindeki rastgele değişim oranı.
    advance_jitter: float = 0.11
    #: Elastik bozulma genliği, x-yüksekliğinin oranı.
    elastic: float = 0.06
    #: İki komşu harfin birleştirilme olasılığı.
    connect_prob: float = 0.45
    #: Satır boyunca taban çizgisi kayması, x-yüksekliğinin oranı.
    drift_ratio: float = 0.10
    #: Kelime arası boşluğun x-yüksekliğine oranı.
    space_ratio: float = 0.75
    #: Mürekkep koyuluğu (0 = siyah).
    ink_level: int = 42
    #: Yazı tipi dosyası; None ise sistemden bulunur.
    font_path: str | None = None


@dataclass
class SynthChar:
    """Sentezlenen tek bir karakterin gerçek konumu (kanonik sayfa koordinatı)."""

    band: int
    index: int
    char: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class SynthPage:
    """Sentezlenmiş bir sayfa ve gerçek konum bilgisi."""

    #: Kamera bozulması uygulanmış "fotoğraf".
    photo: np.ndarray
    #: Bozulmamış kanonik sayfa (referans / hata ayıklama).
    canonical: np.ndarray
    #: Yalnızca el yazısı mürekkebi (kanonik koordinat, referans).
    ink: np.ndarray
    truth: list[SynthChar] = field(default_factory=list)
    #: Piksel düzeyinde etiket haritası: her mürekkep pikselinde onu çizen
    #: karakterin `truth` listesindeki indeksi, mürekkepsiz ve bağlantı
    #: çizgilerinde -1.
    #:
    #: Segmentasyon doğruluğunu kutu merkezleri üzerinden değil piksel
    #: üzerinden ölçmeyi sağlar: "bu glife atanan mürekkebin yüzde kaçı
    #: gerçekten o harfe ait?" sorusunun kesin cevabı buradan çıkar.
    labels: np.ndarray | None = None
    #: Serbest sayfalarda üretilen yerleşim (gerçek taban çizgileri burada).
    spec: SheetSpec | None = None


# --------------------------------------------------------------------------
# Glif üretimi
# --------------------------------------------------------------------------


def _measure_xheight(font) -> float:
    """Yazı tipinin x-yüksekliğini ölçer."""
    bbox = font.getbbox("x", anchor="ls")
    return abs(bbox[1])


@lru_cache(maxsize=64)
def _font_for_xheight(target_xheight: float, path: str | None) -> tuple:
    """İstenen x-yüksekliğini verecek yazı tipi boyutunu bulur.

    Boyut araması yineleyicidir ve her bantta aynı sonucu verir; önbelleğe
    alınmazsa sentez süresinin çoğunu bu fonksiyon yer.
    """
    from PIL import ImageFont

    size = max(8, int(target_xheight * 2.0))
    for _ in range(12):
        font = ImageFont.truetype(path, size) if path else find_reference_font(size)
        measured = _measure_xheight(font)
        if measured <= 0:
            break
        ratio = target_xheight / measured
        if abs(ratio - 1.0) < 0.02:
            return font, measured
        size = max(8, int(round(size * ratio)))
    font = ImageFont.truetype(path, size) if path else find_reference_font(size)
    return font, _measure_xheight(font)


def _elastic(mask: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Düşük frekanslı rastgele yer değiştirme alanıyla organik bozulma uygular."""
    if amplitude <= 0.15:
        return mask
    h, w = mask.shape
    sigma = max(3.0, min(h, w) / 4.0)
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    # Bulanıklaştırma genliği düşürür; birim varyansa geri ölçekle.
    for field_ in (dx, dy):
        std = field_.std()
        if std > 1e-6:
            field_ *= 1.0 / std
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + dx * amplitude
    map_y = grid_y + dy * amplitude
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _render_glyph(
    ch: str,
    font,
    style: HandStyle,
    xheight: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Tek bir karakteri bozulmuş halde üretir.

    Döndürülen maskede taban-sol köşe (origin) `origin` konumundadır; böylece
    yerleştirme sırasında taban çizgisi hizalaması kesin olur.
    """
    pad = int(max(xheight * 2.5, 24))
    canvas = Image.new("L", (pad * 3, pad * 4), 0)
    draw = ImageDraw.Draw(canvas)
    origin = (pad, pad * 2)
    draw.text(origin, ch, font=font, fill=255, anchor="ls")
    mask = np.asarray(canvas)

    # Origin'i sabit tutan afin dönüşüm: eğim (shear) + döndürme + ölçek.
    # Eğim taban çizgisi etrafında uygulanır, böylece taban çizgisindeki
    # noktalar yerinde kalır ve yatay konumlar korunur.
    slant = math.tan(math.radians(style.slant_deg))
    angle = math.radians(rng.uniform(-style.rotate_deg, style.rotate_deg))
    scale = 1.0 + rng.uniform(-style.scale_jitter, style.scale_jitter)

    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    # [rotate+scale] · [shear]
    a11, a12 = cos_a, -sin_a - cos_a * slant
    a21, a22 = sin_a, cos_a - sin_a * slant
    ox, oy = origin
    matrix = np.array(
        [
            [a11, a12, ox - a11 * ox - a12 * oy],
            [a21, a22, oy - a21 * ox - a22 * oy],
        ],
        dtype=np.float32,
    )
    mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
    mask = _elastic(mask, style.elastic * xheight, rng)
    return mask, origin


# --------------------------------------------------------------------------
# Bant ve sayfa üretimi
# --------------------------------------------------------------------------


def _render_band(
    text: str,
    band,
    style: HandStyle,
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    band_index: int,
    label_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[SynthChar]]:
    """Bir bandın el yazısını, etiket haritasını ve gerçek konumları üretir."""
    printed_xheight = float(band.baseline_y - band.xheight_y)
    xheight = printed_xheight * style.size_ratio
    font, _ = _font_for_xheight(round(xheight, 1), style.font_path)

    # Kalem, her glife ayrı ayrı uygulanır. Tek seferde tüm bandı şişirmek daha
    # hızlı olurdu ama o zaman etiket haritası kalemin eklediği pikselleri
    # kapsamazdı ve ölçüm bozulurdu.
    pen = max(1, int(round(style.pen_ratio * xheight)))
    pen_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pen, pen)) if pen > 1 else None

    layer = np.zeros(canvas_shape, np.uint8)
    labels = np.full(canvas_shape, -1, np.int32)
    truth: list[SynthChar] = []

    x0_limit, _, x1_limit, _ = band.write_rect
    cursor = float(x0_limit) + rng.uniform(0.0, xheight * 0.4)
    width_available = x1_limit - x0_limit

    # Taban çizgisi satır boyunca yavaşça kayar; gerçek el yazısında olduğu gibi.
    drift_amp = style.drift_ratio * xheight
    drift_phase = rng.uniform(0, 2 * math.pi)

    placed: list[tuple[int, int, int]] = []  # (char_index, x_right, baseline_y)

    for i, ch in enumerate(text):
        progress = (cursor - x0_limit) / max(width_available, 1.0)
        baseline = band.baseline_y + drift_amp * math.sin(drift_phase + progress * 2.2)
        baseline += rng.uniform(-style.baseline_jitter, style.baseline_jitter) * xheight
        baseline_i = int(round(baseline))

        if ch == " ":
            cursor += xheight * style.space_ratio * (1.0 + rng.uniform(-0.2, 0.2))
            placed.append((i, -1, baseline_i))
            continue

        glyph, origin = _render_glyph(ch, font, style, xheight, rng)
        if pen_element is not None:
            glyph = cv2.dilate(glyph, pen_element)

        ys, xs = np.nonzero(glyph > 127)
        if xs.size == 0:
            cursor += xheight * 0.5
            continue

        # Glif maskesini, origin'i (cursor, baseline) noktasına gelecek şekilde
        # sayfaya bindir.
        dx = int(round(cursor)) - origin[0]
        dy = baseline_i - origin[1]
        gx0, gx1 = xs.min() + dx, xs.max() + dx
        gy0, gy1 = ys.min() + dy, ys.max() + dy

        if gx1 >= x1_limit:
            break  # satıra sığmadı

        _blit_max(layer, glyph, dx, dy)
        _blit_label(labels, glyph > 127, dx, dy, label_base + len(truth))
        truth.append(
            SynthChar(band=band_index, index=i, char=ch, x0=int(gx0), y0=int(gy0), x1=int(gx1), y1=int(gy1))
        )
        placed.append((i, int(gx1), baseline_i))

        advance = font.getlength(ch) * (1.0 + rng.uniform(-style.advance_jitter, style.advance_jitter))
        cursor += advance

    # Bağlantı çizgileri hiçbir karaktere ait değildir; etiketsiz (-1) kalırlar
    # ve ölçümde ne lehte ne aleyhte sayılırlar.
    _draw_connections(layer, truth, placed, style, xheight, rng, pen)
    return layer, labels, truth


def _blit_max(target: np.ndarray, patch: np.ndarray, dx: int, dy: int) -> None:
    """`patch`i `target`a maksimum karışımıyla yerleştirir (sınır güvenli)."""
    th, tw = target.shape
    ph, pw = patch.shape
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(tw, dx + pw), min(th, dy + ph)
    if x0 >= x1 or y0 >= y1:
        return
    sub = patch[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    np.maximum(target[y0:y1, x0:x1], sub, out=target[y0:y1, x0:x1])


def _blit_label(
    labels: np.ndarray, mask: np.ndarray, dx: int, dy: int, value: int
) -> None:
    """Etiket haritasına bir glifin kimliğini yazar.

    İki glif çakışırsa o pikseller -2 ("çekişmeli") olarak işaretlenir. O
    piksellerin hangi harfe ait olduğu gerçekten belirsizdir; ölçümde saymamak,
    segmentasyonu olmadığı bir şeyden sorumlu tutmamak için doğru olanıdır.
    """
    th, tw = labels.shape
    ph, pw = mask.shape
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(tw, dx + pw), min(th, dy + ph)
    if x0 >= x1 or y0 >= y1:
        return
    sub = mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    window = labels[y0:y1, x0:x1]
    window[sub & (window == -1)] = value
    window[sub & (window >= 0) & (window != value)] = -2


def _draw_connections(
    layer: np.ndarray,
    truth: list[SynthChar],
    placed: list[tuple[int, int, int]],
    style: HandStyle,
    xheight: float,
    rng: np.random.Generator,
    pen: int = 1,
) -> None:
    """Komşu harfleri rastgele birleştirerek bitişik yazı taklidi yapar.

    Bu, segmentasyon için asıl zor durumu üretir: tek bir bağlantılı bileşen
    içinde birden çok karakter.
    """
    by_index = {c.index: c for c in truth}
    baseline_of = {idx: base for idx, _, base in placed}

    for current, following in zip(truth, truth[1:]):
        if following.index != current.index + 1:
            continue  # araya boşluk girmiş
        if rng.random() > style.connect_prob:
            continue
        gap_start, gap_end = current.x1, following.x0
        if gap_end - gap_start > xheight * 0.9:
            continue  # çok uzak, gerçekçi olmaz
        base = baseline_of.get(current.index, current.y1)
        # Taban çizgisi civarında hafif çukur yapan bir bağlantı yayı.
        mid_x = (gap_start + gap_end) / 2.0
        dip = rng.uniform(0.02, 0.16) * xheight
        points = np.array(
            [
                [gap_start, base - rng.uniform(0.0, 0.12) * xheight],
                [mid_x, base + dip],
                [gap_end, base - rng.uniform(0.0, 0.12) * xheight],
            ],
            dtype=np.float32,
        )
        curve = _quadratic_curve(points, samples=14)
        cv2.polylines(layer, [curve.astype(np.int32)], False, 255, max(1, pen), cv2.LINE_AA)


def _quadratic_curve(control: np.ndarray, samples: int) -> np.ndarray:
    """Üç kontrol noktasından kuadratik Bézier örnekler."""
    t = np.linspace(0.0, 1.0, samples)[:, None]
    p0, p1, p2 = control[0], control[1], control[2]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def synthesize_sheet(
    spec: SheetSpec,
    style: HandStyle | None = None,
    seed: int = 0,
) -> SynthPage:
    """Bir çalışma sayfasını sentetik el yazısıyla doldurur."""
    style = style or HandStyle()
    rng = np.random.default_rng(seed)

    printed = render_template(spec)
    shape = (spec.height, spec.width)

    ink = np.zeros(shape, np.uint8)
    labels = np.full(shape, -1, np.int32)
    truth: list[SynthChar] = []
    for band in spec.bands:
        layer, band_labels, band_truth = _render_band(
            band.text, band, style, rng, shape, band.index, label_base=len(truth)
        )
        np.maximum(ink, layer, out=ink)
        np.copyto(labels, band_labels, where=(band_labels != -1) & (labels == -1))
        truth.extend(band_truth)

    # Mürekkebi basılı sayfanın üstüne koy. Kalem izi basılı içerikten koyudur.
    alpha = ink.astype(np.float32) / 255.0
    ink_level = style.ink_level + rng.uniform(-8, 8)
    canonical = printed.astype(np.float32) * (1 - alpha) + ink_level * alpha
    canonical = np.clip(canonical, 0, 255).astype(np.uint8)

    return SynthPage(
        photo=canonical, canonical=canonical, ink=ink > 127, truth=truth, labels=labels
    )


# --------------------------------------------------------------------------
# Kamera simülasyonu
# --------------------------------------------------------------------------


def simulate_camera(
    page: np.ndarray,
    seed: int = 0,
    warp: float = 0.020,
    rotation: float = 2.5,
    blur: float = 1.0,
    noise: float = 4.0,
    vignette: float = 0.28,
    margin: float = 0.06,
) -> np.ndarray:
    """Kanonik sayfayı bir telefon fotoğrafına benzetir."""
    rng = np.random.default_rng(seed + 9973)
    h, w = page.shape

    pad_x, pad_y = int(w * margin), int(h * margin)
    canvas_w, canvas_h = w + 2 * pad_x, h + 2 * pad_y

    # Masa arkaplanı: hafif dokulu, kağıttan koyu.
    desk = rng.normal(168, 7, (canvas_h, canvas_w)).astype(np.float32)
    desk = cv2.GaussianBlur(desk, (0, 0), 3.0)

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    jitter = np.array(
        [[rng.uniform(-warp, warp) * w, rng.uniform(-warp, warp) * h] for _ in range(4)], np.float32
    )
    dst = src + jitter + np.array([pad_x, pad_y], np.float32)

    angle = math.radians(rng.uniform(-rotation, rotation))
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rel = dst - np.array([cx, cy], np.float32)
    dst = np.stack(
        [rel[:, 0] * cos_a - rel[:, 1] * sin_a + cx, rel[:, 0] * sin_a + rel[:, 1] * cos_a + cy], axis=1
    ).astype(np.float32)

    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        page.astype(np.float32), homography, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_TRANSPARENT, dst=desk.copy(),
    )

    # Düzensiz aydınlatma: düşük frekanslı çarpımsal alan + köşe karartması.
    light = cv2.GaussianBlur(rng.standard_normal((canvas_h, canvas_w)).astype(np.float32), (0, 0),
                             min(canvas_h, canvas_w) / 6.0)
    std = light.std()
    light = light / std if std > 1e-6 else light
    gain = 1.0 + 0.10 * light

    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    gain *= 1.0 - vignette * np.clip(radius / 1.45, 0, 1) ** 2

    out = warped * gain
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise > 0:
        out = out + rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthesize_photo(
    spec: SheetSpec,
    style: HandStyle | None = None,
    seed: int = 0,
    **camera,
) -> SynthPage:
    """Doldurulmuş sayfayı üretip kamera bozulması uygular."""
    page = synthesize_sheet(spec, style, seed)
    page.photo = simulate_camera(page.canonical, seed=seed, **camera)
    return page


# --------------------------------------------------------------------------
# Serbest sayfa: şablonsuz, sıradan (çizgili) kağıt
# --------------------------------------------------------------------------

#: Çizgili defterlerdeki basılı çizginin tipik tonu.
RULE_LEVEL = 178
MARGIN_RULE_LEVEL = 196


def freeform_layout(
    texts: list[str],
    dpi: int = 200,
    xheight_mm: float = 5.0,
    line_spacing_mm: float = 9.0,
    top_mm: float = 22.0,
    left_mm: float = 20.0,
    right_mm: float = 14.0,
    page_mm: tuple[float, float] = (210.0, 297.0),
) -> tuple[SheetSpec, list[Band]]:
    """Sıradan bir defter sayfasının geometrisini üretir (işaretsiz).

    Çalışma sayfasıyla aynı `Band` yapısını kullanır; böylece el yazısı çizimi
    tek bir kod yolundan geçer. Fark şu: bu sayfada ne köşe işareti ne basılı
    örnek metin vardır — boru hattının serbest yolu tam olarak bunu görmeli.
    """
    width = int(round(mm_to_px(page_mm[0], dpi)))
    height = int(round(mm_to_px(page_mm[1], dpi)))
    xheight = mm_to_px(xheight_mm, dpi)
    spacing = mm_to_px(line_spacing_mm, dpi)
    left = int(round(mm_to_px(left_mm, dpi)))
    right = width - int(round(mm_to_px(right_mm, dpi)))

    bands: list[Band] = []
    for index, text in enumerate(texts):
        baseline = int(round(mm_to_px(top_mm, dpi) + (index + 1) * spacing))
        if baseline + xheight > height:
            break
        bands.append(
            Band(
                index=index,
                text=text,
                ref_rect=(0, 0, 0, 0),  # basılı örnek metin yok
                write_rect=(left, int(baseline - 2.2 * xheight), right, int(baseline + xheight)),
                ascender_y=int(baseline - 1.8 * xheight),
                xheight_y=int(baseline - xheight),
                baseline_y=baseline,
                descender_y=int(baseline + 0.8 * xheight),
            )
        )

    spec = SheetSpec(
        page_index=0,
        width=width,
        height=height,
        dpi=dpi,
        marker_ids=[],
        marker_corners=[],
        bands=bands,
    )
    return spec, bands


def synthesize_freeform(
    texts: list[str],
    style: HandStyle | None = None,
    seed: int = 0,
    ruled: bool = True,
    margin_rule: bool = True,
    dpi: int = 200,
    **layout,
) -> SynthPage:
    """Sıradan bir deftere yazılmış gibi bir sayfa üretir.

    `ruled=True` ile basılı defter çizgileri de çizilir. Bu, serbest yolun asıl
    zorluğudur: çizgiler mürekkep gibi görünür, temizlenmezse satır tespitini ve
    segmentasyonu bozarlar. Şablon modunda çizgilerin yerini biliyorduk, burada
    bilmiyoruz.
    """
    style = style or HandStyle()
    rng = np.random.default_rng(seed)
    spec, bands = freeform_layout(texts, dpi=dpi, **layout)

    paper = np.full((spec.height, spec.width), 249, np.uint8)
    if ruled:
        canvas = Image.fromarray(paper)
        draw = ImageDraw.Draw(canvas)
        x0 = int(round(mm_to_px(12.0, dpi)))
        x1 = spec.width - int(round(mm_to_px(10.0, dpi)))
        for band in bands:
            draw.line([(x0, band.baseline_y), (x1, band.baseline_y)], fill=RULE_LEVEL, width=1)
        if margin_rule and bands:
            margin_x = bands[0].write_rect[0] - int(round(mm_to_px(4.0, dpi)))
            draw.line(
                [(margin_x, int(mm_to_px(10.0, dpi))), (margin_x, spec.height - int(mm_to_px(10.0, dpi)))],
                fill=MARGIN_RULE_LEVEL,
                width=1,
            )
        paper = np.asarray(canvas)

    shape = (spec.height, spec.width)
    ink = np.zeros(shape, np.uint8)
    labels = np.full(shape, -1, np.int32)
    truth: list[SynthChar] = []
    for band in bands:
        layer, band_labels, band_truth = _render_band(
            band.text, band, style, rng, shape, band.index, label_base=len(truth)
        )
        np.maximum(ink, layer, out=ink)
        np.copyto(labels, band_labels, where=(band_labels != -1) & (labels == -1))
        truth.extend(band_truth)

    alpha = ink.astype(np.float32) / 255.0
    level = style.ink_level + rng.uniform(-8, 8)
    canonical = np.clip(paper.astype(np.float32) * (1 - alpha) + level * alpha, 0, 255).astype(np.uint8)

    page = SynthPage(
        photo=canonical, canonical=canonical, ink=ink > 127, truth=truth, labels=labels
    )
    page.spec = spec
    return page
