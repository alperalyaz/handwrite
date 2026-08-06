"""Eksik gliflerin üretilmesi.

Tek bir doğal sayfa, küçük harfleri bol bol verir ama büyük harflerin çoğunu,
rakamları ve noktalamayı vermez: düzyazıda "Q" da geçmez, "%" de. Ölçtüğümüz
kadarıyla 500 karakterlik bir sayfadan sonra karakter kümesinin yarısı hâlâ
eksik kalıyor. Kullanıcıdan ikinci bir sayfa istemek yerine bu boşluğu
dolduruyoruz.

Üç kademe, güvenilirlik sırasıyla denenir:

1. **Bileşim** — eksik karakter, kullanıcının *gerçek* kalem izlerinden monte
   edilir: Ç = C + ç'nin çengeli, Ğ = G + ğ'nin şapkası, İ = I + i'nin noktası.
   Her parça gerçekten o elden çıktığı için sonuç kusursuza yakındır.

2. **Büyük/küçük harf aktarımı** — bazı harflerin büyüğü küçüğünün büyütülmüş
   hâlidir (c/C, o/O, s/S, v/V...). Kullanıcının kendi harfi cap-height'a
   ölçeklenir. "a/A" ya da "e/E" gibi biçimi gerçekten farklı olanlar bu
   listede yoktur.

3. **Referans iskelet + kullanıcının kalemi** — geri kalanı için bir referans
   yazı tipinin harf biçimi alınır, kullanıcının ölçülen kalem kalınlığı,
   oranları ve el titremesi uygulanır. Sonuç aynı fontun parçası gibi durur ama
   dikkatli bir göz bunun kullanıcının eli olmadığını anlar.

Üretilen glifler `synthetic=True` ile işaretlenir; teşhis ekranı hangi
karakterlerin gerçek hangilerinin üretilmiş olduğunu söyler. Kullanıcıya
üretilmiş bir harfi kendi yazısıymış gibi göstermek doğru olmaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CHARSET, FontConfig, GlyphConfig
from .glyph import CharacterSet, Glyph, GlyphLibrary
from .template import find_reference_font

#: Bileşim tarifleri: eksik karakter → (taban karakter, aksan bağışçısı).
COMPOSITION: dict[str, tuple[str, str]] = {
    "Ç": ("C", "ç"),
    "Ğ": ("G", "ğ"),
    "İ": ("I", "i"),
    "Ö": ("O", "ö"),
    "Ü": ("U", "ü"),
    "Ş": ("S", "ş"),
    "ç": ("c", "ş"),
    "ş": ("s", "ç"),
    "ö": ("o", "ü"),
    "ü": ("u", "ö"),
    "ğ": ("g", "ö"),
    "İ".lower(): ("ı", "i"),
}

#: Büyük hâli küçüğünün ölçeklenmişi sayılabilecek harfler. Biçimi gerçekten
#: değişenler (a/A, e/E, g/G, ...) bilerek dışarıda bırakıldı.
CASE_COMPATIBLE: dict[str, str] = {
    "C": "c", "O": "o", "S": "s", "U": "u", "V": "v", "W": "w",
    "X": "x", "Z": "z", "K": "k", "P": "p", "Y": "y", "I": "ı",
    "Ç": "ç", "Ö": "ö", "Ü": "ü", "Ş": "ş",
}

#: Sentez sırasında çalışılan kanonik çözünürlük: x-yüksekliği bu kadar piksel.
CANONICAL_XHEIGHT = 100


@dataclass
class StyleProfile:
    """Kullanıcının yazısından ölçülen, sentezde kullanılacak biçim özellikleri."""

    #: Kanonik ölçekte kalem kalınlığı (piksel).
    stroke: float
    #: cap-height / x-height oranı.
    cap_ratio: float
    #: ascender / x-height oranı.
    ascender_ratio: float
    #: Ortalama harf ilerlemesi / x-height.
    advance_ratio: float
    #: Yan boşluk / x-height.
    bearing_ratio: float

    @property
    def cap_height(self) -> float:
        return CANONICAL_XHEIGHT * self.cap_ratio


def measure_style(library: GlyphLibrary, font: FontConfig) -> StyleProfile:
    """Kullanıcının gerçek gliflerinden biçim özelliklerini ölçer."""
    reals = [g for entry in library.characters.values() for g in entry.variants]
    if not reals:
        return StyleProfile(stroke=8.0, cap_ratio=1.4, ascender_ratio=1.45,
                            advance_ratio=1.0, bearing_ratio=0.08)

    # Kalem kalınlığı: mesafe dönüşümünün tepe değeri, em uzayına taşınır.
    widths = []
    for glyph in reals[:60]:
        if glyph.mask.size == 0 or not glyph.mask.any():
            continue
        distance = cv2.distanceTransform(glyph.mask.astype(np.uint8), cv2.DIST_L2, 5)
        peaks = distance[distance > 0]
        if peaks.size:
            # piksel → em → kanonik
            em = float(np.percentile(peaks, 75)) * 2.0 * glyph.scale
            widths.append(em / font.x_height * CANONICAL_XHEIGHT)
    stroke = float(np.median(widths)) if widths else 8.0

    def ratio(chars: str, fallback: float) -> float:
        values = [
            (g.ascent + g.descent) / font.x_height
            for ch in chars
            if ch in library.characters
            for g in library.characters[ch].variants
        ]
        return float(np.median(values)) if values else fallback

    cap_ratio = ratio("ABDEFHKLMNPRTİI", font.cap_height / font.x_height)
    ascender_ratio = ratio("bdfhklt", 1.45)

    advances = [entry.advance for entry in library.characters.values()]
    bearings = [entry.bearing for entry in library.characters.values()]

    return StyleProfile(
        stroke=max(2.0, stroke),
        cap_ratio=max(1.05, cap_ratio),
        ascender_ratio=max(1.05, ascender_ratio),
        advance_ratio=float(np.median(advances)) / font.x_height if advances else 1.0,
        bearing_ratio=float(np.median(bearings)) / font.x_height if bearings else 0.08,
    )


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------


def _to_canonical(glyph: Glyph, font: FontConfig) -> tuple[np.ndarray, float, float]:
    """Bir glifi kanonik ölçeğe taşır (x-yüksekliği = CANONICAL_XHEIGHT).

    Döndürür: (maske, taban çizgisinin maske üstünden uzaklığı, sol boşluk).
    """
    factor = glyph.scale / font.x_height * CANONICAL_XHEIGHT
    height = max(1, int(round(glyph.mask.shape[0] * factor)))
    width = max(1, int(round(glyph.mask.shape[1] * factor)))
    mask = cv2.resize(glyph.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA) > 0.4
    return mask, glyph.ascent_px * factor, glyph.bearing_px * factor


def _stroke_width(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = distance[distance > 0]
    return float(np.percentile(peaks, 75)) * 2.0 if peaks.size else 1.0


def _match_weight(mask: np.ndarray, target: float) -> np.ndarray:
    """Maskenin kalem kalınlığını hedefe yaklaştırır."""
    current = _stroke_width(mask)
    delta = target - current
    radius = int(round(abs(delta) / 2.0))
    if radius < 1:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    operation = cv2.dilate if delta > 0 else cv2.erode
    return operation(mask.astype(np.uint8), element) > 0


def _humanize(mask: np.ndarray, stroke: float, seed: int) -> np.ndarray:
    """Matbaa harfine el yazısı düzensizliği katar.

    İki etki: köşelerin yuvarlanması (kalem keskin köşe çizmez) ve düşük
    frekanslı elastik bozulma (el titremesi). İkisi de hafif tutulur; abartmak
    harfi tanınmaz yapar.
    """
    rng = np.random.default_rng(seed)
    radius = max(1, int(round(stroke * 0.18)))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    rounded = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, element)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_OPEN, element)

    height, width = rounded.shape
    amplitude = stroke * 0.22
    sigma = max(4.0, min(height, width) / 3.5)
    field_x = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigma)
    field_y = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigma)
    for field in (field_x, field_y):
        deviation = field.std()
        if deviation > 1e-6:
            field *= 1.0 / deviation

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    warped = cv2.remap(
        rounded,
        grid_x + field_x * amplitude,
        grid_y + field_y * amplitude,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _as_glyph(
    char: str, mask: np.ndarray, ascent: float, bearing: float, advance: float, font: FontConfig
) -> Glyph | None:
    """Kanonik ölçekteki bir maskeyi Glyph nesnesine çevirir."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    tight = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return Glyph(
        char=char,
        mask=tight,
        scale=font.x_height / CANONICAL_XHEIGHT,
        bearing_px=bearing + float(cols[0]),
        ascent_px=ascent - float(rows[0]),
        advance_px=advance,
        line_index=-1,
        char_index=-1,
        synthetic=True,
    )


# --------------------------------------------------------------------------
# 1. kademe: bileşim
# --------------------------------------------------------------------------


def compose(char: str, library: GlyphLibrary, font: FontConfig, seed: int = 0) -> Glyph | None:
    """Eksik karakteri kullanıcının gerçek parçalarından monte eder."""
    recipe = COMPOSITION.get(char)
    if recipe is None:
        return None
    base_char, donor_char = recipe
    if base_char not in library.characters or donor_char not in library.characters:
        return None

    base_glyph = library.characters[base_char].variants[0]
    donor_glyph = library.characters[donor_char].variants[0]

    base, base_ascent, base_bearing = _to_canonical(base_glyph, font)
    donor, donor_ascent, _ = _to_canonical(donor_glyph, font)

    mark, mark_offset = _extract_mark(donor, donor_ascent)
    if mark is None:
        return None

    return _attach(char, base, base_ascent, base_bearing, mark, mark_offset,
                   library.characters[base_char].advance / font.x_height * CANONICAL_XHEIGHT, font)


def _extract_mark(donor: np.ndarray, ascent: float) -> tuple[np.ndarray | None, float]:
    """Bağışçı gliften aksan işaretini ayırır.

    İşaret, gövdeden ayrı duran ve x-yüksekliği bandının dışında kalan bağlantılı
    bileşendir: noktalar ve şapka üstte, çengel altta.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        donor.astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return None, 0.0

    body = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    marks = [i for i in range(1, count) if i != body]
    if not marks:
        return None, 0.0

    selected = np.isin(labels, marks)
    rows = np.flatnonzero(selected.any(axis=1))
    cols = np.flatnonzero(selected.any(axis=0))
    if rows.size == 0:
        return None, 0.0

    mark = selected[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    # İşaretin taban çizgisine göre dikey konumu; tabanın üstündeyse pozitif.
    return mark, ascent - float(rows[0])


def _attach(
    char: str,
    base: np.ndarray,
    base_ascent: float,
    base_bearing: float,
    mark: np.ndarray,
    mark_offset: float,
    advance: float,
    font: FontConfig,
) -> Glyph | None:
    """İşareti tabanın üstüne (ya da altına) yerleştirir."""
    base_rows = np.flatnonzero(base.any(axis=1))
    base_cols = np.flatnonzero(base.any(axis=0))
    if base_rows.size == 0:
        return None

    above = mark_offset > 0 and mark_offset > (base_ascent - base_rows[0]) * 0.6
    gap = max(2, int(round(CANONICAL_XHEIGHT * 0.10)))

    if above:
        mark_top = int(base_rows[0]) - gap - mark.shape[0]
    else:
        mark_top = int(base_rows[-1]) + gap
    mark_left = int((base_cols[0] + base_cols[-1]) / 2 - mark.shape[1] / 2)

    pad_top = max(0, -mark_top)
    pad_bottom = max(0, mark_top + mark.shape[0] - base.shape[0])
    pad_left = max(0, -mark_left)
    pad_right = max(0, mark_left + mark.shape[1] - base.shape[1])

    canvas = np.zeros(
        (base.shape[0] + pad_top + pad_bottom, base.shape[1] + pad_left + pad_right), dtype=bool
    )
    canvas[pad_top : pad_top + base.shape[0], pad_left : pad_left + base.shape[1]] = base

    row = mark_top + pad_top
    col = mark_left + pad_left
    canvas[row : row + mark.shape[0], col : col + mark.shape[1]] |= mark

    return _as_glyph(char, canvas, base_ascent + pad_top, base_bearing - pad_left, advance, font)


# --------------------------------------------------------------------------
# 2. kademe: büyük/küçük harf aktarımı
# --------------------------------------------------------------------------


def transfer_case(
    char: str, library: GlyphLibrary, style: StyleProfile, font: FontConfig, seed: int = 0
) -> Glyph | None:
    """Büyük harfi, kullanıcının küçük harfini cap-height'a büyüterek üretir."""
    source_char = CASE_COMPATIBLE.get(char)
    if source_char is None or source_char not in library.characters:
        return None

    source = library.characters[source_char].variants[0]
    mask, ascent, bearing = _to_canonical(source, font)

    factor = style.cap_ratio
    height = max(1, int(round(mask.shape[0] * factor)))
    width = max(1, int(round(mask.shape[1] * factor)))
    scaled = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR) > 0.4

    # Büyütmek kalem izini de kalınlaştırır; kullanıcının gerçek kalınlığına
    # geri çekilir, yoksa büyük harfler kalın görünür.
    scaled = _match_weight(scaled, style.stroke)
    scaled = _humanize(scaled, style.stroke, seed)

    advance = library.characters[source_char].advance / font.x_height * CANONICAL_XHEIGHT * factor
    return _as_glyph(char, scaled, ascent * factor, bearing * factor, advance, font)


# --------------------------------------------------------------------------
# 3. kademe: referans iskelet + kullanıcının kalemi
# --------------------------------------------------------------------------


def _reference_font(target: int = CANONICAL_XHEIGHT):
    """Referans yazı tipini, x-yüksekliği kanonik ölçeğe gelecek boyutta verir."""
    size = int(target * 2.2)
    reference = find_reference_font(size)
    measured = abs(reference.getbbox("x", anchor="ls")[1]) or target
    return find_reference_font(max(8, int(round(size * target / measured))))


def reference_width_correction(library: GlyphLibrary, font: FontConfig) -> float:
    """Referans fontun genişliklerini kullanıcının yazısına kalibre eder.

    Aynı x-yüksekliğinde matbaa harfleri el yazısından belirgin şekilde geniştir.
    Referansın ilerleme genişliğini olduğu gibi kullanmak, üretilen harflerin
    çevrelerinde kocaman boşluklar bırakmasına yol açar. Düzeltme katsayısı,
    kullanıcıda *gerçekten var olan* harflerin genişliklerini referansın aynı
    harflerdeki genişlikleriyle karşılaştırarak ölçülür.
    """
    try:
        reference = _reference_font()
    except RuntimeError:
        return 1.0

    ratios = []
    for char, entry in library.characters.items():
        if entry.synthetic or entry.advance <= 0:
            continue
        width = reference.getlength(char)
        if width and width > 1:
            user = entry.advance / font.x_height * CANONICAL_XHEIGHT
            ratios.append(user / width)
    return float(np.median(ratios)) if len(ratios) >= 5 else 1.0


def from_reference(
    char: str,
    style: StyleProfile,
    font: FontConfig,
    width_correction: float = 1.0,
    seed: int = 0,
) -> Glyph | None:
    """Referans yazı tipinin harf biçimini kullanıcının kalemiyle yeniden çizer."""
    from PIL import Image, ImageDraw

    target = CANONICAL_XHEIGHT
    try:
        reference = _reference_font(target)
    except RuntimeError:
        return None

    pad = int(target * 1.8)
    canvas = Image.new("L", (pad * 3, pad * 3), 0)
    ImageDraw.Draw(canvas).text((pad, pad * 2), char, font=reference, fill=255, anchor="ls")
    mask = np.asarray(canvas) > 127
    if not mask.any():
        return None

    mask = _match_weight(mask, style.stroke)
    mask = _humanize(mask, style.stroke, seed)
    if not mask.any():
        return None

    advance = (reference.getlength(char) or target * style.advance_ratio) * width_correction
    # Glif, orijini x=pad olan dolgulu bir tuvale çizildi. `_as_glyph` mürekkebin
    # sol kenarını ekleyeceği için dolgu payı burada geri alınmalı; alınmazsa
    # yan boşluk yüz piksel şişer ve harfler birbirinin üstüne biner.
    return _as_glyph(char, mask, float(pad * 2), float(-pad), float(advance), font)


# --------------------------------------------------------------------------
# Üst seviye
# --------------------------------------------------------------------------


@dataclass
class SynthesisReport:
    """Hangi karakterin nasıl üretildiği."""

    composed: list[str]
    transferred: list[str]
    referenced: list[str]
    failed: list[str]

    @property
    def total(self) -> int:
        return len(self.composed) + len(self.transferred) + len(self.referenced)

    def summary_lines(self) -> list[str]:
        rows = []
        if self.composed:
            rows.append("kendi parçalarından monte edildi: " + " ".join(self.composed))
        if self.transferred:
            rows.append("küçük harften büyütüldü: " + " ".join(self.transferred))
        if self.referenced:
            rows.append("referans biçim + sizin kaleminiz: " + " ".join(self.referenced))
        if self.failed:
            rows.append("üretilemedi: " + " ".join(self.failed))
        return rows


def fill_missing(
    library: GlyphLibrary,
    font: FontConfig,
    glyph_cfg: GlyphConfig,
    charset: str = CHARSET,
) -> SynthesisReport:
    """Eksik karakterleri üretip kütüphaneye ekler.

    Kademeler güvenilirlik sırasıyla denenir; ilk başarılı olan kullanılır.
    Kütüphane yerinde değiştirilir.
    """
    style = measure_style(library, font)
    correction = reference_width_correction(library, font)
    report = SynthesisReport([], [], [], [])

    # Bileşim, başka üretilmiş gliflere değil gerçek olanlara dayanmalı; bu
    # yüzden önce bütün bileşimler, sonra diğer kademeler yapılır.
    for char in charset:
        if char in library.characters:
            continue
        glyph = compose(char, library, font, seed=ord(char))
        if glyph is not None:
            library.characters[char] = _single(char, glyph, font)
            report.composed.append(char)

    for index, char in enumerate(charset):
        if char in library.characters:
            continue
        glyph = transfer_case(char, library, style, font, seed=ord(char))
        if glyph is not None:
            library.characters[char] = _single(char, glyph, font)
            report.transferred.append(char)
            continue

        glyph = from_reference(char, style, font, correction, seed=ord(char) + index)
        if glyph is not None:
            library.characters[char] = _single(char, glyph, font)
            report.referenced.append(char)
        else:
            report.failed.append(char)

    return report


def _single(char: str, glyph: Glyph, font: FontConfig) -> CharacterSet:
    """Tek örnekli bir karakter kümesi kurar.

    Üretilmiş glifin varyantı olmaz: aynı şeyi üç kez göstermek yerine tek
    biçim kullanılır. `calt` zaten bu glifi tekrarlayacaktır.
    """
    return CharacterSet(
        char=char,
        variants=[glyph],
        seen=0,
        dropped=0,
        advance=glyph.advance,
        bearing=glyph.bearing,
        descent=glyph.descent,
        synthetic=True,
    )
