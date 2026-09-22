"""Satır çıkarma ve satır metriklerinin ölçümü.

Çalışma sayfası modunda satırların *nerede* olduğu zaten bilinir (bantlar);
burada yapılan iş mürekkebi doğru banda atamak ve o satırın gerçek taban
çizgisi ile x-yüksekliğini ölçmektir. Basılı kılavuz çizgileri güçlü bir önsel
verir ama kimse tam olarak çizginin üstüne yazmaz; ölçüm buna göre düzeltilir.

Serbest modda (şablonsuz) satırlar yatay projeksiyon profilinden bulunur.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import LineConfig
from .preprocess import Page


@dataclass
class Line:
    """Tek bir yazı satırı ve ölçülmüş dikey metrikleri."""

    index: int
    #: Bu satıra karşılık gelen metin (biliniyorsa).
    text: str
    #: Satırın mürekkep maskesi (kırpılmış).
    ink: np.ndarray
    #: Kırpmanın sayfa üzerindeki sol-üst köşesi.
    origin: tuple[int, int]
    #: Kırpma koordinatlarında ölçülen taban çizgisi.
    baseline: float
    #: Ölçülen x-yüksekliği (piksel).
    xheight: float
    #: Ölçüm basılı kılavuzdan mı türetildi yoksa tamamen mürekkepten mi.
    measured: bool = True

    @property
    def width(self) -> int:
        return self.ink.shape[1]

    @property
    def height(self) -> int:
        return self.ink.shape[0]

    def page_baseline(self) -> float:
        return self.baseline + self.origin[1]


# --------------------------------------------------------------------------
# Dikey metrik ölçümü
# --------------------------------------------------------------------------


def measure_metrics(
    ink: np.ndarray,
    baseline_prior: float | None = None,
    xheight_prior: float | None = None,
) -> tuple[float, float]:
    """Bir satırın taban çizgisini ve x-yüksekliğini mürekkepten ölçer.

    Yöntem: satır içi yatay projeksiyon profili alınır. Küçük harflerin gövdesi
    x-yüksekliği bandında yoğunlaşır, bu yüzden profil orada bir plato yapar.
    Platonun alt kenarı taban çizgisi, üst kenarı x-yüksekliği çizgisidir.
    Ascender ve descender'lar profilin kuyruklarını oluşturur ve eşiklemeyle
    dışarıda kalır.

    Basılı kılavuzdan gelen önsel varsa arama onun etrafında sınırlandırılır;
    bu, tek harflik ya da çok kısa satırlarda ölçümün saçmalamasını engeller.
    """
    profile = ink.sum(axis=1).astype(np.float64)
    if profile.max() <= 0:
        return (baseline_prior or ink.shape[0] * 0.75, xheight_prior or ink.shape[0] * 0.4)

    threshold = profile.max() * 0.5
    rows = np.flatnonzero(profile >= threshold)
    if rows.size == 0:
        return (baseline_prior or ink.shape[0] * 0.75, xheight_prior or ink.shape[0] * 0.4)

    top, bottom = float(rows[0]), float(rows[-1] + 1)

    # Önsel varsa ölçümün ondan çok uzaklaşmasına izin verme.
    if baseline_prior is not None and xheight_prior is not None:
        slack = 0.45 * xheight_prior
        if abs(bottom - baseline_prior) > slack:
            bottom = baseline_prior
        if abs((bottom - top) - xheight_prior) > slack:
            top = bottom - xheight_prior

    xheight = max(bottom - top, 1.0)
    return bottom, xheight


# --------------------------------------------------------------------------
# Şablon modu: bantlardan satır
# --------------------------------------------------------------------------


def lines_from_bands(page: Page, cfg: LineConfig) -> list[Line]:
    """Sayfa bantlarından satırları çıkarır.

    Mürekkep bileşenleri banda göre kırpılarak değil *atanarak* dağıtılır:
    kırpma, bir üst satırın descender'ını ("g", "y" kuyruğu) alt satıra
    sızdırır ve o satırın metriklerini bozardı. Her bağlantılı bileşen, gövde
    ağırlık merkezinin düştüğü tek bir banda gider.
    """
    assert page.spec is not None, "lines_from_bands yalnızca şablon modunda çalışır"
    spec = page.spec

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        page.ink.astype(np.uint8), connectivity=8
    )

    baselines = np.array([b.baseline_y for b in spec.bands], dtype=np.float64)
    # Bir bileşenin bir banda ait sayılması için ağırlık merkezinin o bandın
    # yazı bölgesine makul uzaklıkta olması gerekir. En yakın banda koşulsuz
    # atamak, sayfada kalan her lekeyi bir satıra sokar ve o satırın
    # hizalamasını bozar.
    reach = np.array(
        [(b.descender_y - b.ascender_y) * 0.85 for b in spec.bands], dtype=np.float64
    )

    owner = np.full(count, -1, dtype=np.int32)
    for i in range(1, count):
        cy = centroids[i][1]
        distances = np.abs(baselines - cy)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= reach[nearest]:
            owner[i] = nearest
    owner[0] = -1  # arkaplan

    lines: list[Line] = []
    for band in spec.bands:
        member = np.flatnonzero(owner == band.index)
        if member.size == 0:
            continue

        # Bu banda ait bileşenlerin ortak sınır kutusu.
        x0 = int(min(stats[i, cv2.CC_STAT_LEFT] for i in member))
        y0 = int(min(stats[i, cv2.CC_STAT_TOP] for i in member))
        x1 = int(max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in member))
        y1 = int(max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in member))

        pad = int(round(cfg.padding_ratio * (band.baseline_y - band.xheight_y)))
        x0, y0 = max(0, x0 - 2), max(0, y0 - pad)
        x1, y1 = min(page.ink.shape[1], x1 + 2), min(page.ink.shape[0], y1 + pad)

        crop_labels = labels[y0:y1, x0:x1]
        mask = np.isin(crop_labels, member)

        baseline, xheight = measure_metrics(
            mask,
            baseline_prior=band.baseline_y - y0,
            xheight_prior=float(band.baseline_y - band.xheight_y),
        )
        lines.append(
            Line(
                index=band.index,
                text=band.text,
                ink=mask,
                origin=(x0, y0),
                baseline=baseline,
                xheight=xheight,
            )
        )
    return lines


# --------------------------------------------------------------------------
# Serbest mod: projeksiyon profilinden satır
# --------------------------------------------------------------------------


def estimate_line_pitch(profile: np.ndarray, minimum: int = 12) -> float:
    """Satır aralığını (pitch) yatay profilin otokorelasyonundan ölçer.

    Satır yüksekliğini "mürekkep içeren blokların kalınlığı" diye tahmin etmek
    cazip ama kırılgandır: bir satırın descender'ı alttakinin ascender'ına
    değdiği anda bütün sayfa tek bir bloğa dönüşür ve tahmin sayfa boyu kadar
    çıkar. Oysa satırlar *periyodiktir* — insan hep aynı aralıkla yazar.
    Otokorelasyon bu periyodu, satırlar birbirine değse de bulur.

    En küçük güçlü tepe seçilir; en büyüğü seçmek periyodun iki katına
    kilitlenip satırları ikişer ikişer birleştirme riski taşır.
    """
    if profile.size < 4 * minimum:
        return float(max(profile.size / 4.0, minimum))

    centred = profile - profile.mean()
    correlation = np.correlate(centred, centred, mode="full")[profile.size - 1 :]
    if correlation[0] <= 0:
        return float(minimum)
    correlation = correlation / correlation[0]

    high = max(minimum + 1, profile.size // 3)
    window = correlation[minimum:high]
    if window.size < 3:
        return float(minimum)

    peaks = np.flatnonzero(
        (window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:]) & (window[1:-1] > 0.25)
    )
    if peaks.size == 0:
        return float(minimum + int(np.argmax(window)))
    return float(minimum + int(peaks[0]) + 1)


def detect_lines(page: Page, cfg: LineConfig, texts: list[str] | None = None) -> list[Line]:
    """Şablonsuz bir sayfada satırları yatay projeksiyon profilinden bulur."""
    profile = page.ink.sum(axis=1).astype(np.float64)
    if profile.max() <= 0:
        return []

    pitch = estimate_line_pitch(profile)

    window = max(3, int(pitch * cfg.smooth_ratio) | 1)
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, window), 0).ravel()

    bands = _split_at_valleys(smooth, pitch, cfg)

    lines: list[Line] = []
    for index, (y0, y1) in enumerate(bands):
        band = page.ink[y0:y1]
        if not band.any():
            continue
        columns = np.flatnonzero(band.any(axis=0))
        x0, x1 = int(columns[0]), int(columns[-1] + 1)
        mask = band[:, x0:x1]
        baseline, xheight = measure_metrics(mask)
        lines.append(
            Line(
                index=index,
                text="",
                ink=mask,
                origin=(x0, y0),
                baseline=baseline,
                xheight=xheight,
                measured=True,
            )
        )

    lines = _drop_fragment_lines(lines)
    lines = _drop_edge_streaks(lines)
    for index, line in enumerate(lines):
        line.index = index
        line.text = texts[index] if texts and index < len(texts) else ""
    return lines


def _drop_fragment_lines(lines: list[Line], floor: float = 0.45) -> list[Line]:
    """Satır sayılamayacak kadar cılız bantları eler.

    Tepe bulma, bir satırın ascender'larından ya da descender'larından oluşan
    ince şeritleri ayrı bir satır sanabilir. Bunlar ölçülen x-yüksekliğiyle
    kendilerini belli eder: gerçek satırların onda biri kadar çıkarlar. Fonta
    girerlerse hizalama o "satırda" tamamen saçmalar.
    """
    if len(lines) < 3:
        return lines
    reference = float(np.median([line.xheight for line in lines]))
    if reference <= 0:
        return lines
    return [line for line in lines if line.xheight >= floor * reference]


#: Bir bileşenin "yazı değil, çizgi" sayılması için gereken en/boy oranı ve
#: satırın en fazla kaç bileşenden oluşabileceği. Ölçüm: kağıt kenarı şeritleri
#: 15,1 ve 15,2 en/boy ile 4 ve 6 bileşen; gerçek yazı satırlarının hiçbiri
#: 5,8'i ve hiçbiri 19 bileşenin altını görmüyor. Eşikler aradaki boşluğa
#: konuyor, iki koşul birden aranıyor ki altı çizili bir kelime tek başına bir
#: satırı sildirmesin.
_STREAK_ASPECT = 10.0
_STREAK_MAX_PARTS = 12


def _drop_edge_streaks(lines: list[Line]) -> list[Line]:
    """Kağıt kenarını satır sanmayı önler.

    Kağıt algılama sayfayı dışbükey bir çokgen olarak modeller; gerçek sayfa
    kenarı hafifçe eğri ya da eğik olduğunda ikisinin arasında ince, uca doğru
    incelen bir kama kalır. Bu kama sayfanın en üstünde ya da en altında,
    genişliği bir yazı satırı kadar olan koyu bir şerit olarak görünür ve
    yoğunluk, x-yüksekliği, sütun doluluğu gibi ölçütlerin hepsinde gerçek bir
    satırdan ayırt edilemez — ölçüldü, hepsinde gerçek satırların aralığına
    düşüyor.

    Ayıran şey *yapı*: bir yazı satırı çok sayıda ayrı parçadan (harfler,
    kelimeler) oluşur ve hiçbiri kendi boyunun birkaç katından geniş değildir.
    Kama ise tek parçadır ve satır boyunca uzanır. Hiçbir harf ya da kelime
    boyunun on katı kadar geniş olmaz.
    """
    kept: list[Line] = []
    for line in lines:
        mask = line.ink.astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            continue
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        widest = int(np.argmax(widths))
        aspect = widths[widest] / max(int(heights[widest]), 1)
        if aspect >= _STREAK_ASPECT and (count - 1) <= _STREAK_MAX_PARTS:
            continue
        kept.append(line)
    return kept


# --------------------------------------------------------------------------
# Eğim (slant) tahmini ve düzeltme
# --------------------------------------------------------------------------


def shear_array(array: np.ndarray, angle_deg: float, baseline: float, fill: float = 0) -> np.ndarray:
    """Diziyi taban çizgisi etrafında yatay olarak makaslar (dtype korunur).

    Makaslama merkezinin taban çizgisi olması önemli: taban çizgisi
    seviyesindeki x koordinatları değişmez, dolayısıyla ölçtüğümüz karakter
    ilerleme genişlikleri makaslamadan etkilenmez.

    Etiket haritası gibi tamsayı dizilerinde de kullanılabilsin diye en yakın
    komşu örneklemesi yapılır; ara değer üretmez.
    """
    if abs(angle_deg) < 0.05:
        return array
    factor = np.tan(np.radians(angle_deg))
    matrix = np.array([[1.0, factor, -factor * baseline], [0.0, 1.0, 0.0]], dtype=np.float32)

    source = array
    as_bool = source.dtype == bool
    if as_bool:
        source = source.astype(np.uint8)

    sheared = cv2.warpAffine(
        source,
        matrix,
        (array.shape[1], array.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return sheared > 0 if as_bool else sheared


def shear_about_baseline(ink: np.ndarray, angle_deg: float, baseline: float) -> np.ndarray:
    """`shear_array`in boolean maskeler için kısayolu."""
    return shear_array(ink, angle_deg, baseline, fill=0)


def deslant_padding(height: int, angle: float) -> int:
    """Makaslamanın mürekkebi kırpma dışına taşımaması için gereken yatay pay."""
    return int(np.ceil(abs(np.tan(np.radians(angle))) * height)) + 2


def estimate_slant(
    ink: np.ndarray, baseline: float, limit: float = 40.0, step: float = 1.0
) -> float:
    """Yazıyı dikleştirecek **düzeltme açısını** ölçer.

    Dikkat: dönen değer yazının eğimi değil, `shear_array`e doğrudan verilecek
    düzeltme açısıdır — yani `shear_array(ink, estimate_slant(ink, b), b)`
    dikleştirilmiş görüntüyü verir. İşareti tersine çevirmek eğimi gidermek
    yerine ikiye katlar.

    Ölçüt klasiktir: doğru açıyla makaslandığında dikey kalem darbeleri aynı
    sütuna hizalanır ve dikey projeksiyon profili birkaç sütunda yoğunlaşır.
    Profilin kareler toplamı bu yoğunlaşmayı ölçer ve doğru açıda tepe yapar.

    Eğimi *segmentasyondan önce* gidermek iki yerde işe yarar: dikişler dikey
    kalabildiği için yana kayma cezası anlamlı olur, ve gliflerin tamamı ortak
    bir eğime oturduğu için font tutarlı görünür.
    """
    if not ink.any():
        return 0.0

    best_angle, best_score = 0.0, -1.0
    angle = -limit
    while angle <= limit:
        sheared = shear_about_baseline(ink, angle, baseline)
        profile = sheared.sum(axis=0).astype(np.float64)
        score = float((profile * profile).sum())
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return best_angle


def page_slant(lines: list[Line], limit: float = 40.0, coarse: float = 2.0) -> float:
    """Sayfanın ortak eğimini satır tahminlerinin medyanı olarak bulur.

    Satır başına ayrı eğim düzeltmek yanlış olurdu: gerçek el yazısında eğim
    kişiye özgü ve sayfa boyunca sabittir, satırdan satıra gözlenen fark
    ölçüm gürültüsüdür. Medyan almak hem gürültüyü hem de bir iki bozuk satırı
    eler; ortak bir eğim kullanmak da fontun tutarlılığını korur.
    """
    if not lines:
        return 0.0
    estimates = [
        estimate_slant(line.ink, line.baseline, limit=limit, step=coarse)
        for line in lines
        if line.ink.any()
    ]
    if not estimates:
        return 0.0
    return float(np.median(estimates))


def deslant_line(line: Line, angle: float) -> Line:
    """Satırı dikleştirir. `angle`, `estimate_slant`/`page_slant` düzeltme açısıdır.

    Makaslama mürekkebi yana taşıdığı için tuval önce genişletilir; aksi halde
    ascender'ların ve descender'ların uçları kırpma sınırında kesilirdi. `origin`
    aynı miktarda geriye alınır, böylece satır-içi koordinatlar sayfa
    koordinatlarına dönüştürülebilir kalmayı sürdürür.
    """
    if abs(angle) < 0.05:
        return line

    height, width = line.ink.shape
    extra = deslant_padding(height, angle)

    padded = np.zeros((height, width + 2 * extra), dtype=bool)
    padded[:, extra : extra + width] = line.ink

    return Line(
        index=line.index,
        text=line.text,
        ink=shear_about_baseline(padded, angle, line.baseline),
        origin=(line.origin[0] - extra, line.origin[1]),
        baseline=line.baseline,
        xheight=line.xheight,
        measured=line.measured,
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Ardışık True bloklarını (başlangıç, bitiş) listesi olarak döndürür."""
    padded = np.concatenate([[False], flags, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])]


def _split_at_valleys(
    smooth: np.ndarray, pitch: float, cfg: LineConfig
) -> list[tuple[int, int]]:
    """Profilin tepelerini satır olarak alır, aralarındaki en derin vadiden böler.

    Eşiğin üstündeki blokları satır saymak yetmez: bir satırın descender'ı
    alttakinin ascender'ına değdiğinde iki satır tek blok olur ve eşik onları
    asla ayıramaz. Tepeleri bulup aralarındaki en düşük noktadan *aktif olarak*
    kesmek bu duruma dayanıklıdır.
    """
    if smooth.max() <= 0:
        return []

    threshold = smooth.max() * cfg.min_peak_ratio
    separation = max(3, int(pitch * cfg.min_separation_ratio))

    # Tepe seçimi: en yüksekten başlayarak, seçilenlerin çevresini bloke ederek
    # ilerle (maksimum olmayanların bastırılması).
    peaks: list[int] = []
    blocked = np.zeros(smooth.size, dtype=bool)
    for index in np.argsort(smooth)[::-1]:
        if smooth[index] < threshold:
            break
        if blocked[index]:
            continue
        peaks.append(int(index))
        blocked[max(0, index - separation) : index + separation + 1] = True
    if not peaks:
        return []
    peaks.sort()

    pad = pitch * (0.5 + cfg.padding_ratio)
    edges = [max(0, int(peaks[0] - pad))]
    for current, following in zip(peaks, peaks[1:]):
        edges.append(current + int(np.argmin(smooth[current : following + 1])))
    edges.append(min(smooth.size, int(peaks[-1] + pad)))

    return [(start, end) for start, end in zip(edges, edges[1:]) if end - start > 3]
