"""Tarama/fotoğraf temizleme: perspektif düzeltme, aydınlatma, mürekkep ayırma.

Bu modülün çıktısı boru hattının geri kalanının tek girdisidir: bir boolean
mürekkep maskesi ve o maskenin hangi kanonik sayfa geometrisine oturduğu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import PreprocessConfig
from .template import ARUCO_DICT, SheetSpec, render_template, writable_mask


class RegistrationError(RuntimeError):
    """Sayfa köşe işaretleri bulunamadığında atılır."""


@dataclass
class Page:
    """Kanonik koordinatlara oturtulmuş, mürekkebi ayrılmış bir sayfa."""

    #: El yazısı mürekkebi (True = mürekkep).
    ink: np.ndarray
    #: Düzeltilmiş gri tonlamalı görüntü (görselleştirme ve hata ayıklama için).
    gray: np.ndarray
    #: Sayfa geometrisi; serbest modda None.
    spec: SheetSpec | None
    #: Bulunan köşe işaretlerinin id listesi.
    found_markers: list[int]
    #: Tahmini kalem kalınlığı (piksel).
    stroke_width: float


# --------------------------------------------------------------------------
# Yükleme ve temel işlemler
# --------------------------------------------------------------------------


def load_gray(path: str | Path) -> np.ndarray:
    """Görüntüyü gri tonlamalı olarak yükler."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_illumination(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Düzensiz aydınlatmayı giderir.

    Arkaplan, morfolojik kapama ile tahmin edilir: çekirdek kalem kalınlığından
    belirgin şekilde büyük olduğu için yazı yutulur ve geriye sadece kağıdın
    aydınlanma haritası kalır. Bölme ile bu harita düzleştirilir.
    """
    size = kernel if kernel % 2 == 1 else kernel + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, element)
    background = cv2.GaussianBlur(background, (0, 0), size / 4.0)
    background = np.maximum(background.astype(np.float32), 1.0)
    normalized = gray.astype(np.float32) / background * 235.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def despeckle(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Verilen alandan küçük bağlantılı bileşenleri siler."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def estimate_stroke_width(mask: np.ndarray) -> float:
    """Kalem kalınlığını mesafe dönüşümünün iskelet üzerindeki değerinden tahmin eder.

    Mürekkep alanının toplamı / iskelet uzunluğu da işe yarar ama mesafe
    dönüşümünün tepe değerlerinin medyanı aykırı değerlere karşı daha dayanıklı.
    """
    if not mask.any():
        return 1.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # Yerel maksimumlar iskeleti yaklaşık olarak verir; oradaki mesafe kalem
    # yarıçapıdır.
    peaks = dist[dist > 0]
    if peaks.size == 0:
        return 1.0
    radius = float(np.percentile(peaks, 75))
    return max(1.0, radius * 2.0)


# --------------------------------------------------------------------------
# Sayfa kaydı (registration)
# --------------------------------------------------------------------------


def detect_markers(gray: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco köşe işaretlerini bulur; {id: 4x2 köşe dizisi} döndürür."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    # Telefon fotoğrafları bulanık olabilir; eşikleme penceresini genişletiyoruz.
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 8
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def rectify_to_sheet(
    gray: np.ndarray, spec: SheetSpec, min_markers: int = 3
) -> tuple[np.ndarray, list[int]]:
    """Fotoğrafı kanonik sayfa koordinatlarına oturtur.

    Her işaretin dört köşesi bir nokta eşleşmesi verdiğinden üç işaret bile 12
    nokta demektir; homografi için fazlasıyla yeterli ve gürültüye dayanıklıdır.
    """
    found = detect_markers(gray)
    wanted = {mid: np.array(c, dtype=np.float64) for mid, c in zip(spec.marker_ids, spec.marker_corners)}
    common = sorted(set(found) & set(wanted))

    if len(common) < min_markers:
        raise RegistrationError(
            f"Sayfa köşe işaretleri bulunamadı ({len(common)}/{len(wanted)} bulundu). "
            "Fotoğrafta dört köşedeki kare işaretlerin tamamı görünmeli ve okunaklı olmalı."
        )

    src = np.vstack([found[i] for i in common])
    dst = np.vstack([wanted[i] for i in common])
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None:
        raise RegistrationError("Homografi hesaplanamadı; fotoğrafı daha düz çekmeyi deneyin.")

    rectified = cv2.warpPerspective(
        gray,
        homography,
        (spec.width, spec.height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rectified, common


# --------------------------------------------------------------------------
# Mürekkep ayırma
# --------------------------------------------------------------------------


def _threshold_freeform(norm: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Şablonsuz sayfada mürekkebi ayırır — basılı defter çizgilerini eleyerek.

    Buradaki asıl iş kalemi kağıttan ayırmak değil, kalemi *basılı defter
    çizgisinden* ayırmaktır. İkisi de koyudur ama aynı ölçüde değil: matbaa
    çizgisi kasten soluk basılır, kalem ise koyu yazar. Aydınlatma
    düzeltmesinden sonra bu fark küresel bir tonda okunabilir hale gelir.

    Otsu eşiği "kağıt olmayan her şeyi" verir; çizgiler de o tarafa düşer. Eşiği
    ölçülen kalem tonuna doğru bir miktar sıkmak çizgileri eler. Bedeli, kalem
    izlerinin kenarındaki yumuşak pikselleri de kaybetmektir — yani darbeler
    kıl payı incelir, ki bu glif biçimini gözle görülür şekilde değiştirmez.
    Alternatif olan uyarlamalı eşikleme burada işe yaramaz: yerel ortalamaya
    göre karar verdiği için soluk çizgiyi kendi çevresinde koyu sayar.
    """
    threshold, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = norm[norm < threshold]
    if dark.size == 0:
        return norm < threshold

    pen_level = float(np.median(dark))
    strict = pen_level + cfg.pen_separation * (float(threshold) - pen_level)
    return norm < strict


def fit_print_response(norm: np.ndarray, template: np.ndarray) -> tuple[float, float, float]:
    """Basım + tarama zincirinin gri seviye tepkisini ölçer.

    Şablonu biz ürettiğimiz için her pikselin *nominal* gri değerini biliriz
    (kağıt 255, örnek metin 150, taban çizgisi 168, ...). Taramada bu değerler
    yazıcıya, kağıda, tarayıcıya ve ışığa göre kayar. Her nominal seviye için
    taramadaki medyan değeri ölçüp aralarına bir doğru uydurarak bu kaymayı
    çözeriz: `gözlenen ≈ a · nominal + b`.

    Medyan kullanılması önemli: kullanıcı kılavuz çizgilerinin üstüne yazdığında
    o piksellerin bir kısmı kalem mürekkebi olur, ama azınlıkta kaldıkları için
    medyanı kaydıramazlar.

    Döndürür: (a, b, kağıdın gözlenen seviyesi).
    """
    paper_mask = template >= 250
    paper_level = float(np.percentile(norm[paper_mask], 60)) if paper_mask.any() else 235.0

    samples: list[tuple[float, float]] = [(255.0, paper_level)]
    for level in np.unique(template):
        if level >= 250:
            continue
        mask = template == level
        # Kenar yumuşatmadan gelen ara tonlar az sayıdadır; onlara güvenmeyiz.
        if int(mask.sum()) < 200:
            continue
        samples.append((float(level), float(np.median(norm[mask]))))

    if len(samples) < 2:
        return paper_level / 255.0, 0.0, paper_level

    nominal = np.array([s[0] for s in samples], dtype=np.float64)
    observed = np.array([s[1] for s in samples], dtype=np.float64)
    design = np.vstack([nominal, np.ones_like(nominal)]).T
    (a, b), *_ = np.linalg.lstsq(design, observed, rcond=None)

    # Uydurma saçmaladıysa (ör. sayfada neredeyse hiç basılı içerik yok)
    # basit orantıya düş.
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.1:
        return paper_level / 255.0, 0.0, paper_level
    return float(a), float(b), paper_level


def extract_ink(
    gray: np.ndarray,
    cfg: PreprocessConfig,
    template: np.ndarray | None = None,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """El yazısı mürekkebini ayırır.

    Şablon verilmişse tek ve basit bir kural uygulanır:

        *bir piksel, şablonun orada olmasını söylediği şeyden belirgin şekilde
        koyuysa el yazısıdır.*

    Bu kural hem kağıt üstündeki kalemi, hem kılavuz çizgisinin üstünden geçen
    kalemi doğru yakalar; basılı içeriği ise konumuna bakmadan, kendi nominal
    tonuyla karşılaştırarak eler. Sabit eşik ya da kaba dikdörtgen maskeleme
    ikisini birden yapamıyordu.

    Şablon, gri tonlamalı aşındırma (erosion) ile hafifçe "yayılır": kamera
    bulanıklığı ve perspektif düzeltmedeki yeniden örnekleme basılı işaretlerin
    koyuluğunu birkaç piksel dışına taşırır, aşındırma da beklenen değeri aynı
    şekilde taşıyarak bu saçağı karşılar.
    """
    norm = normalize_illumination(gray, cfg.illumination_kernel)

    if template is None:
        return despeckle(_threshold_freeform(norm, cfg), cfg.despeckle_min_area)

    slope, intercept, paper_level = fit_print_response(norm, template)

    radius = max(1, cfg.template_dilate)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    spread = cv2.erode(template, element)

    expected = slope * spread.astype(np.float32) + intercept
    margin = cfg.ink_margin_ratio * paper_level
    ink = norm.astype(np.float32) < expected - margin
    if region is not None:
        ink &= region
    return despeckle(ink, cfg.despeckle_min_area)


# --------------------------------------------------------------------------
# Basılı cetvel çizgilerinin temizlenmesi (serbest mod)
# --------------------------------------------------------------------------


def _dense_thin_runs(
    density: np.ndarray, min_fill: float, extent: int, max_thickness_ratio: float = 0.006
) -> list[tuple[int, int]]:
    """Basılı cetvel çizgilerinin bulunduğu piksel satırlarını döndürür.

    İki koşul birden aranır. Yalnız yoğunluğa bakmak yetmez: sıkışık yazılmış
    bir metin satırının x-yüksekliği bandı da yoğun çıkar. Ama o bant kalındır,
    çizgi ise birkaç piksel; incelik koşulu ikisini ayırır.

    Kritik ayrıntı: yoğun aralığın *tamamı* değil yalnız tepesi döndürülür.
    Harfler çizginin üstüne oturduğu için, çizginin hemen üstündeki birkaç satır
    da yoğun çıkar — o satırları da çizgi sayıp silmek harflerin ayaklarını
    kesmek demektir. Silinecek olan yalnızca yoğunluğun tepe yaptığı, yani
    gerçekten basılı çizginin bulunduğu satırlardır.
    """
    dense = density > min_fill
    if not dense.any():
        return []

    padded = np.concatenate([[False], dense, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    limit = max(3, int(extent * max_thickness_ratio))

    runs: list[tuple[int, int]] = []
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > limit:
            continue
        window = density[start:end]
        peak = float(window.max())
        # Tepe değerinin belirgin şekilde altında kalan satırlar çizgiye değil,
        # üstüne oturan yazıya aittir.
        core = np.flatnonzero(window >= peak * 0.85)
        runs.append((int(start + core[0]), int(start + core[-1] + 1)))
    return runs


def remove_rules(
    ink: np.ndarray,
    min_length_ratio: float = 0.30,
    crossing_gap: int = 3,
    thickness: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Defter çizgilerini siler, üstlerinden geçen kalem izlerini korur.

    Şablon modunda çizgilerin nerede olduğunu biliyorduk; serbest modda
    bilmiyoruz, o yüzden biçimlerinden tanıyoruz: bir defter çizgisi sayfa
    genişliğinin belirgin bir kısmı boyunca kesintisiz yataydır. El yazısında
    bu kadar uzun kesintisiz yatay iz bulunmaz.

    Asıl incelik silerken: bir harfin dikey gövdesi çizgiyi kestiğinde o
    piksellerin de çizgiye ait *görünmesi*dir. Onları da silmek harfi ikiye
    böler. Ayırt etmek için her çizgi pikselinin birkaç piksel üstünde *ve*
    altında mürekkep olup olmadığına bakılır: ikisi de varsa oradan bir kalem
    geçiyor demektir ve piksel korunur.

    Döndürür: (temizlenmiş mürekkep, silinen çizgi maskesi).
    """
    height, width = ink.shape

    # Çizgileri kesintisiz uzunluklarından değil, *bir satırı boydan boya
    # doldurmalarından* tanıyoruz. Basılı bir defter çizgisi tarama sonrası
    # parça parça çıkabilir — o zaman sabit uzunluklu morfolojik açma onu
    # bulamaz — ama parçalar hep aynı piksel satırında kalır. Satır yoğunluğu
    # bu yüzden çok daha sağlam bir ölçüttür.
    rule_rows = _dense_thin_runs(ink.sum(axis=1) / width, min_length_ratio, height)
    rule_cols = _dense_thin_runs(ink.sum(axis=0) / height, min_length_ratio, width)

    horizontal_rules = np.zeros_like(ink)
    for start, end in rule_rows:
        horizontal_rules[start:end] = ink[start:end]
    vertical_rules = np.zeros_like(ink)
    for start, end in rule_cols:
        vertical_rules[:, start:end] = ink[:, start:end]

    # Komşuluk sınaması, çizgi *dışındaki* mürekkebe karşı yapılır. Ham mürekkebe
    # karşı yapılsaydı, birkaç piksel kalınlaşmış bir çizgi kendi kalınlığını
    # "üstümde ve altımda mürekkep var" diye okur ve kendini korurdu.
    other = ink & ~(horizontal_rules | vertical_rules)

    gap = max(crossing_gap, thickness // 2 + 1)
    above = np.zeros_like(ink)
    below = np.zeros_like(ink)
    above[: height - gap] = other[gap:]
    below[gap:] = other[: height - gap]
    left = np.zeros_like(ink)
    right = np.zeros_like(ink)
    left[:, : width - gap] = other[:, gap:]
    right[:, gap:] = other[:, : width - gap]

    # Kesişme testi çizginin yönüne göre ayrılmalıdır. Yatay bir çizgiyi "sağında
    # ve solunda mürekkep var mı" diye sınamak anlamsızdır: çizginin *her*
    # pikselinin sağında ve solunda kendisi vardır, dolayısıyla hiçbir şey
    # silinmez. Yatay çizgiyi kesen şey dikey bir kalem izidir, o da ancak
    # üstünde ve altında mürekkep olmasıyla anlaşılır. Dikey çizgide durum
    # simetriktir.
    removable = (horizontal_rules & ~(above & below)) | (vertical_rules & ~(left & right))

    # Silinen çizginin uçlarında kalan kırıntıları da temizle.
    grown = cv2.dilate(removable.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    removable = grown & (horizontal_rules | vertical_rules)

    return ink & ~removable, removable


# --------------------------------------------------------------------------
# Serbest mod için eğiklik düzeltme
# --------------------------------------------------------------------------


def estimate_skew(ink: np.ndarray, cfg: PreprocessConfig) -> float:
    """Sayfa eğikliğini yatay projeksiyon keskinliğini maksimize ederek bulur.

    Satırlar yataya paralel olduğunda projeksiyon profili en keskin tepeleri
    verir; varyansı maksimize eden açı doğru açıdır.
    """
    # Arama, küçültülmüş görüntü üzerinde yapılır: sonuç aynı, maliyet düşük.
    scale = 600.0 / max(ink.shape)
    small = cv2.resize(ink.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    best_angle, best_score = 0.0, -1.0
    h, w = small.shape
    center = (w / 2.0, h / 2.0)
    angle = -cfg.deskew_range
    while angle <= cfg.deskew_range:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, matrix, (w, h), flags=cv2.INTER_LINEAR)
        profile = rotated.sum(axis=1).astype(np.float64)
        score = float(np.var(profile))
        if score > best_score:
            best_score, best_angle = score, angle
        angle += cfg.deskew_step
    return best_angle


def rotate_image(image: np.ndarray, angle: float, fill: int = 0) -> np.ndarray:
    """Görüntüyü merkez etrafında döndürür."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill
    )


# --------------------------------------------------------------------------
# Üst seviye
# --------------------------------------------------------------------------


def prepare_page(
    image: np.ndarray,
    spec: SheetSpec | None,
    cfg: PreprocessConfig,
) -> Page:
    """Ham gri görüntüden temizlenmiş, kanonik koordinatlı bir sayfa üretir."""
    if spec is not None:
        rectified, found = rectify_to_sheet(image, spec)
        template = render_template(spec)
        ink = extract_ink(rectified, cfg, template, region=writable_mask(spec))
        return Page(
            ink=ink,
            gray=rectified,
            spec=spec,
            found_markers=found,
            stroke_width=estimate_stroke_width(ink),
        )

    # Serbest mod: ölçekle, cetvel çizgilerini temizle, eğikliği düzelt.
    if cfg.target_short_side:
        scale = cfg.target_short_side / min(image.shape[:2])
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ink = extract_ink(image, cfg, None)
    if cfg.remove_rules:
        ink, _ = remove_rules(ink, min_length_ratio=cfg.rule_min_length_ratio)
        ink = despeckle(ink, cfg.despeckle_min_area)

    angle = estimate_skew(ink, cfg)
    if abs(angle) > 0.05:
        ink = rotate_image(ink.astype(np.uint8), angle, fill=0) > 0
        image = rotate_image(image, angle, fill=255)

    return Page(
        ink=ink,
        gray=image,
        spec=None,
        found_markers=[],
        stroke_width=estimate_stroke_width(ink),
    )
