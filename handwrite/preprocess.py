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
# Kağıdın zeminden ayrılması (serbest mod)
# --------------------------------------------------------------------------


def detect_paper(
    gray: np.ndarray,
    min_area: float = 0.25,
    max_area: float = 0.995,
    min_ink_kept: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fotoğraftaki kağıdı bulup dikdörtgene oturtur; bulamazsa None döner.

    Şablon modunda kağıdın nerede olduğunu köşe işaretleri söylüyordu. Serbest
    modda böyle bir bilgi yok ve kağıdın *dışını* işlemeye devam etmek ölümcül:
    masa, gölge, kağıt kenarı — hepsi mürekkep sanılır. Ölçtüğümüz kadarıyla
    koyu bir masada (ton ~45) satır tespiti 6 satır yerine 226 satır buluyor ve
    her "satır" birkaç piksellik bir kırıntı oluyor; font da kırıntılardan
    oluşuyor.

    Kağıt, kadrajın en büyük *parlak* bölgesidir. Dört köşesi bulunabilirse
    perspektif de düzeltilir — telefonla eğik çekilmiş sayfa düzleşir.

    Tek bir aday üretip ona güvenmiyoruz. "En büyük parlak bölge" sayfanın
    üstünden geçen bir gölge yüzünden sayfanın yalnız bir yarısı olabilir;
    ölçülen gerçek bir fotoğrafta bu, mürekkebin %44'ünü çöpe atıyordu. Bu
    yüzden birkaç aday üretilip her biri ölçülebilir bir ölçütle sınanır:
    *yazının ne kadarı içeride kalıyor?* İlk geçen aday kullanılır, adaylar
    dardan genişe sıralanır (dar kırpma daha çok işe yarar, ama yazıyı kesen
    dar kırpma hiç işe yaramaz).

    Döndürür: (kırpılmış görüntü, özgün koordinatlardaki bölge maskesi) ya da
    None. Bölge maskesi, kırpmanın yazıyı koruyup korumadığını denetlemek için
    gerekir: iki görüntüde ayrı ayrı mürekkep saymak kararsızdır, hangi
    piksellerin içeride kaldığını saymak kesindir.
    """
    if min_ink_kept is None:
        min_ink_kept = PreprocessConfig().paper_min_ink_kept

    for region, corners in _paper_candidates(gray, min_area, max_area):
        if not _keeps_ink(gray, region, min_ink_kept):
            continue

        # Kağıdın *dışını* beyazlatmak, kırpmaktan farklı ve daha önemlidir.
        # Kırpma yalnız dikdörtgen bir sınır koyar; masanın köşelerde kalan
        # koyu parçaları içeride kalmaya devam eder. Ölçülen bir fotoğrafta
        # mürekkebin yarısı bu artıklardan geliyordu ve satır tespiti tamamen
        # çöküyordu.
        cleaned = np.where(region, gray, 255).astype(np.uint8)

        if corners is not None:
            return _warp_to_rectangle(cleaned, corners), region

        ys, xs = np.nonzero(region)
        return cleaned[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], region

    return None


def _paper_candidates(
    gray: np.ndarray, min_area: float, max_area: float
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Kağıt olabilecek bölgeleri dardan genişe üretir.

    İki aday var, ikisi de aynı parlaklık eşiğinden türüyor ama farklı
    varsayımla:

    1. **En büyük parlak bölge.** Kağıt tek parça göründüğünde doğru olan ve en
       dar kırpmayı veren aday.
    2. **Parlak parçaların dışbükey örtüsü.** Sayfanın üstünden geçen bir gölge
       kağıdı eşiğin iki yakasına düşürdüğünde kağıt iki ayrı parçaya bölünür;
       o iki parçanın örtüsü sayfanın kendisidir. Ölçüldü: %45 derinlikte bir
       gölgede birinci aday kadrajın %38'ine düşüyor, örtü sayfayı bütün
       hâlinde geri veriyor.

    Örtü fazla kapsayabilir (masada duran başka bir beyaz nesne içeri girer).
    Bunun bedeli kırpmanın daha az işe yaramasıdır — yazı kaybı değil; o yüzden
    ikinci sıradadır, birincisi yazıyı kesmediği sürece kullanılmaz.
    """
    height, width = gray.shape
    scale = 900.0 / max(height, width)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    # Otsu, "parlak kağıt" ile "koyu zemin" arasını ayırır. Kağıt üstündeki
    # yazı azınlıkta kaldığı için bu ayrım kağıdın kendisini verir.
    #
    # Aydınlatma burada bilerek *düzeltilmez*: kağıdı zeminden ayıran şey zaten
    # büyük ölçekli parlaklık farkıdır ve onu düzleştirmek aranan sinyali siler.
    # Ölçüldü — düzleştirilince koyu masada (ton 25) "en büyük parlak bölge"
    # kadrajın %100'ü çıkıyor, yani kağıt hiç bulunamıyor.
    blurred = cv2.GaussianBlur(small, (0, 0), 3.0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Kapama, gölgenin açtığı *ince* çatlakları kapatır. Ölçülen fotoğrafta
    # 9x9 kare çekirdek sayfanın %53'ünü alıyordu, 25'lik elips %97'sini.
    # Geniş bir gölge bandını kapatmaya yetmez — orada ikinci aday devreye girer.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_PAPER_CLOSE,) * 2)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    frame = float(small.shape[0] * small.shape[1])
    back = 1.0 / scale if scale < 1 else 1.0
    candidates: list[tuple[np.ndarray, np.ndarray | None]] = []

    shapes = [max(contours, key=cv2.contourArea)]
    # Gölgede bölünen sayfanın parçaları; kırıntılar örtüyü şişirmesin diye
    # kadrajın %5'inden küçük olanlar alınmaz.
    pieces = [c for c in contours if cv2.contourArea(c) >= frame * 0.05]
    if len(pieces) > 1:
        shapes.append(cv2.convexHull(np.vstack(pieces)))

    for shape in shapes:
        fraction = cv2.contourArea(shape) / frame
        # Kadrajın neredeyse tamamıysa zaten kağıt doludur, kırpmaya gerek yok;
        # çok küçükse bulduğumuz şey kağıt değildir.
        if not (min_area <= fraction <= max_area):
            continue
        built = _region_from_contour(shape, back, height, width)
        if built is not None:
            candidates.append(built)
    return candidates


def _region_from_contour(
    shape: np.ndarray, back: float, height: int, width: int
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Küçültülmüş görüntüdeki bir konturu tam çözünürlükte bölge maskesine çevirir.

    Bölge, konturun *dışbükey örtüsüdür*. Bir kağıt yaprağı dışbükeydir;
    fotoğrafta öyle görünmemesinin sebebi kağıt değil, üstüne düşen gölgedir.
    İki uç da ölçüldü ve ikisi de kötü:

    - Konturu çevreleyen **dikdörtgen** fazlasını alır. Gerçek bir fotoğrafta
      kağıdın konturu çevreleyen dikdörtgenin yalnız %74'ünü kaplıyordu; kalan
      %26 masa ve gölgeydi. O koyu şerit kırpmanın içinde kalınca mürekkep
      sayılıyor, üstelik sayfanın tamamını saran tek bir bileşen olarak yazı
      satırlarını birbirine bağlıyordu: mürekkebin %39'u oradan geliyor ve iki
      satır tek satır sanılıyordu.
    - **Konturun kendisi** eksiğini alır. Sayfaya vuran gölge, parlak bölgeye
      kenardan içeri giren bir çentik açıyor; kontur o çentiği sadakatle takip
      edip sayfanın ortasındaki bir şerit yazıyı dışarıda bırakıyordu —
      mürekkebin %28'i.

    Dışbükey örtü ikisini de çözer: gölge çentikleri dışbükeyleştirmede
    kapanır, kağıdın dışındaki koyu şerit örtünün dışında kalır. Ölçüm: yazının
    %100'ü içeride (kontur ile %72), kırpmanın sol kenarı 96 ton yerine 240 —
    yani masa değil kağıt.
    """
    region = np.zeros((height, width), np.uint8)
    hull = cv2.convexHull(shape)
    scaled = (hull.astype(np.float32) * back).astype(np.int32)

    x, y, w, h = cv2.boundingRect(scaled)
    if w < width * 0.2 or h < height * 0.2:
        return None

    cv2.drawContours(region, [scaled], -1, 1, cv2.FILLED)

    # Dört köşe çıkarsa perspektif de düzeltilebilir — telefonla eğik çekilmiş
    # sayfa düzleşir. Köşeler örtüden okunur; ham kontur gölge çentikleri
    # yüzünden nadiren dört köşeli görünür.
    approximation = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    corners = (
        _order_corners(approximation.reshape(4, 2).astype(np.float32) * back)
        if len(approximation) == 4
        else None
    )

    # Bölge içeri çekilir. Çekilecek miktar keyfi değil: yukarıdaki kapama,
    # gölgenin açtığı çatlakları kapatırken parlak maskeyi kendi yarıçapı kadar
    # *şişirir* de. O şişme geri alınmazsa kağıdın dışından bir şerit içeride
    # kalır — ölçüldü, 19 satırlık bir sayfada bu şerit 20. bir "satır" olarak
    # çıkıyordu. Üstüne kağıdın kendi kenar gölgesi için biraz daha eklenir.
    inset = max(3, int(_PAPER_CLOSE // 2 * back) + int(min(height, width) * 0.004))
    region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1,) * 2))
    mask = region > 0
    if not mask.any():
        return None
    return mask, corners


#: Parlak maskeyi kapatırken kullanılan çekirdek (900 piksellik çalışma
#: ölçeğinde). Gölgenin açtığı ince çatlakları kapatır; bedeli maskeyi kendi
#: yarıçapı kadar şişirmesidir, o yüzden bölge sonradan aynı kadar içeri çekilir.
_PAPER_CLOSE = 25


#: Bir pikselin "kalem izi" sayılması için çevresinden kaç ton koyu olması
#: gerektiği. Kamera gürültüsü (σ≈6) ile kağıt üstündeki mürekkep (150+ ton)
#: arasında geniş bir boşluk var; eşiğin yeri kritik değil, yeter ki gürültünün
#: birkaç katı olsun. Gölgede kalmış soluk yazı bile 100'ün üstünde kalıyor.
_INK_CONTRAST = 40


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Kaba bir yazı maskesi — yalnız kırpmayı denetlemek için.

    Koyu piksel saymak burada yetmez: kağıdın dışındaki masa da koyudur ve
    sayıma girerse "kırpma mürekkep kaybediyor" sonucu çıkar; oysa kaybedilen
    şey masadır. Kalem izini masadan ayıran şey *incelik*: bir harf darbesi
    birkaç piksel kalınlığındadır, masa ise geniş bir alandır. Kalem
    kalınlığından büyük bir çekirdekle kapama yapıp farka bakmak ikisini
    ayırır: "kara şapka" (blackhat) her pikselin yakın çevresindeki en parlak
    yapıdan ne kadar koyu olduğunu verir. Kalem izi, üstünde durduğu kağıttan
    çok koyudur; masanın ortasındaki bir piksel ise komşularıyla aynıdır.

    Fark *mutlak* ölçülür, orana bölünerek değil. Oran, koyu bir zeminde
    gürültüyü uçuruyordu: ton 25 üstündeki ±6'lık kamera gürültüsü %24'lük bir
    değişim demek ve masanın yarısı "mürekkep" çıkıyordu — ölçüldü, doğru bir
    kırpmada bile mürekkebin %92'si masadan geliyor görünüyordu.
    """
    size = max(9, int(min(gray.shape) * 0.02) | 1)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, element)
    return contrast > _INK_CONTRAST


def _keeps_ink(gray: np.ndarray, region: np.ndarray, minimum: float) -> bool:
    """Kırpmanın yazının çoğunu koruyup korumadığını denetler.

    Kağıt algılama tek bir şekilde değil, akla gelmeyecek biçimlerde
    yanılabilir: gölge sayfayı ikiye böler, kağıdın bir kısmı kadraj dışında
    kalır, masadaki başka bir beyaz nesne sayfa sanılır. Hepsinin ortak
    sonucu aynıdır — yazının bir kısmı çöpe gider ve font onsuz üretilir.

    Bu yüzden kırpmanın *sebebini* değil *sonucunu* denetliyoruz: mürekkebin
    belirgin bir kısmını kaybeden aday elenir, sıradaki daha geniş aday denenir,
    hiçbiri geçmezse kırpma yapılmaz. Kırpmamak, yanlış kırpmaktan iyidir;
    kırpmanın çözdüğü sorun (koyu masa) kırpmasız da kısmen çözülebilir, ama
    yazının yarısı gittiğinde yapılacak bir şey kalmaz.

    Ölçüm, iki görüntüde ayrı ayrı mürekkep sayarak değil, tek bir maskenin ne
    kadarının bölge içinde kaldığına bakarak yapılır; ayrı sayım, eşik her
    görüntüde yeniden hesaplandığı için kırpma doğruyken bile %10'a varan fark
    üretiyordu.
    """
    ink = _ink_mask(gray)
    total = int(ink.sum())
    if total <= 0:
        return True
    return int((ink & region).sum()) / total >= minimum


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Dört köşeyi sol-üst, sağ-üst, sağ-alt, sol-alt sırasına dizer."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def _warp_to_rectangle(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Dört köşesi bilinen kağıdı düz bir dikdörtgene açar."""
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])

    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    if width < 100 or height < 100:
        return gray

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC)


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

    # Serbest mod: önce kağıdı zeminden ayır. Kağıdın dışını işlemeye devam
    # etmek masayı, gölgeyi ve kağıt kenarını mürekkep sayar; koyu bir masada
    # satır tespiti tamamen çöker.
    if cfg.detect_paper:
        found = detect_paper(image, min_ink_kept=cfg.paper_min_ink_kept)
        if found is not None:
            image = found[0]

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
