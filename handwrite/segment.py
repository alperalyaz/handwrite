"""Kısıtlı karakter segmentasyonu — projenin çekirdek algoritması.

Calligraphr'ın kutu doldurtmasının tek sebebi şu: bir karalamanın hangi harf
olduğunu bilmek zor. Biz o soruyu hiç sormuyoruz. Kullanıcı *ne yazdığını*
bildiğimiz bir metni yazdığı için problem "tanıma"dan "hizalama"ya iniyor:

    elimizde N karakterlik bir metin ve o metnin yazılmış hali var;
    mürekkebi soldan sağa tam olarak N parçaya nereden bölmeliyiz?

Bu, çözülmüş bir problem türü: kısıtlı en iyi bölme. Dinamik programlama ile
global en iyi bölme noktaları bulunur. İki maliyet dengelenir:

1. **Kesim maliyeti** — kesim çizgisi ne kadar mürekkep kesiyor. Kesim düz bir
   dikey çizgi değil, aşağı doğru sağa/sola kayabilen bir *dikiş*tir (seam);
   böylece eğik bağlantıları doğru yerden ayırır. Taban çizgisi civarındaki
   mürekkebi kesmek indirimlidir, çünkü bitişik yazıda harfleri birbirine
   bağlayan çizgi tam oradadır ve kesilmesi *gereken* yer orasıdır.

2. **Genişlik maliyeti** — parçanın genişliği o karakterden beklenene ne kadar
   uyuyor. "m" geniş, "i" dardır; bu önsel, mürekkebin ipucu vermediği
   yerlerde (tamamen bitişik yazı) kesimi doğru yere oturtur.

Karışık yazı bu iki maliyetin bileşimiyle tek bir mekanizmadan geçer: harfler
ayrıksa kesim maliyeti boşlukta zaten sıfırdır ve kesim oraya oturur; harfler
bitişikse genişlik önseli devreye girip en az mürekkep kesen yeri seçer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SegmentConfig, width_prior
from .lines import Line

INF = 1e18


@dataclass
class Priors:
    """Karakter genişliği önselleri.

    Başlangıçta `config.width_prior` tablosundaki genel tahminlerdir. Bir
    geçişten sonra bu tahminler kullanıcının *kendi* yazısından ölçülenlerle
    değiştirilir: kimi insanın "m"si dar, kiminin "a"sı geniştir ve genel tablo
    bunu bilemez. Ölçüp tekrar bölmek, özellikle dar harflerde (i, l, ı, r)
    belirgin fark yaratır — onlarda birkaç piksellik kayma bile glifin yarısını
    komşusundan almak demektir.
    """

    table: dict[str, float] = field(default_factory=dict)
    space: float = 0.75

    def width(self, ch: str) -> float:
        return self.table.get(ch, width_prior(ch))


@dataclass
class CharBox:
    """Segmentasyondan çıkan tek bir karakter örneği."""

    line_index: int
    char_index: int
    char: str
    #: Karakterin mürekkebi, satır kırpması boyutunda değil sıkı kırpılmış.
    mask: np.ndarray
    #: Sıkı kırpmanın satır koordinatlarındaki sol-üst köşesi.
    x0: int
    y0: int
    #: Kesim sınırları (satır koordinatı). Aradaki fark, karakterin *yazılmış*
    #: ilerleme genişliğidir — font metrikleri buradan ölçülür, tahmin edilmez.
    cut_left: int
    cut_right: int
    #: Satırın taban çizgisi ve x-yüksekliği (satır koordinatı / piksel).
    baseline: float
    xheight: float
    #: Bu karakterin içinde bulunduğu kelimenin hizalama maliyeti (karakter
    #: başına). Metin yanlışsa yükselir; hangi gliflerin şüpheli olduğunu
    #: buradan biliriz.
    word_cost: float = 0.0
    #: Kelimenin ölçeği: kapladığı piksel / beklenen bağıl genişlik toplamı.
    #: Okunan metinden bir harf düşerse aynı mürekkebe daha az karakter
    #: sığdırılır ve bu değer belirgin şekilde yükselir. Maliyetten daha
    #: duyarlı bir sinyaldir, çünkü doğrudan ölçülen bir orandır.
    word_scale: float = 0.0
    #: İçinde bulunduğu kelimenin karakter sayısı.
    word_length: int = 1

    @property
    def advance(self) -> int:
        return self.cut_right - self.cut_left

    @property
    def width(self) -> int:
        return self.mask.shape[1]

    @property
    def height(self) -> int:
        return self.mask.shape[0]

    @property
    def left_bearing(self) -> int:
        """Kesim sınırı ile mürekkebin başlangıcı arasındaki boşluk."""
        return self.x0 - self.cut_left

    @property
    def right_bearing(self) -> int:
        return self.cut_right - (self.x0 + self.width)


# --------------------------------------------------------------------------
# Dikiş (seam) maliyet alanı
# --------------------------------------------------------------------------


@dataclass
class SeamField:
    """Bir satır için önceden hesaplanmış dikiş maliyetleri ve yolları."""

    #: cost[c] = c sütunundan başlayan en ucuz dikişin maliyeti.
    cost: np.ndarray
    #: Geri izleme için her (satır, sütun) hücresindeki yatay adım (-1, 0, +1).
    step: np.ndarray
    height: int
    width: int

    def path(self, column: int) -> np.ndarray:
        """Verilen sütundan başlayan dikişin her satırdaki x konumunu döndürür."""
        xs = np.empty(self.height, dtype=np.int32)
        x = int(np.clip(column, 0, self.width - 1))
        for y in range(self.height):
            xs[y] = x
            # step int8'dir; Python tamsayısıyla doğrudan toplanırsa numpy
            # işlemi int8'de yapmaya çalışıp taşar.
            x = min(max(x + int(self.step[y, x]), 0), self.width - 1)
        return xs


def build_seam_field(
    ink: np.ndarray,
    baseline: float,
    xheight: float,
    stroke_width: float,
    cfg: SegmentConfig,
) -> SeamField:
    """Satırın her sütunu için yukarıdan aşağı en ucuz dikişi hesaplar.

    Dikiş her satırda en fazla bir piksel yana kayabilir ve kayma maliyetlidir;
    böylece dikişler makul ölçüde dik kalır ama eğik bağlantıları takip
    edebilir.

    Maliyetler kalem kalınlığına bölünerek normalize edilir: sonuç "kaç kalem
    izi kesildi" birimindedir, yani farklı çözünürlük ve kalem kalınlıklarında
    aynı anlama gelir. Genişlik maliyetiyle karşılaştırılabilir olması buna
    bağlıdır.
    """
    height, width = ink.shape
    unit = max(stroke_width, 1.0)

    ink_cost = np.where(ink, cfg.seam_ink_cost, 0.0).astype(np.float64)

    # Taban çizgisi bandında kesim indirimi: bitişik yazının bağlantı çizgisi
    # buradadır ve orayı kesmek doğru davranıştır.
    band = cfg.seam_baseline_band * xheight
    rows = np.arange(height, dtype=np.float64)
    in_band = np.abs(rows - baseline) <= band
    ink_cost[in_band] *= cfg.seam_baseline_discount

    ink_cost /= unit
    lateral = cfg.seam_lateral_cost

    accumulated = np.empty((height, width), dtype=np.float64)
    step = np.zeros((height, width), dtype=np.int8)
    accumulated[height - 1] = ink_cost[height - 1]

    for y in range(height - 2, -1, -1):
        below = accumulated[y + 1]
        left = np.empty(width)
        left[0] = INF
        left[1:] = below[:-1] + lateral
        right = np.empty(width)
        right[-1] = INF
        right[:-1] = below[1:] + lateral

        options = np.stack([left, below, right])
        choice = np.argmin(options, axis=0)
        accumulated[y] = ink_cost[y] + options[choice, np.arange(width)]
        step[y] = choice.astype(np.int8) - 1

    return SeamField(cost=accumulated[0], step=step, height=height, width=width)


# --------------------------------------------------------------------------
# Kısıtlı bölme DP'si
# --------------------------------------------------------------------------


def split_span(
    lo: int,
    hi: int,
    expected: np.ndarray,
    cut_cost: np.ndarray,
    cfg: SegmentConfig,
) -> list[int] | None:
    """`solve_split`in yalnızca sınırları döndüren kısayolu."""
    bounds, _ = solve_split(lo, hi, expected, cut_cost, cfg)
    return bounds


def solve_split(
    lo: int,
    hi: int,
    expected: np.ndarray,
    cut_cost: np.ndarray,
    cfg: SegmentConfig,
) -> tuple[list[int] | None, float]:
    """[lo, hi) aralığını, verilen bağıl genişliklere göre en iyi şekilde böler.

    `expected` her parçanın bağıl genişlik önselidir; ölçek aralığın toplam
    genişliğinden türetilir. `cut_cost[c]` c sütunundan kesmenin maliyetidir.

    Sınırlarla birlikte *en iyi çözümün maliyeti* de döndürülür. Bu maliyet
    sadece bir ara değer değil, hizalamanın kendi kendini denetleme aracıdır:
    metin gerçekten yazılana uyuyorsa parçalar genişlik önsellerine oturur ve
    maliyet düşük çıkar. Metin yanlışsa (ör. okuyan model bir harf düşürmüşse)
    aynı mürekkebe yanlış sayıda karakter sığdırılmaya çalışılır ve maliyet
    belirgin şekilde yükselir.

    Döndürülen liste `len(expected) + 1` sınır içerir; ilki `lo`, sonuncusu
    `hi`'dir. Kısıtlar sağlanamıyorsa (None, sonsuz) döner.

    DP, min-plus evrişimidir ve her parça için izin verilen genişlik aralığı
    üzerinde vektörleştirilmiştir; maliyet O(parça · genişlik_aralığı · uzunluk).
    """
    n = len(expected)
    span = hi - lo
    if n == 0 or span <= 0:
        return None, INF
    if n == 1:
        return [lo, hi], 0.0

    total = float(expected.sum())
    if total <= 0:
        return None, INF
    scale = span / total
    widths = np.maximum(expected * scale, 1.0)

    dp = np.full((n + 1, span + 1), INF)
    back = np.zeros((n + 1, span + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    local_cut = cut_cost[lo : hi + 1]

    for i in range(1, n + 1):
        target_width = widths[i - 1]
        low = max(1, int(round(cfg.min_width_factor * target_width)))
        high = min(span, max(low, int(round(cfg.max_width_factor * target_width))))

        row = np.full(span + 1, INF)
        row_back = np.zeros(span + 1, dtype=np.int32)
        previous = dp[i - 1]

        for step in range(low, high + 1):
            relative = (step - target_width) / target_width
            penalty = cfg.width_cost_weight * relative * relative
            candidate = previous[: span + 1 - step] + penalty
            view = row[step:]
            better = candidate < view
            view[better] = candidate[better]
            row_back[step:][better] = step

        if i < n:
            row = row + local_cut
        dp[i] = row
        back[i] = row_back

    if not np.isfinite(dp[n, span]) or dp[n, span] >= INF:
        return None, INF

    boundaries = [span]
    position = span
    for i in range(n, 0, -1):
        step = int(back[i, position])
        if step <= 0:
            return None, INF
        position -= step
        boundaries.append(position)
    boundaries.reverse()
    # Maliyet parça başına normalize edilir; uzun kelimeler doğal olarak daha
    # çok terim topladığı için ham toplam kelimeler arası karşılaştırılamaz.
    return [b + lo for b in boundaries], float(dp[n, span]) / n


def _relaxed_split(
    lo: int, hi: int, expected: np.ndarray, cut_cost: np.ndarray, cfg: SegmentConfig
) -> tuple[list[int], float]:
    """`split_span`i gevşetilmiş kısıtlarla dener, olmazsa orantılı böler.

    Kısıtların sağlanamaması gerçek bir durumdur: kullanıcı satırın sonunu
    sıkıştırmış ya da bir harfi atlamış olabilir. Böyle bir satırı tamamen
    çöpe atmak yerine elden geldiğince bölüp devam ederiz; kötü çıkan glifler
    zaten sonraki aykırı değer elemesinde düşer.
    """
    result, cost = solve_split(lo, hi, expected, cut_cost, cfg)
    if result is not None:
        return result, cost

    relaxed = SegmentConfig(**{**cfg.__dict__, "min_width_factor": 0.12, "max_width_factor": 6.0})
    result, cost = solve_split(lo, hi, expected, cut_cost, relaxed)
    if result is not None:
        # Gevşetilmiş kısıtlarla çözüldüyse hizalama zaten şüphelidir; maliyet
        # buna göre cezalandırılır ki aşağıdaki eleme onu görebilsin.
        return result, cost + 1.0

    fractions = np.concatenate([[0.0], np.cumsum(expected) / expected.sum()])
    return [int(round(lo + f * (hi - lo))) for f in fractions], INF


# --------------------------------------------------------------------------
# Metnin belirteçlere ayrılması
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, bool]]:
    """Metni (parça, boşluk_mu) çiftlerine ayırır; ardışık boşluklar birleşir."""
    tokens: list[tuple[str, bool]] = []
    for ch in text:
        is_space = ch == " "
        if tokens and tokens[-1][1] and is_space:
            tokens[-1] = (tokens[-1][0] + ch, True)
        elif tokens and not tokens[-1][1] and not is_space:
            tokens[-1] = (tokens[-1][0] + ch, False)
        else:
            tokens.append((ch, is_space))
    return tokens


def _token_width(token: str, is_space: bool, priors: Priors) -> float:
    if is_space:
        return priors.space * len(token)
    return sum(priors.width(ch) for ch in token)


# --------------------------------------------------------------------------
# Satır segmentasyonu
# --------------------------------------------------------------------------


def segment_line(
    line: Line,
    stroke_width: float,
    cfg: SegmentConfig,
    priors: Priors | None = None,
) -> list[CharBox]:
    """Bir satırı, metnini bilerek karakterlerine böler.

    İki kademede yapılır. Önce satır kelimelere ve boşluklara bölünür; sonra
    her kelime kendi içinde karakterlere bölünür. Tek kademeli global bir DP de
    mümkündü ama iki kademe, ölçeği her kelimede yerel olarak yeniden tahmin
    ettiği için kullanıcının satır içinde büyüyüp küçülmesine dayanıklıdır.
    """
    priors = priors or Priors()
    text = line.text
    if not text.strip() or not line.ink.any():
        return []

    columns = np.flatnonzero(line.ink.any(axis=0))
    if columns.size == 0:
        return []
    lo, hi = int(columns[0]), int(columns[-1] + 1)

    field = build_seam_field(line.ink, line.baseline, line.xheight, stroke_width, cfg)
    cut_cost = np.concatenate([field.cost, [field.cost[-1]]])

    tokens = _tokenize(text)
    token_widths = np.array(
        [_token_width(tok, sp, priors) for tok, sp in tokens], dtype=np.float64
    )
    token_bounds, _ = _relaxed_split(lo, hi, token_widths, cut_cost, cfg)

    boxes: list[CharBox] = []
    char_offset = 0
    for (token, is_space), start, end in zip(tokens, token_bounds, token_bounds[1:]):
        if is_space:
            char_offset += len(token)
            continue

        char_widths = np.array([priors.width(ch) for ch in token], dtype=np.float64)
        bounds, cost = _relaxed_split(start, end, char_widths, cut_cost, cfg)
        expected = float(char_widths.sum())
        scale = (end - start) / expected if expected > 0 else 0.0

        for k, ch in enumerate(token):
            box = _extract_char(
                line, field, ch, line.index, char_offset + k, bounds[k], bounds[k + 1]
            )
            if box is not None:
                box.word_cost = cost
                box.word_scale = scale
                box.word_length = len(token)
                boxes.append(box)
        char_offset += len(token)

    return boxes


# --------------------------------------------------------------------------
# Belge düzeyinde segmentasyon: önselleri ölçüp yeniden böl
# --------------------------------------------------------------------------


def line_scale(boxes: list[CharBox], priors: Priors) -> float:
    """Bir satırdaki "bir birim önsel kaç piksel" ölçeğini döndürür."""
    if not boxes:
        return 0.0
    expected = sum(priors.width(b.char) for b in boxes)
    if expected <= 0:
        return 0.0
    return sum(b.advance for b in boxes) / expected


def measure_priors(
    boxes_by_line: dict[int, list[CharBox]], current: Priors, min_samples: int = 3
) -> Priors:
    """Ölçülen ilerleme genişliklerinden yeni bir önsel tablosu üretir.

    Her satır kendi ölçeğine göre normalize edilir; kullanıcı sayfanın
    ilerleyen satırlarında büyüyüp küçülse bile ölçümler karşılaştırılabilir
    kalır. Karakter başına medyan alınır, çünkü hatalı kesilmiş birkaç örnek
    ortalamayı kolayca kaydırır.
    """
    samples: dict[str, list[float]] = {}
    space_samples: list[float] = []

    for boxes in boxes_by_line.values():
        scale = line_scale(boxes, current)
        if scale <= 0:
            continue
        for box in boxes:
            samples.setdefault(box.char, []).append(box.advance / scale)
        # Boşluklar: aralarında tam bir karakter atlanmış ardışık kutular.
        for left, right in zip(boxes, boxes[1:]):
            if right.char_index == left.char_index + 2:
                gap = right.cut_left - left.cut_right
                if gap > 0:
                    space_samples.append(gap / scale)

    table = dict(current.table)
    for ch, values in samples.items():
        if len(values) >= min_samples:
            table[ch] = float(np.median(values))

    space = current.space
    if len(space_samples) >= min_samples:
        space = float(np.median(space_samples))

    return Priors(table=table, space=max(space, 0.1))


@dataclass
class DocumentSegmentation:
    """Bir belgenin tüm karakter kutuları ve segmentasyon teşhisi."""

    boxes: list[CharBox]
    priors: Priors
    #: Satır indeksinden o satırın ölçeğine.
    scales: dict[int, float] = field(default_factory=dict)
    #: Güvenilmez bulunup elenen satırlar ve sebepleri.
    rejected: dict[int, str] = field(default_factory=dict)
    #: Hizalama maliyeti yüksek olduğu için atılan glif sayısı.
    dropped_words: int = 0


def segment_document(
    lines: list[Line], stroke_width: float, cfg: SegmentConfig
) -> DocumentSegmentation:
    """Tüm satırları böler, önselleri ölçüp yeniden böler ve bozuk satırları eler."""
    priors = Priors()

    def run(current: Priors) -> dict[int, list[CharBox]]:
        return {
            line.index: segment_line(line, stroke_width, cfg, current)
            for line in lines
        }

    by_line = run(priors)
    for _ in range(max(0, cfg.refine_passes)):
        priors = measure_priors(by_line, priors)
        by_line = run(priors)

    scales = {index: line_scale(boxes, priors) for index, boxes in by_line.items()}
    rejected = _reject_outlier_lines(scales, cfg)

    boxes = [box for index, group in by_line.items() if index not in rejected for box in group]
    boxes, dropped = drop_misaligned_words(boxes, cfg)
    return DocumentSegmentation(
        boxes=boxes,
        priors=priors,
        scales=scales,
        rejected=rejected,
        dropped_words=dropped,
    )


def drop_misaligned_words(
    boxes: list[CharBox], cfg: SegmentConfig
) -> tuple[list[CharBox], int]:
    """Hizalama maliyeti anormal yüksek kelimelerin gliflerini atar.

    Bu, metnin dışarıdan geldiği (bir modelin okuduğu) durumda asıl güvenlik
    ağıdır. Okunan metinde tek bir harf eksik ya da fazlaysa, o kelimenin
    mürekkebine yanlış sayıda karakter sığdırılır ve kelimedeki *bütün* glifler
    kayar. Maliyet bunu görür: doğru metinde parçalar genişlik önsellerine
    oturur, yanlış metinde oturmaz.

    Hasarın kelimeyle sınırlı kalması, segmentasyonun iki kademeli olmasının
    doğal sonucudur — önce kelimeler, sonra kelime içi karakterler bölünür.
    Tek kademeli bir hizalamada bir harflik hata bütün satırı kaydırırdı.

    Eşik mutlak değil, belgenin kendi maliyet dağılımına göre belirlenir:
    kullanıcının yazısı ne kadar düzgünse taban maliyet o kadar düşüktür.
    """
    if len(boxes) < 10:
        return boxes, 0

    costs = np.array([box.word_cost for box in boxes], dtype=np.float64)
    finite = costs[np.isfinite(costs)]
    if finite.size:
        median = float(np.median(finite))
        deviation = float(np.median(np.abs(finite - median))) or 1e-6
        cost_limit = median + cfg.word_cost_tolerance * deviation
    else:
        cost_limit = INF

    # Ölçek sinyali maliyetten daha duyarlıdır. Beş harflik bir kelimeden bir
    # harf düşerse maliyet ancak biraz artar (genişlik cezası karesel ve küçük
    # hatalarda cılızdır), ama kelimenin ölçeği doğrudan ~%25 yükselir. Ölçüt
    # olarak belgenin kendi medyan ölçeği kullanılır.
    scales = np.array(
        [box.word_scale for box in boxes if box.word_length >= 3 and box.word_scale > 0],
        dtype=np.float64,
    )
    reference = float(np.median(scales)) if scales.size >= 8 else 0.0

    def suspicious(box: CharBox) -> bool:
        if box.word_cost > cost_limit:
            return True
        if reference <= 0 or box.word_length < 3 or box.word_scale <= 0:
            return False
        return abs(box.word_scale / reference - 1.0) > cfg.word_scale_tolerance

    kept = [box for box in boxes if not suspicious(box)]
    # Her şeyi atmak, hiçbir şey atmamaktan kötüdür: ölçüt yanlış kalibre
    # olmuşsa fontu tamamen boşaltmak yerine elemeyi iptal ederiz.
    if len(kept) < len(boxes) * 0.4:
        return boxes, 0
    return kept, len(boxes) - len(kept)


def _reject_outlier_lines(scales: dict[int, float], cfg: SegmentConfig) -> dict[int, str]:
    """Ölçeği sayfanın geri kalanından kopuk satırları eler.

    Asıl yakalamak istediğimiz durum: kullanıcı satırı bitirmemiş. O zaman
    hizalama, yazılmamış harfleri de mürekkebe sığdırmaya çalışır ve satırdaki
    *bütün* glifler kayar. Böyle bir satırı fontta kullanmak, düzgün satırlardan
    kazanılan her şeyi bozar; elemek doğru olandır.

    Ölçüt olarak medyandan sapma kullanılır: insanlar sayfa boyunca aşağı yukarı
    aynı boyda yazar, kopan satır ölçüsüyle kendini belli eder.
    """
    valid = {index: value for index, value in scales.items() if value > 0}
    if len(valid) < 3:
        return {}

    values = np.array(list(valid.values()), dtype=np.float64)
    median = float(np.median(values))
    if median <= 0:
        return {}

    rejected: dict[int, str] = {}
    for index, value in valid.items():
        deviation = value / median - 1.0
        if deviation < -cfg.line_scale_tolerance:
            rejected[index] = (
                f"satır beklenenden %{abs(deviation) * 100:.0f} sıkışık — "
                "metnin tamamı yazılmamış olabilir"
            )
        elif deviation > cfg.line_scale_tolerance * 1.6:
            rejected[index] = (
                f"satır beklenenden %{deviation * 100:.0f} geniş — "
                "fazladan işaret ya da leke olabilir"
            )
    for index, value in scales.items():
        if value <= 0:
            rejected[index] = "satırda kullanılabilir mürekkep bulunamadı"
    return rejected


def _extract_char(
    line: Line,
    field: SeamField,
    ch: str,
    line_index: int,
    char_index: int,
    cut_left: int,
    cut_right: int,
) -> CharBox | None:
    """İki dikiş arasındaki mürekkebi tek bir karakter olarak çıkarır.

    Kesim düz dikey çizgiyle değil dikiş yoluyla yapılır: eğik bir bağlantıyı
    dikey kesmek komşu harfin ucundan parça koparırdı.
    """
    if cut_right <= cut_left:
        return None

    left_path = field.path(cut_left)
    right_path = field.path(cut_right)

    columns = np.arange(field.width)[None, :]
    inside = (columns >= left_path[:, None]) & (columns < right_path[:, None])
    mask = line.ink & inside

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)

    return CharBox(
        line_index=line_index,
        char_index=char_index,
        char=ch,
        mask=mask[y0:y1, x0:x1],
        x0=x0,
        y0=y0,
        cut_left=cut_left,
        cut_right=cut_right,
        baseline=line.baseline,
        xheight=line.xheight,
    )
