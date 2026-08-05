"""Glif normalizasyonu, aykırı değer elemesi ve varyant seçimi.

Segmentasyondan çıkan ham kesitler doğrudan fonta konulursa ortaya bir fidye
mektubu çıkar: her harf farklı boyda, farklı yükseklikte, kimi taban çizgisinin
altında kimi havada. Buradaki iş, o kesitleri ortak bir tipografik ızgaraya
oturtmak — ama yazının kendine has düzensizliğini de tamamen öldürmeden.

Üç adım:

1. **Normalizasyon** — her glif, ait olduğu satırın ölçülen x-yüksekliğine ve
   taban çizgisine göre em birimlerine taşınır. Ölçek satır başına hesaplanır;
   kullanıcı sayfanın sonuna doğru büyüdüyse o satırın glifleri küçültülür.

2. **Aykırı eleme** — segmentasyon her zaman doğru kesmez. Aynı harfin
   örnekleri karşılaştırılıp merkezden uzak olanlar atılır. Bu, boru hattının
   geri kalanını segmentasyonun %10'luk hatalı kuyruğundan korur.

3. **Varyant seçimi** — harf başına birden çok temsilci saklanır. Fontun sahte
   görünmemesinin asıl sebebi budur: her "a" piksel piksel aynı olmadığında göz
   yazıyı el yazısı olarak okur.

Temsilciler ortalama alınarak değil, gerçek örnekler seçilerek (medoid)
belirlenir. El yazısı örneklerinin ortalaması bulanık bir hayalet üretir; asıl
istenen ise gerçekten yazılmış bir harftir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import FontConfig, GlyphConfig, vertical_class
from .segment import CharBox


@dataclass
class Glyph:
    """Em birimlerine taşınmış tek bir glif örneği."""

    char: str
    #: Glifin mürekkep maskesi (sıkı kırpılmış, piksel).
    mask: np.ndarray
    #: Piksel başına em birimi.
    scale: float
    #: Mürekkebin sol kenarının, glif orijinine (kesim sınırı) piksel uzaklığı.
    bearing_px: float
    #: Taban çizgisinin mürekkebin üst kenarına piksel uzaklığı
    #: (pozitif = mürekkep taban çizgisinin üstünde başlıyor).
    ascent_px: float
    #: Karakterin yazılmış ilerleme genişliği (piksel).
    advance_px: float
    line_index: int
    char_index: int

    # -- em birimindeki metrikler ------------------------------------------

    @property
    def advance(self) -> float:
        return self.advance_px * self.scale

    @property
    def bearing(self) -> float:
        """Sol yan boşluk (em birimi)."""
        return self.bearing_px * self.scale

    @property
    def ink_width(self) -> float:
        return self.mask.shape[1] * self.scale

    @property
    def ascent(self) -> float:
        """Mürekkebin taban çizgisi üstündeki yüksekliği (em birimi)."""
        return self.ascent_px * self.scale

    @property
    def descent(self) -> float:
        """Mürekkebin taban çizgisi altına inen miktarı (em birimi, pozitif)."""
        return (self.mask.shape[0] - self.ascent_px) * self.scale


#: x-yüksekliğini ölçmek için güvenilir harfler: gövdesi tam olarak x-yüksekliği
#: bandında olan ve üstünde nokta/aksan taşımayanlar. "i", "ı", "ö", "ü" bilerek
#: dışarıda: mürekkep yükseklikleri gövdelerinden büyüktür.
XHEIGHT_REFERENCE = set("acemnorsuvwxz")


def measure_xheights(
    boxes: list[CharBox], min_samples: int = 4
) -> tuple[dict[int, float], float]:
    """Satır başına ve belge geneli x-yüksekliğini ölçülen gliflerden hesaplar.

    Neden segmentasyondan *sonra*: satırın x-yüksekliğini mürekkep profilinden
    kestirmek, satırda küçük harf olduğu sürece çalışır. Salt büyük harften
    oluşan bir satırda (alfabe satırı gibi) profil platosu cap-height'tır ve
    ölçüm %40 şişer — o satırın bütün glifleri fontta olması gerekenden küçük
    çıkar. Hangi mürekkebin hangi harfe ait olduğunu bildikten sonra ise
    x-yüksekliği tanımı gereği doğru ölçülür.

    Yeterli örneği olmayan satırlar belge medyanına düşer; kişinin yazı boyu
    sayfa boyunca aşağı yukarı sabit olduğu için bu iyi bir tahmindir.
    """
    per_line: dict[int, list[float]] = {}
    for box in boxes:
        if box.char in XHEIGHT_REFERENCE and box.mask.size:
            per_line.setdefault(box.line_index, []).append(float(box.mask.shape[0]))

    every = [h for values in per_line.values() for h in values]
    if not every:
        return {}, 0.0

    document = float(np.median(every))
    measured = {
        index: float(np.median(values)) if len(values) >= min_samples else document
        for index, values in per_line.items()
    }
    return measured, document


def apply_xheights(
    boxes: list[CharBox], measured: dict[int, float], document: float
) -> list[CharBox]:
    """Kutuların x-yüksekliğini ölçülen değerlerle değiştirir."""
    if document <= 0:
        return boxes
    for box in boxes:
        box.xheight = measured.get(box.line_index, document)
    return boxes


def normalize(box: CharBox, font: FontConfig) -> Glyph | None:
    """Bir karakter kutusunu em birimlerine taşır."""
    if box.xheight <= 0 or box.mask.size == 0 or not box.mask.any():
        return None
    scale = font.x_height / box.xheight
    return Glyph(
        char=box.char,
        mask=box.mask,
        scale=scale,
        bearing_px=float(box.x0 - box.cut_left),
        ascent_px=float(box.baseline - box.y0),
        advance_px=float(box.advance),
        line_index=box.line_index,
        char_index=box.char_index,
    )


def normalize_all(boxes: list[CharBox], font: FontConfig) -> list[Glyph]:
    return [g for g in (normalize(b, font) for b in boxes) if g is not None]


# --------------------------------------------------------------------------
# Karşılaştırma uzayı
# --------------------------------------------------------------------------


def render_signature(glyph: Glyph, cfg: GlyphConfig, font: FontConfig) -> np.ndarray:
    """Glifi karşılaştırma için sabit bir ızgaraya çizer.

    Izgara em uzayında tanımlıdır: taban çizgisi ve glif orijini her örnekte
    aynı hücrede olur. Böylece karşılaştırma yalnız şekli değil *konumu* da
    hesaba katar — havada kalmış ya da yanlış yerden kesilmiş bir parça,
    şeklen makul görünse bile merkezden uzağa düşer.

    Karşılaştırmadan önce bulanıklaştırma yapılır: bir iki piksellik kalem
    farkının mesafeyi domine etmesini engeller.
    """
    size = cfg.compare_grid
    canvas = np.zeros((size, size), dtype=np.float32)

    # Em uzayında görünür pencere: yatayda [-0.15, 1.15], dikeyde
    # [descender, ascender]. Ölçek ikisinde de aynı tutulur ki şekil bozulmasın.
    span = 1.30 * font.units_per_em
    units_per_cell = span / size
    origin_col = 0.15 * font.units_per_em / units_per_cell
    baseline_row = size * 0.72

    height, width = glyph.mask.shape
    cell_scale = glyph.scale / units_per_cell
    target_w = max(1, int(round(width * cell_scale)))
    target_h = max(1, int(round(height * cell_scale)))
    if target_w > size * 2 or target_h > size * 2:
        return canvas

    small = cv2.resize(
        glyph.mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    col = int(round(origin_col + glyph.bearing_px * cell_scale))
    row = int(round(baseline_row - glyph.ascent_px * cell_scale))

    x0, y0 = max(0, col), max(0, row)
    x1, y1 = min(size, col + target_w), min(size, row + target_h)
    if x0 >= x1 or y0 >= y1:
        return canvas
    canvas[y0:y1, x0:x1] = small[y0 - row : y1 - row, x0 - col : x1 - col]

    if cfg.compare_blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), cfg.compare_blur)
    return canvas


def _distance_matrix(signatures: np.ndarray) -> np.ndarray:
    """Düzleştirilmiş imzalar arasındaki öklit mesafe matrisi."""
    flat = signatures.reshape(len(signatures), -1)
    square = (flat * flat).sum(axis=1)
    gram = flat @ flat.T
    squared = np.maximum(square[:, None] + square[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(squared)


# --------------------------------------------------------------------------
# Aykırı eleme ve varyant seçimi
# --------------------------------------------------------------------------


@dataclass
class CharacterSet:
    """Tek bir karakter için seçilmiş varyantlar ve ölçülmüş metrikler."""

    char: str
    variants: list[Glyph]
    #: Elenmeden önceki örnek sayısı.
    seen: int
    #: Aykırı bulunup atılan örnek sayısı.
    dropped: int
    #: Ölçülen ilerleme genişliği (em birimi).
    advance: float
    #: Ölçülen sol yan boşluk (em birimi).
    bearing: float
    #: Mürekkebin taban çizgisi altına inen medyan miktarı (em birimi).
    #: Varyantların dikey düzenlenmesinde hedef alınır: harflerin taban
    #: çizgisine oturması, en çok göze çarpan tutarlılık ölçütüdür.
    descent: float = 0.0

    @property
    def kept(self) -> int:
        return self.seen - self.dropped


def _select_variants(distances: np.ndarray, count: int, core_percentile: float) -> list[int]:
    """Birbirinden farklı ama *tipik* `count` temsilci seçer.

    Buradaki denge inceliklidir ve yanlış tarafa düşmek fontu bozar. Varyantlar
    birbirine benzemezse iyi: üç neredeyse özdeş "a", tek "a" ile aynı yapay
    görüntüyü verir. Ama çeşitliliği doğrudan "merkeze en uzak örnek" diye
    aramak, tam da hasarlı örnekleri seçmek demektir — komşu harften parça
    kopmuş bir "e" ya da yanına "r" yapışmış bir "o", tanımı gereği en uzaktaki
    örneklerdir.

    Çözüm, çeşitliliği *çekirdek* içinde aramak: önce merkeze yakın tipik
    örnekler ayrılır, sonra farklılık yalnızca onların arasından seçilir. Böylece
    varyantlar birbirinden ayırt edilebilir ama hepsi gerçekten o harftir.
    """
    n = len(distances)
    count = min(count, n)
    medoid = int(np.argmin(distances.sum(axis=1)))
    if count <= 1:
        return [medoid]

    to_medoid = distances[medoid]
    cutoff = float(np.percentile(to_medoid, core_percentile))
    core = np.flatnonzero(to_medoid <= cutoff)
    if core.size < count:
        core = np.argsort(to_medoid)[: max(count, 1)]

    block = distances[np.ix_(core, core)]
    local_medoid = int(np.argmin(block.sum(axis=1)))

    chosen = [local_medoid]
    while len(chosen) < min(count, len(core)):
        nearest = block[chosen].min(axis=0)
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest)))

    # Lloyd turları: temsilcileri kendi kümelerinin merkezine çek.
    for _ in range(6):
        assignment = np.argmin(block[chosen], axis=0)
        updated = []
        for k in range(len(chosen)):
            members = np.flatnonzero(assignment == k)
            if members.size == 0:
                updated.append(chosen[k])
                continue
            inner = block[np.ix_(members, members)]
            updated.append(int(members[np.argmin(inner.sum(axis=1))]))
        if updated == chosen:
            break
        chosen = updated

    return [int(core[i]) for i in chosen]


#: Tipografik sınıfa göre beklenen dikey ölçüler, x-yüksekliğinin katı olarak:
#: (en az yükseklik, en çok yükseklik, en çok taban altı iniş).
#: Sınırlar bilerek geniş: amaç el yazısının doğal değişkenliğini budamak değil,
#: kaba segmentasyon hatalarını yakalamak.
_PLAUSIBLE: dict[str, tuple[float, float, float]] = {
    "xheight": (0.55, 1.60, 0.30),
    "ascender": (1.00, 2.20, 0.30),
    "descender": (0.60, 2.20, 1.20),
    "other": (0.10, 2.60, 1.30),
}


def is_plausible(glyph: Glyph, font: FontConfig) -> bool:
    """Glifin boyutlarının harfin tipografik sınıfına uyup uymadığına bakar.

    Segmentasyonun en sık ve en zararlı hatası, bir kesimin komşu harften parça
    koparmasıdır. Sonuç genelde şekil olarak makul ama *boy olarak* imkânsız
    olur: x-yüksekliğinde olması gereken bir "a", yanındaki "l"nin gövdesini de
    aldığı için iki katı uzunlukta çıkar. Bu ucuz kontrol, kümelemeye
    girmeden önce böyle örnekleri eler; kümelemeye bırakılsalar örnek sayısı
    azken medoidi kendilerine çekebilirlerdi.
    """
    unit = font.x_height
    if unit <= 0:
        return True
    low, high, max_descent = _PLAUSIBLE[vertical_class(glyph.char)]
    height = (glyph.ascent + glyph.descent) / unit
    if not low <= height <= high:
        return False
    return glyph.descent / unit <= max_descent


def build_character_set(
    char: str, glyphs: list[Glyph], cfg: GlyphConfig, font: FontConfig
) -> CharacterSet | None:
    """Bir karakterin örneklerinden varyantları ve metriklerini üretir."""
    if len(glyphs) < cfg.min_samples:
        return None

    plausible = [g for g in glyphs if is_plausible(g, font)]
    # Hepsi elenirse ölçüt yanlış demektir (ör. kullanıcının yazısı beklenenden
    # çok farklı); o durumda elemeyi uygulamamak, karakteri hiç üretmemekten
    # iyidir.
    rejected_by_shape = len(glyphs) - len(plausible)
    if len(plausible) >= max(1, cfg.min_samples):
        glyphs = plausible
    else:
        rejected_by_shape = 0

    signatures = np.stack([render_signature(g, cfg, font) for g in glyphs])
    distances = _distance_matrix(signatures)

    keep = list(range(len(glyphs)))
    if len(glyphs) >= 4:
        # Merkeze uzaklık dağılımının kuyruğunu at. Kaç örnek atılacağı üstten
        # sınırlıdır: bir harfin bütün örnekleri gerçekten çeşitliyse hepsini
        # atmak yerine olduğu gibi tutmak daha iyidir.
        medoid = int(np.argmin(distances.sum(axis=1)))
        to_medoid = distances[medoid]
        cutoff = float(np.percentile(to_medoid, cfg.outlier_percentile))
        candidate = [i for i in keep if to_medoid[i] <= cutoff]
        floor = int(np.ceil(len(glyphs) * (1.0 - cfg.max_outlier_fraction)))
        if len(candidate) >= max(2, floor):
            keep = candidate

    kept = [glyphs[i] for i in keep]
    sub = distances[np.ix_(keep, keep)]
    picks = _select_variants(sub, cfg.variants_per_char, cfg.variant_core_percentile)
    variants = [kept[i] for i in picks]

    return CharacterSet(
        char=char,
        variants=variants,
        seen=len(glyphs) + rejected_by_shape,
        dropped=len(glyphs) - len(kept) + rejected_by_shape,
        advance=float(np.median([g.advance for g in kept])),
        bearing=float(np.median([g.bearing for g in kept])),
        descent=float(np.median([g.descent for g in kept])),
    )


@dataclass
class GlyphLibrary:
    """Fonta girecek bütün karakterler."""

    characters: dict[str, CharacterSet] = field(default_factory=dict)
    #: Ölçülen boşluk genişliği (em birimi); ölçülemezse None.
    space_advance: float | None = None

    def coverage_report(self) -> list[tuple[str, int, int, int]]:
        """(karakter, görülen, atılan, varyant) satırları — en zayıf önce."""
        rows = [
            (c.char, c.seen, c.dropped, len(c.variants)) for c in self.characters.values()
        ]
        rows.sort(key=lambda r: (r[1] - r[2], r[1]))
        return rows


def build_library(
    glyphs: list[Glyph], cfg: GlyphConfig, font: FontConfig, space_advance: float | None = None
) -> GlyphLibrary:
    """Normalize edilmiş gliflerden karakter kütüphanesini kurar."""
    grouped: dict[str, list[Glyph]] = {}
    for glyph in glyphs:
        grouped.setdefault(glyph.char, []).append(glyph)

    library = GlyphLibrary(space_advance=space_advance)
    for char, samples in grouped.items():
        entry = build_character_set(char, samples, cfg, font)
        if entry is not None:
            library.characters[char] = entry
    return library
