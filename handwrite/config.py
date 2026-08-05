"""Ayarlanabilir sabitler, karakter kümesi ve tipografik önseller.

Buradaki her sayı bir varsayımdır; boru hattının hiçbir yerinde sabit sayı
gömülü değildir, hepsi buradan gelir. Kalibrasyon yaparken tek dosyaya bakılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Karakter kümesi
# --------------------------------------------------------------------------

TURKISH_LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"
FOREIGN_LOWER = "qwx"
TURKISH_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
FOREIGN_UPPER = "QWX"
DIGITS = "0123456789"
PUNCTUATION = ".,;:!?'\"()-"

#: Fontta üretilmesi hedeflenen tüm karakterler.
CHARSET = TURKISH_LOWER + FOREIGN_LOWER + TURKISH_UPPER + FOREIGN_UPPER + DIGITS + PUNCTUATION

# --------------------------------------------------------------------------
# Tipografik sınıflar
#
# Bir karakterin dikey olarak nereye oturduğunu bilmek iki yerde işe yarar:
#   1. taban çizgisi (baseline) tahminini doğrulamak,
#   2. bir glifin yanlış kesildiğini anlamak (ör. "o" çıkması gereken yerden
#      askılı bir şekil çıkmışsa kesim kaymıştır).
# --------------------------------------------------------------------------

#: x-yüksekliğinin üstüne çıkan küçük harfler.
ASCENDERS = set("bdfhklt")

#: Taban çizgisinin altına inen küçük harfler (ç ve ş'nin çengeli dahil).
DESCENDERS = set("gğjpqyçş")

#: Sadece x-yüksekliği bandında kalan küçük harfler.
XHEIGHT_ONLY = set("acemnoörsuüvwxzıi")

#: Gövdesi x-yüksekliğinde olup üstünde ayrı bir işaret taşıyan harfler.
#: Kesim sırasında bu işaretlerin gövdeyle birlikte taşınması gerekir.
DIACRITIC_CHARS = set("iöüğçşİÖÜĞÇŞ")


def vertical_class(ch: str) -> str:
    """Karakterin dikey sınıfını döndürür: ascender / descender / xheight / other."""
    if ch in ASCENDERS:
        return "ascender"
    if ch in DESCENDERS:
        return "descender"
    if ch in XHEIGHT_ONLY:
        return "xheight"
    if ch.isupper() or ch.isdigit():
        return "ascender"
    return "other"


# --------------------------------------------------------------------------
# Genişlik önselleri
#
# Kısıtlı segmentasyon DP'si başlangıçta bu tabloyu kullanır, sonra ölçülen
# gerçek genişliklerle tabloyu günceller ve tekrar koşar (EM benzeri).
# Değerler ortalama küçük harf genişliğine göre orandır.
# --------------------------------------------------------------------------

_WIDTH_PRIOR: dict[str, float] = {
    "i": 0.34, "ı": 0.34, "l": 0.34, "j": 0.40, "t": 0.52, "f": 0.50,
    "r": 0.56, "İ": 0.40, "I": 0.36, "1": 0.50, "!": 0.30, "'": 0.24,
    '"': 0.40, ".": 0.30, ",": 0.30, ";": 0.30, ":": 0.30, "-": 0.45,
    "(": 0.38, ")": 0.38, "?": 0.62,
    "m": 1.55, "w": 1.40, "M": 1.60, "W": 1.60,
    "s": 0.72, "z": 0.76, "c": 0.78, "ç": 0.78, "e": 0.84, "v": 0.82,
    "x": 0.80, "y": 0.82, "ş": 0.72,
}

#: Genişlik önseli tabloda yoksa kullanılan varsayılan oran.
DEFAULT_WIDTH_PRIOR = 1.0


def width_prior(ch: str) -> float:
    """Karakterin ortalama harfe göre bağıl genişlik önselini döndürür."""
    if ch in _WIDTH_PRIOR:
        return _WIDTH_PRIOR[ch]
    if ch.isupper():
        return 1.15
    return DEFAULT_WIDTH_PRIOR


# --------------------------------------------------------------------------
# Konfigürasyon nesneleri
# --------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Tarama/fotoğraf temizleme ayarları."""

    #: Kısa kenarın ölçekleneceği hedef piksel (0 = ölçekleme yok).
    target_short_side: int = 1600
    #: Aydınlatma düzeltmesi için morfolojik açma çekirdeği (kalem kalınlığının
    #: birkaç katı olmalı ki yazı değil sadece arkaplan yakalansın).
    illumination_kernel: int = 41
    #: Sauvola benzeri uyarlamalı eşikleme penceresi (tek sayı olmalı).
    binarize_window: int = 35
    #: Uyarlamalı eşikleme sabiti; büyütmek daha az mürekkep bırakır.
    binarize_offset: int = 12
    #: Sayfa eğikliği araması bu aralıkta yapılır (derece).
    deskew_range: float = 8.0
    #: Eğiklik arama adımı (derece).
    deskew_step: float = 0.2
    #: Bu piksel sayısından küçük lekeler gürültü sayılıp silinir.
    despeckle_min_area: int = 6
    #: Şablon maskesinin şişirilme yarıçapı (piksel).
    #:
    #: Kamera bulanıklığı ve perspektif düzeltmedeki yeniden örnekleme, basılı
    #: içeriğin kenarlarını kanonik konumunun birkaç piksel dışına taşırır. Bu
    #: saçak maskelenmezse basılı örnek metin mürekkep sanılır. Şişirmenin
    #: bedeli düşüktür: şişirilen bölgede yalnızca daha katı olan "kalem" eşiği
    #: uygulanır, yani kullanıcının o bölgeye yazdığı gerçek mürekkep korunur.
    template_dilate: int = 4
    #: Bir pikselin mürekkep sayılması için şablonun öngördüğü tondan ne kadar
    #: koyu olması gerektiği; kağıt seviyesinin oranı olarak. Küçültmek soluk
    #: kalemleri yakalar ama basılı içeriği sızdırma riskini artırır.
    ink_margin_ratio: float = 0.22


@dataclass
class LineConfig:
    """Satır segmentasyonu ayarları."""

    #: Yatay projeksiyon profilinin yumuşatma penceresi, tahmini satır
    #: yüksekliğine oranla.
    smooth_ratio: float = 0.25
    #: Bir satır sayılmak için gereken minimum mürekkep yoğunluğu (tepe oranı).
    min_peak_ratio: float = 0.12
    #: İki satır arası minimum mesafe, tahmini satır yüksekliğine oranla.
    min_separation_ratio: float = 0.55
    #: Satır kutusuna eklenen dikey pay (satır yüksekliğine oranla).
    padding_ratio: float = 0.35
    #: Eğim (slant) araması bu aralıkta yapılır (derece).
    slant_limit: float = 40.0
    #: Eğim arama adımı (derece).
    slant_step: float = 1.0


@dataclass
class WordConfig:
    """Kelime segmentasyonu ayarları."""

    #: Kelime boşluğu eşiği: boşluk genişliği bu değeri aşarsa kelime sınırı.
    #: x-yüksekliğine oranla ifade edilir.
    gap_ratio: float = 0.62
    #: Bu genişlikten dar parçalar komşusuna yapıştırılır (x-yüksekliği oranı).
    min_word_width_ratio: float = 0.15


@dataclass
class SegmentConfig:
    """Kısıtlı karakter segmentasyonu (çekirdek algoritma) ayarları."""

    #: Dikişin yatay kayması için birim maliyet. Büyütmek dikişi dikleştirir.
    seam_lateral_cost: float = 0.55
    #: Mürekkep pikselini kesmenin maliyeti.
    seam_ink_cost: float = 1.0
    #: Taban çizgisi civarındaki mürekkebi kesmenin ek maliyeti. Bitişik yazıda
    #: bağlantılar taban çizgisinde olur; oradan kesmek doğru olduğu için bu
    #: değer 1'in altında tutulur (indirim).
    seam_baseline_discount: float = 0.45
    #: Taban çizgisi indirim bandının yarı yüksekliği (x-yüksekliği oranı).
    seam_baseline_band: float = 0.22
    #: Karakter genişliğinin önselden sapmasının maliyet katsayısı.
    #:
    #: Dikiş maliyeti kalem kalınlığına bölünerek normalize edildiği için "1
    #: birim" kabaca "bir kalem izini kesmek" demektir. Genişlik cezası
    #: `weight · (bağıl_hata)²` olduğundan, %50 genişlik hatası bu katsayıyla
    #: bir kalem izi kesmeye denk gelir. Katsayıyı düşürmek kesimleri
    #: boşluklara doğru kaydırır (bitişik yazıda hatalı), yükseltmek ise
    #: mürekkebi umursamadan eşit parçalara böler.
    #:
    #: Değer, tek bir sentetik sayfada değil farklı yazı stillerinde (eğik
    #: bitişik / dik ayrık / geriye eğik) ölçülerek seçildi: tek sayfada en iyi
    #: puanı veren ayar (7.0) diğer stillerde en kötüsü çıkıyor.
    width_cost_weight: float = 2.0
    #: Bir karakterin alabileceği en dar genişlik (önselin katı).
    min_width_factor: float = 0.35
    #: Bir karakterin alabileceği en geniş genişlik (önselin katı).
    max_width_factor: float = 2.6
    #: Genişlik önsellerinin ölçülen değerlerle kaç kez güncelleneceği.
    refine_passes: int = 2
    #: Bir satırın ölçeği sayfa medyanından bu oranın ötesinde saparsa satır
    #: güvenilmez sayılıp elenir. Asıl hedefi, metni tamamlanmamış satırları
    #: yakalamaktır: orada hizalama bütün satır boyunca kayar.
    line_scale_tolerance: float = 0.22


@dataclass
class GlyphConfig:
    """Glif normalizasyonu ve varyant seçimi ayarları."""

    #: Karşılaştırma için glifleri bu ızgaraya oturt.
    compare_grid: int = 48
    #: Karşılaştırma öncesi bulanıklaştırma (ızgara birimi).
    compare_blur: float = 1.6
    #: Medoide uzaklığı bu çeyrekliğin üstünde olan örnekler elenir.
    outlier_percentile: float = 68.0
    #: Aykırı eleme en fazla bu oranda örnek atabilir.
    max_outlier_fraction: float = 0.45
    #: Karakter başına saklanacak varyant sayısı.
    variants_per_char: int = 3
    #: Varyantların seçileceği "çekirdek" havuzun genişliği (yüzdelik).
    #: Küçültmek varyantları birbirine benzetir ama hasarlı örnek riskini
    #: düşürür; büyütmek tersi. Çeşitlilik bu havuzun *içinden* aranır.
    variant_core_percentile: float = 55.0
    #: Bir karakterin fonta girmesi için gereken minimum örnek sayısı.
    min_samples: int = 1
    #: Varyantların dikey konumunun karakter medyanına çekilme oranı.
    #:
    #: Yüksek tutulmasının sebebi ince ama önemli: kağıtta dikey titreme her
    #: harfte *rastgeledir* ve gözü rahatsız etmez. Fontta ise aynı glif
    #: defalarca kullanılır, yani titreme *sistematik* hale gelir — bir kez
    #: yukarıda kalmış "a", metnin her yerinde yukarıda kalır. Bu, kağıttaki
    #: doğal düzensizlikten çok daha kötü görünür. Doğal his, dikey konumdan
    #: değil harf biçimlerinin çeşitliliğinden (varyantlardan) gelir.
    vertical_regularize: float = 0.90
    #: Yan boşlukların karakter medyanına çekilme oranı; aynı gerekçe yatayda.
    horizontal_regularize: float = 0.85


@dataclass
class FontConfig:
    """Font metrikleri ve üretim ayarları."""

    family_name: str = "Handwrite"
    style_name: str = "Regular"
    units_per_em: int = 1000
    ascender: int = 800
    descender: int = -200
    x_height: int = 500
    cap_height: int = 700
    line_gap: int = 90
    #: Vektörleştirme öncesi glif bitmap'inin büyütme katsayısı. Merdiven
    #: etkisini azaltır; büyütmek dosya boyutunu değil sadece süreyi artırır.
    trace_upscale: int = 4
    #: potrace köşe eşiği (alphamax): büyütmek daha yuvarlak sonuç verir.
    trace_alphamax: float = 1.1
    #: potrace gürültü temizleme alanı.
    trace_turdsize: int = 3
    #: Kübik → kuadratik dönüşüm toleransı (em biriminin oranı).
    cu2qu_tolerance: float = 0.0012
    #: Ölçülemeyen durumlarda kullanılacak yan boşluk (em birimi).
    default_sidebearing: int = 40
    #: Boşluk karakterinin genişliği (em birimi); ölçüm varsa o kullanılır.
    default_space_width: int = 260


@dataclass
class Config:
    """Tüm boru hattının tek konfigürasyon nesnesi."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    line: LineConfig = field(default_factory=LineConfig)
    word: WordConfig = field(default_factory=WordConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    glyph: GlyphConfig = field(default_factory=GlyphConfig)
    font: FontConfig = field(default_factory=FontConfig)
    #: Ara adım görsellerinin yazılacağı klasör (None = üretme).
    debug_dir: str | None = None
