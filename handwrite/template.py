"""Çalışma sayfası üretimi ve sayfa geometrisi.

Kullanıcı boş bir kağıda rastgele yazmak yerine bu modülün ürettiği sayfaya
yazar. Kazandığımız şey büyük:

* Köşelerdeki ArUco işaretleri sayesinde fotoğraf perspektifi tam olarak
  düzeltilir; eğiklik/dewarp tahmin edilmez, çözülür.
* Her yazı bandının hangi metne karşılık geldiği kesin bilinir. Satır ile
  metni eşleştirme problemi tamamen ortadan kalkar.
* Basılı taban çizgisi, glif normalizasyonu için tahmin değil ölçüm sağlar.

Sayfa geometrisi bir JSON yan dosyasına yazılır; boru hattı onu okur.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Yazılacak örnek metin
# --------------------------------------------------------------------------

#: Varsayılan korpus.
#:
#: İki kısıt altında seçildi:
#:   1. Kapsam — Türkçe'nin bütün harfleri, yabancı harfler, rakamlar ve
#:      noktalama; küçük harfler en az 5, geri kalanı en az 3 kez geçmeli.
#:      Tek örnekli bir harf için varyant üretilemez, font sahte görünür.
#:   2. Genişlik — satırlar ~40 karakteri aşmamalı. Basılı örnek metin dar
#:      sığar ama el yazısı çok daha geniştir; 5 mm x-yüksekliğinde ortalama
#:      harf ilerlemesi ~4,5 mm olur ve 174 mm'lik yazı alanına ~38 karakter
#:      girer. Uzun satır kullanıcıyı yazıyı sıkıştırmaya zorlar, o da
#:      ölçümleri bozar.
#:
#: `tests/test_template.py` her iki kısıtı da doğrular.
DEFAULT_CORPUS: list[str] = [
    # -- düz metin: küçük harf yoğunluğu ------------------------------------
    "Pijamalı hasta yağız şoföre güvendi.",
    "Öfkeli müzisyen bahçede uğraşıp yoruldu.",
    "Şu ağır jeolog Türkçe konuşmayı sever.",
    "Fazla vitrin ışığı gözleri kamaştırdı.",
    "Yavaş kağnı bozkırda toz kaldırıp gitti.",
    "Genç avukat dosyayı sabırla özetledi.",
    "Sarı zarftaki mektup buruşmuş görünüyor.",
    "Deniz kıyısında oturup uzun düşündüm.",
    "Kahverengi tilki çitten hızla atladı.",
    "Bahçıvan fidanı özenle söküp yerine dikti.",
    "Çocuklar sokakta gülüşerek koşuyordu.",
    "Ihlamur ağacının gölgesinde uyukladık.",
    "Öğretmen sınıfa yeni bir soru yöneltti.",
    "Rüzgar perdeyi savurup pencereyi çarptı.",
    "Yorgun işçi çayını yudumlayıp dinlendi.",
    "Küçük ejderha mağarasında hazine sakladı.",
    # -- seyrek harfler için hedefli satırlar --------------------------------
    "Jandarma jipi jeneratörü taşıyordu.",
    "Japon jokey jübile için jimnastik yaptı.",
    "Baraj projesi jüri raporuyla onaylandı.",
    "Quiz, xenon, watt yabancı sözcüklerdir.",
    "wolfram, sandwich, taxi, query, quiz",
    "wax, fix, quorum, web, quartz, aqua, oxit",
    # -- büyük harfler -------------------------------------------------------
    # Alfabe satırı üç kez tekrarlanır: Ğ gibi hiçbir Türkçe sözcüğün başında
    # bulunmayan harfler için tek örnek kaynağı budur.
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ QWX",
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ QWX",
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ QWX",
    "Ankara Bursa Çorum Diyarbakır Edirne",
    "Fethiye Giresun Hatay Isparta İzmir",
    "Jandarma Konya Malatya Niğde Ordu Ölçü",
    "Pazar Rize Samsun Şırnak Trabzon Uşak",
    "Ünye Van Yozgat Zonguldak Quebec Wien",
    "Xerox Quantum Wolfram Vitrin Ünlem Zarf",
    # -- rakam ve noktalama --------------------------------------------------
    "0123456789 0123456789 0123456789",
    "1453 1923 2024 3,14 1.000 %50 42 87",
    "6 7 8 9 tane 5 kilo 3 litre 90 derece",
    "Nokta. Virgül, noktalı; iki nokta:",
    "Liste: bir; iki; üç (dört) beş. Son!",
    "Ünlem! Soru? Tire- (parantez) 'tek'",
    "\"Çift tırnak\" ile 'tek tırnak' örneği.",
    "Ünlem!! Soru?? Tire-- (yine) ; : ' \"",
]


def coverage(lines: list[str], charset: str) -> dict[str, int]:
    """Verilen satırlarda charset'teki her karakterin kaç kez geçtiğini sayar."""
    counts = dict.fromkeys(charset, 0)
    for line in lines:
        for ch in line:
            if ch in counts:
                counts[ch] += 1
    return counts


def missing_chars(lines: list[str], charset: str, minimum: int = 1) -> list[str]:
    """Korpusta `minimum` kereden az geçen karakterleri döndürür."""
    counts = coverage(lines, charset)
    return [ch for ch, n in counts.items() if n < minimum]


# --------------------------------------------------------------------------
# Geometri
# --------------------------------------------------------------------------

#: Sayfa ölçüleri ve yerleşim, milimetre cinsinden.
PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
MARGIN_MM = 12.0
MARKER_MM = 13.0
CONTENT_TOP_MM = 35.0
CONTENT_BOTTOM_MM = 297.0 - 32.0

REF_TEXT_MM = 4.6          # basılı örnek metnin yüksekliği
REF_GAP_MM = 2.0           # örnek metin ile yazı alanı arası
ASC_TO_XHEIGHT_MM = 4.0    # ascender çizgisi ile x-yüksekliği çizgisi arası
XHEIGHT_MM = 5.0           # x-yüksekliği bandı
DESC_MM = 4.0              # taban çizgisi altındaki descender payı
BAND_GAP_MM = 3.0          # bantlar arası boşluk

WRITE_MM = ASC_TO_XHEIGHT_MM + XHEIGHT_MM + DESC_MM
BAND_MM = REF_TEXT_MM + REF_GAP_MM + WRITE_MM + BAND_GAP_MM

#: Basım tonları. Kalem mürekkebi tipik olarak 90'ın altındadır; buradaki
#: değerler eşiklemede rahatça elenecek kadar açık seçildi.
INK_REF_TEXT = 150
INK_BASELINE = 168
INK_XHEIGHT = 205
INK_EDGE = 224

ARUCO_DICT = cv2.aruco.DICT_4X4_50

#: Sözlükte 50 işaret var, sayfa başına 4 tanesi kullanılıyor.
MAX_PAGES = 12


@dataclass
class Band:
    """Sayfadaki tek bir yazı bandı. Koordinatlar kanonik sayfa pikselidir."""

    index: int
    text: str
    #: Basılı örnek metnin kutusu (x0, y0, x1, y1) — maskelenip atılır.
    ref_rect: tuple[int, int, int, int]
    #: El yazısının bulunduğu kutu (x0, y0, x1, y1).
    write_rect: tuple[int, int, int, int]
    #: Basılı kılavuz çizgilerinin y konumları.
    ascender_y: int
    xheight_y: int
    baseline_y: int
    descender_y: int


@dataclass
class SheetSpec:
    """Bir çalışma sayfasının tam geometrisi."""

    page_index: int
    width: int
    height: int
    dpi: int
    marker_ids: list[int]
    #: Her işaretin kanonik sayfadaki dört köşesi, ArUco sırasıyla
    #: (sol-üst, sağ-üst, sağ-alt, sol-alt).
    marker_corners: list[list[list[float]]]
    bands: list[Band] = field(default_factory=list)

    # -- serileştirme ------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "SheetSpec":
        raw = json.loads(text)
        bands = [Band(**b) for b in raw.pop("bands", [])]
        for band in bands:
            band.ref_rect = tuple(band.ref_rect)      # type: ignore[assignment]
            band.write_rect = tuple(band.write_rect)  # type: ignore[assignment]
        return cls(bands=bands, **raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SheetSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- yardımcılar -------------------------------------------------------

    @property
    def xheight_px(self) -> float:
        """Basılı x-yüksekliğinin piksel karşılığı."""
        if not self.bands:
            return mm_to_px(XHEIGHT_MM, self.dpi)
        band = self.bands[0]
        return float(band.baseline_y - band.xheight_y)


@dataclass
class SheetSet:
    """Bir çalışma sayfası takımı (çok sayfalı olabilir)."""

    sheets: list[SheetSpec]

    def to_json(self) -> str:
        return json.dumps(
            {"sheets": [json.loads(s.to_json()) for s in self.sheets]},
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SheetSet":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(sheets=[SheetSpec.from_json(json.dumps(s)) for s in raw["sheets"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def mm_to_px(mm: float, dpi: int) -> float:
    return mm * dpi / 25.4


# --------------------------------------------------------------------------
# Yazı tipi bulma (basılı örnek metin için)
# --------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def find_reference_font(size: int) -> ImageFont.FreeTypeFont:
    """Türkçe karakterleri olan bir sistem fontu bulur."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError(
        "Örnek metni basacak bir sistem fontu bulunamadı. "
        "handwrite.template._FONT_CANDIDATES listesine bir .ttf yolu ekleyin."
    )


# --------------------------------------------------------------------------
# Sayfa çizimi
# --------------------------------------------------------------------------


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    y: int,
    x0: int,
    x1: int,
    color: int,
    dash: int = 9,
    gap: int = 7,
) -> None:
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + dash, x1), y)], fill=color, width=1)
        x += dash + gap


def _place_markers(page: np.ndarray, dpi: int, ids: list[int]) -> list[list[list[float]]]:
    """Dört köşeye ArUco işareti basar ve kanonik köşe koordinatlarını döndürür."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    size = int(round(mm_to_px(MARKER_MM, dpi)))
    margin = int(round(mm_to_px(MARGIN_MM, dpi)))
    h, w = page.shape

    positions = [
        (margin, margin),                          # id[0] sol-üst
        (w - margin - size, margin),               # id[1] sağ-üst
        (w - margin - size, h - margin - size),    # id[2] sağ-alt
        (margin, h - margin - size),               # id[3] sol-alt
    ]

    corners: list[list[list[float]]] = []
    for marker_id, (x, y) in zip(ids, positions):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        page[y : y + size, x : x + size] = img
        # ArUco köşe sırası: sol-üst, sağ-üst, sağ-alt, sol-alt.
        # generateImageMarker'ın çıktısında işaretin dış sınırı tam olarak
        # yerleştirdiğimiz karedir; son piksel dahil olduğu için size-1.
        corners.append(
            [
                [float(x), float(y)],
                [float(x + size - 1), float(y)],
                [float(x + size - 1), float(y + size - 1)],
                [float(x), float(y + size - 1)],
            ]
        )
    return corners


def build_sheet(
    lines: list[str],
    dpi: int = 200,
    page_index: int = 0,
    title: str | None = None,
) -> tuple[Image.Image, SheetSpec]:
    """Tek bir çalışma sayfası görüntüsü ve geometrisini üretir.

    `lines` bu sayfaya sığacak kadar satır içermelidir; sığmayanlar atılır.
    Çok sayfa için `build_sheets` kullanın.
    """
    w = int(round(mm_to_px(PAGE_W_MM, dpi)))
    h = int(round(mm_to_px(PAGE_H_MM, dpi)))
    page = np.full((h, w), 255, np.uint8)

    # Her sayfa kendi işaret kimliklerini kullanır. Böylece bir fotoğrafın hangi
    # sayfaya ait olduğu görüntüden okunur; kullanıcı fotoğrafları sıraya
    # dizmek ya da adlandırmak zorunda kalmaz.
    marker_ids = [4 * page_index + k for k in range(4)]
    if marker_ids[-1] >= 50:
        raise ValueError(
            f"{MAX_PAGES} sayfadan fazlası desteklenmiyor (ArUco sözlüğü 50 işaret içeriyor)."
        )
    marker_corners = _place_markers(page, dpi, marker_ids)

    image = Image.fromarray(page)
    draw = ImageDraw.Draw(image)

    left = int(round(mm_to_px(MARGIN_MM + 6.0, dpi)))
    right = w - left
    ref_font = find_reference_font(int(round(mm_to_px(REF_TEXT_MM * 0.82, dpi))))

    band_h = mm_to_px(BAND_MM, dpi)
    top = mm_to_px(CONTENT_TOP_MM, dpi)
    max_bands = int((mm_to_px(CONTENT_BOTTOM_MM, dpi) - top) // band_h)

    bands: list[Band] = []
    for i, text in enumerate(lines[:max_bands]):
        y0 = top + i * band_h
        ref_y0 = int(round(y0))
        ref_y1 = int(round(y0 + mm_to_px(REF_TEXT_MM, dpi)))
        write_y0 = int(round(y0 + mm_to_px(REF_TEXT_MM + REF_GAP_MM, dpi)))

        asc_y = write_y0
        xh_y = int(round(write_y0 + mm_to_px(ASC_TO_XHEIGHT_MM, dpi)))
        base_y = int(round(write_y0 + mm_to_px(ASC_TO_XHEIGHT_MM + XHEIGHT_MM, dpi)))
        desc_y = int(round(base_y + mm_to_px(DESC_MM, dpi)))

        # Kılavuz çizgileri: taban çizgisi en koyu, kullanıcı onu takip etsin.
        _draw_dashed_line(draw, asc_y, left, right, INK_EDGE, dash=5, gap=11)
        _draw_dashed_line(draw, xh_y, left, right, INK_XHEIGHT)
        draw.line([(left, base_y), (right, base_y)], fill=INK_BASELINE, width=1)
        _draw_dashed_line(draw, desc_y, left, right, INK_EDGE, dash=5, gap=11)

        draw.text((left, ref_y0), text, font=ref_font, fill=INK_REF_TEXT)

        bands.append(
            Band(
                index=i,
                text=text,
                ref_rect=(left - 4, ref_y0 - 2, right, ref_y1 + 2),
                write_rect=(left, asc_y - 2, right, desc_y + 2),
                ascender_y=asc_y,
                xheight_y=xh_y,
                baseline_y=base_y,
                descender_y=desc_y,
            )
        )

    # Sayfa başlığı — işaretlerin dışında kalır, maskelenmesine gerek yok.
    head_font = find_reference_font(int(round(mm_to_px(3.4, dpi))))
    heading = title or "handwrite — örnek metni kendi el yazınızla, taban çizgisini takip ederek yazın"
    draw.text(
        (left, int(round(mm_to_px(MARGIN_MM + MARKER_MM + 2.5, dpi)))),
        f"{heading}   ·   sayfa {page_index + 1}",
        font=head_font,
        fill=INK_REF_TEXT,
    )

    spec = SheetSpec(
        page_index=page_index,
        width=w,
        height=h,
        dpi=dpi,
        marker_ids=marker_ids,
        marker_corners=marker_corners,
        bands=bands,
    )
    return image, spec


def bands_per_page(dpi: int = 200) -> int:
    """Bir sayfaya sığan bant sayısı."""
    top = mm_to_px(CONTENT_TOP_MM, dpi)
    return int((mm_to_px(CONTENT_BOTTOM_MM, dpi) - top) // mm_to_px(BAND_MM, dpi))


def writable_mask(spec: SheetSpec, expand: float = 0.30) -> np.ndarray:
    """El yazısının bulunabileceği bölgeleri işaretler.

    Kullanıcıdan çizgilerin üstüne yazması istendiği için yazı alanlarının
    dışında kalan hiçbir koyu piksel el yazısı değildir: köşe işaretleri, sayfa
    başlığı, kağıdın kenarındaki gölge, kahve lekesi. Bunları en baştan elemek,
    sonraki her adımı bu tür çöpten korur.

    `expand`, yazı bandının yüksekliğinin oranı kadar dikey pay bırakır; büyük
    yazan ve kılavuzu taşıran kullanıcıların harfleri kırpılmasın diye.
    """
    mask = np.zeros((spec.height, spec.width), dtype=bool)
    for band in spec.bands:
        x0, y0, x1, y1 = band.write_rect
        pad = int(round(expand * (y1 - y0)))
        mask[max(0, y0 - pad) : min(spec.height, y1 + pad), max(0, x0) : min(spec.width, x1)] = True
    return mask


def render_template(spec: SheetSpec) -> np.ndarray:
    """Bir SheetSpec'in kanonik basılı görüntüsünü yeniden üretir.

    Ön işleme adımı bunu "şablon çıkarma" için kullanır: düzeltilmiş taramada
    koyu olan bir piksel şablonda da koyuysa bu basılı bir işarettir (kılavuz
    çizgisi veya örnek metin), el yazısı değil. Böylece basılı içeriği kaba
    dikdörtgen maskelerle silmek zorunda kalmayız — kullanıcı kılavuz
    çizgisinin üstüne yazdığında harfinin bir parçası kesilmez.
    """
    image, _ = build_sheet(
        [band.text for band in spec.bands], dpi=spec.dpi, page_index=spec.page_index
    )
    return np.asarray(image.convert("L"))


def build_sheets(
    lines: list[str] | None = None,
    dpi: int = 200,
) -> tuple[list[Image.Image], SheetSet]:
    """Korpusu sayfalara bölerek tam bir çalışma sayfası takımı üretir."""
    lines = lines or DEFAULT_CORPUS
    per_page = bands_per_page(dpi)
    images: list[Image.Image] = []
    specs: list[SheetSpec] = []
    for page_index, start in enumerate(range(0, len(lines), per_page)):
        image, spec = build_sheet(lines[start : start + per_page], dpi, page_index)
        images.append(image)
        specs.append(spec)
    return images, SheetSet(specs)
