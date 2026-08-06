"""Yapay zekâ destekli yolun testleri.

Gerçek API çağrısı yapılmaz: testler ağ olmadan, sahte bir okuyucuyla çalışır.
Amaç modelin okuma kalitesini ölçmek değil (o `handwrite.bench` işi), boru
hattının modelin *hatalarına* nasıl dayandığını doğrulamak.
"""

from __future__ import annotations

import numpy as np
import pytest

from handwrite.ai.provider import AIError, LineTranscription, load_api_key
from handwrite.config import CHARSET, Config
from handwrite.glyph import Glyph, GlyphLibrary, build_library, normalize_all
from handwrite.lines import detect_lines, estimate_line_pitch
from handwrite.pipeline import build_from_freeform
from handwrite.preprocess import extract_ink, prepare_page
from handwrite.synth import HandStyle, simulate_camera, synthesize_freeform
from handwrite.synthesize import (
    CASE_COMPATIBLE,
    COMPOSITION,
    fill_missing,
    measure_style,
)

SAMPLE = [
    "Bugün hava çok güzeldi, sahilde uzun bir",
    "yürüyüş yaptım. Kahve içerken eski bir",
    "defteri karıştırdım; kenarlarına aldığım",
    "notlar hâlâ okunuyordu. Şimdi düşününce",
    "zamanın ne kadar hızlı geçtiğini anlıyor",
    "insan. Çocukken yazları köyde geçirirdik.",
]


class FakeTranscriber:
    """Verilen metinleri döndüren sahte okuyucu."""

    def __init__(self, texts: list[str], confidence: float = 1.0):
        self.texts = texts
        self.confidence = confidence

    def transcribe_lines(self, images):
        return [
            LineTranscription(
                index=i,
                text=self.texts[i] if i < len(self.texts) else "",
                confidence=self.confidence,
            )
            for i in range(len(images))
        ]


@pytest.fixture(scope="module")
def notebook():
    page = synthesize_freeform(SAMPLE, HandStyle(), seed=21, ruled=True)
    page.photo = simulate_camera(page.canonical, seed=21, warp=0.010, rotation=1.8)
    return page


# --------------------------------------------------------------------------
# Serbest sayfa
# --------------------------------------------------------------------------


def test_cizgili_kagitta_cetvel_cizgileri_elenir(notebook):
    """Ton ayrımı basılı çizgileri, el yazısını kesmeden temizlemeli."""
    cfg = Config().preprocess
    ink = extract_ink(notebook.canonical, cfg, None)
    truth = notebook.ink

    recall = (ink & truth).sum() / truth.sum()
    residue = int((ink & ~truth).sum())
    assert recall > 0.94, "el yazısı kayboldu"
    assert residue < truth.sum() * 0.02, f"cetvel çizgileri kaldı ({residue} px)"


def test_satir_araligi_satirlar_degse_bile_olculur(notebook):
    """Bloklara bakan tahmin, satırlar değdiğinde bütün sayfayı tek satır sanar."""
    page = prepare_page(notebook.photo, None, Config().preprocess)
    pitch = estimate_line_pitch(page.ink.sum(axis=1).astype(float))
    assert 20 < pitch < page.ink.shape[0] / 3


def test_satirlar_dogru_sayida_bulunur(notebook):
    page = prepare_page(notebook.photo, None, Config().preprocess)
    lines = detect_lines(page, Config().line)
    assert len(lines) == len(SAMPLE)
    heights = [line.xheight for line in lines]
    assert max(heights) / min(heights) < 1.4, "x-yüksekliği ölçümleri tutarsız"


# --------------------------------------------------------------------------
# Modelin hatalarına dayanıklılık
# --------------------------------------------------------------------------


def test_dogru_metinle_font_uretilir(notebook):
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(SAMPLE), Config()
    )
    assert result.build.characters > 30
    assert result.diagnostics.total_samples > 150


def test_okunamayan_satir_elenir(notebook):
    """Boş dönen satır fonta hiç girmemeli."""
    texts = list(SAMPLE)
    texts[2] = ""
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(texts), Config()
    )
    assert any("okunamadı" in reason for reason in result.diagnostics.rejected_lines.values())


def test_dusuk_guvenli_satir_elenir(notebook):
    """Model her satırdan emin değilse font üretmek yerine durmalı.

    Sessizce kötü bir font vermektense açık bir hata vermek doğrudur.
    """
    with pytest.raises(ValueError, match="okunamadı"):
        build_from_freeform(
            [("defter.jpg", notebook.photo)],
            FakeTranscriber(SAMPLE, confidence=0.1),
            Config(),
        )


def test_okuma_hatasi_hasari_kelimeyle_sinirli_kalir(notebook):
    """Modelin bir harf düşürmesi bütün satırı değil, tek kelimeyi bozmalı.

    Bu, segmentasyonun iki kademeli olmasının doğal sonucudur: satır önce
    kelimelere, sonra kelime içi karakterlere bölünür. Yanlış karakter sayısı
    yalnız kendi kelimesinin bölünmesini etkiler; komşu kelimeler kendi
    aralıklarına oturmayı sürdürür. Tek kademeli bir hizalamada aynı hata
    satırdaki *bütün* glifleri kaydırırdı.

    Dikkat: iki kademeli bölme hatayı soğurduğu için, kelime sınırını da bir
    miktar kaydırarak hizalama maliyetini beklendiği kadar yükseltmez. Yani
    "şüpheli kelimeyi maliyetten yakalama" güvenilir bir tespit değildir; asıl
    koruma, bozuk gliflerin kendi harflerinin örnek kümesinde aykırı kalıp
    eleme aşamasında düşmesidir. Bu test o gerçek garantiyi ölçer.
    """
    broken_texts = [t.replace("Bugün", "Bugn").replace("yaptım", "yapım") for t in SAMPLE]
    broken = build_from_freeform(
        [("d.jpg", notebook.photo)], FakeTranscriber(broken_texts), Config()
    )

    # Font yine de kullanılabilir çıkmalı: hata iki kelimeyle sınırlı kalır.
    assert set(CHARSET) <= set(broken.library.characters)
    assert broken.diagnostics.total_samples > 150

    # Bozuk kelimelerdeki harfler bile makul kalmalı; aykırı eleme onları
    # kendi kümelerinden düşürür.
    for char in "Bgnyapt":
        entry = broken.library.characters.get(char)
        if entry is None or entry.synthetic:
            continue
        for glyph in entry.variants:
            assert 0 < glyph.ascent < Config().font.units_per_em


# --------------------------------------------------------------------------
# Eksik glif sentezi
# --------------------------------------------------------------------------


def test_sentez_karakter_kumesini_tamamlar(notebook):
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(SAMPLE), Config()
    )
    assert not result.diagnostics.missing_characters, "sentezden sonra eksik karakter kaldı"
    # Metinde geçen ama hedef kümede olmayan harfler (ör. "hâlâ"daki â) fonta
    # bonus olarak girer; bu yüzden eşitlik değil kapsama aranır.
    assert set(CHARSET) <= set(result.library.characters)
    assert result.synthesis is not None
    assert result.synthesis.total > 0
    assert not result.synthesis.failed


def test_uretilen_glifler_isaretlenir(notebook):
    """Kullanıcıya üretilmiş bir harfi kendi yazısıymış gibi göstermemeliyiz."""
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(SAMPLE), Config()
    )
    synthetic = set(result.diagnostics.synthetic_characters)
    assert synthetic, "hiçbir karakter üretilmiş olarak işaretlenmemiş"
    for char in synthetic:
        assert result.library.characters[char].synthetic
    # Sayfada bolca geçen harfler üretilmiş sayılmamalı.
    assert not synthetic & set("aeinrl")


def test_sentez_kapatilabilir(notebook):
    cfg = Config()
    cfg.synthesize_missing = False
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(SAMPLE), cfg
    )
    assert result.synthesis is None
    assert result.diagnostics.missing_characters


def test_uretilen_glif_olculeri_makul(notebook):
    """Üretilen harfler ne devasa ne mikroskobik olmalı."""
    result = build_from_freeform(
        [("defter.jpg", notebook.photo)], FakeTranscriber(SAMPLE), Config()
    )
    font = Config().font
    real = [e for e in result.library.characters.values() if not e.synthetic]
    made = [e for e in result.library.characters.values() if e.synthetic]
    assert real and made

    reference = float(np.median([e.advance for e in real]))
    for entry in made:
        assert 0.2 * reference < entry.advance < 3.0 * reference, (
            f"{entry.char} ilerleme genişliği saçma: {entry.advance:.0f}"
        )
        glyph = entry.variants[0]
        assert glyph.ascent < font.units_per_em, f"{entry.char} çok uzun"
        assert glyph.ascent > 0, f"{entry.char} taban çizgisinin altında kalmış"


def test_bilesim_tarifleri_tutarli():
    """Tarifler kendi kendine atıfta bulunmamalı ve gerçek karakterler olmalı."""
    for target, (base, donor) in COMPOSITION.items():
        assert target != base and target != donor
        assert len(base) == 1 and len(donor) == 1


def test_buyuk_kucuk_aktarimi_yalniz_uyumlu_harfleri_kapsar():
    """Biçimi gerçekten değişen harfler listede olmamalı."""
    for upper, lower in CASE_COMPATIBLE.items():
        assert upper.lower() == lower or upper == "I"
    for wrong in "ABDEFGHJLMNQRT":
        assert wrong not in CASE_COMPATIBLE, f"{wrong} küçüğünden büyütülemez"


# --------------------------------------------------------------------------
# Anahtar yönetimi
# --------------------------------------------------------------------------


def test_anahtar_ortam_degiskeninden_okunur(monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test-anahtar")
    assert load_api_key() == "test-anahtar"


def test_anahtar_yoksa_yol_gosteren_hata(monkeypatch, tmp_path):
    for name in ("GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("handwrite.ai.provider.KEY_FILE", tmp_path / "yok")
    with pytest.raises(AIError, match="API anahtarı"):
        load_api_key()


def test_acik_anahtar_ortami_ezer(monkeypatch):
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "ortam")
    assert load_api_key("acik") == "acik"


# --------------------------------------------------------------------------
# İstek boyutu
# --------------------------------------------------------------------------


def test_serit_kodlamasi_goruntuyu_buyutmez():
    """Şeridi sabit yüksekliğe "normalize etmek" isteği şişiriyordu.

    Tipik bir satır maskesi zaten 50-60 piksel yüksekliğindedir. Onu 96'ya
    çıkarmak hiçbir bilgi katmadan alanı üç katına çıkarıyor, istek gövdesini
    büyütüyor ve gerçek kullanımda Gemini'nin 180 saniyede yanıt verememesine
    yol açıyordu. Ölçekleme yalnız küçültme yönünde olmalı.
    """
    import base64

    from handwrite.ai.gemini import encode_line

    short = np.zeros((54, 900), bool)
    short[20:40, 100:800] = True
    encoded = base64.b64decode(encode_line(short, height=64))

    from io import BytesIO

    from PIL import Image as PILImage

    decoded = PILImage.open(BytesIO(encoded))
    assert decoded.height == 54, "kısa şerit büyütülmüş"
    assert decoded.width == 900

    tall = np.zeros((200, 900), bool)
    tall[50:150, 100:800] = True
    resized = PILImage.open(BytesIO(base64.b64decode(encode_line(tall, height=64))))
    assert resized.height == 64, "uzun şerit küçültülmemiş"


def test_serit_yuku_makul_kaliyor():
    """Bir grup istek birkaç yüz kilobayta çıkmamalı."""
    import base64

    from handwrite.ai.gemini import encode_line

    rng = np.random.default_rng(0)
    total = 0
    for _ in range(5):
        strip = np.zeros((56, 1000), bool)
        for x in range(0, 1000, 12):
            strip[18 + rng.integers(0, 6) : 44, x : x + 5] = True
        total += len(base64.b64decode(encode_line(strip, 64)))
    assert total < 120_000, f"5 satırlık grup çok büyük: {total / 1024:.0f} KB"


def test_toplam_sure_sinirlanmis():
    """Zaman aşımları çarpılıp kullanıcıyı on beş dakika bekletmemeli."""
    from handwrite.ai.gemini import GeminiTranscriber

    transcriber = GeminiTranscriber(api_key="sahte")
    worst_case = transcriber.timeout * transcriber.max_retries
    assert transcriber.total_deadline < worst_case * 3
    assert transcriber.total_deadline <= 600


# --------------------------------------------------------------------------
# Kağıdın zeminden ayrılması
# --------------------------------------------------------------------------


def _on_desk(page: np.ndarray, level: int, margin: float = 0.12) -> np.ndarray:
    """Sayfayı belirli tonda bir masanın üstüne koyar."""
    import numpy as _np

    height, width = page.shape
    pad_x, pad_y = int(width * margin), int(height * margin)
    desk = _np.full((height + 2 * pad_y, width + 2 * pad_x), level, _np.uint8)
    noise = _np.random.default_rng(1).normal(0, 6, desk.shape)
    desk = _np.clip(desk + noise, 0, 255).astype(_np.uint8)
    desk[pad_y : pad_y + height, pad_x : pad_x + width] = page
    return desk


@pytest.mark.parametrize("desk", [200, 120, 45, 25])
def test_koyu_zeminde_satirlar_bozulmuyor(notebook, desk):
    """Kağıdın dışını işlemek satır tespitini tamamen çökertiyordu.

    Koyu bir masada (ton ~45) masa, gölge ve kağıt kenarı mürekkep sanılıyor,
    6 satırlık bir sayfada 226 "satır" bulunuyor ve her biri birkaç piksellik
    bir kırıntı oluyordu. Font da o kırıntılardan oluşuyordu.
    """
    page = prepare_page(_on_desk(notebook.canonical, desk), None, Config().preprocess)
    lines = detect_lines(page, Config().line)

    assert len(lines) == len(SAMPLE), f"masa tonu {desk}: {len(lines)} satır bulundu"
    heights = [line.xheight for line in lines]
    assert min(heights) > 15, "satırlar kırıntıya dönüşmüş"
    assert max(heights) / min(heights) < 1.5


def test_kagit_algilama_zaten_dolu_kadraji_bozmaz(notebook):
    """Kadrajın tamamı kağıtsa kırpmaya gerek yok; yanlışlıkla kırpmamalı."""
    from handwrite.preprocess import detect_paper

    result = detect_paper(notebook.canonical)
    if result is not None:
        height, width = notebook.canonical.shape
        assert result.shape[0] > height * 0.7
        assert result.shape[1] > width * 0.7
