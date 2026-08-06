"""Web arayüzü testleri.

Arayüz, son kullanıcının gördüğü tek yüz olduğu için buradaki testler
"çalışıyor mu"dan çok "kullanıcıyı yalnız bırakıyor mu" sorusuna bakar:
desteklenmeyen bir dosya atıldığında ne söylüyor, anahtar yokken bunu ne zaman
haber veriyor, yazacak metni olmayan biri ne görüyor.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from handwrite.config import CHARSET, FOREIGN_LOWER, TURKISH_LOWER
from handwrite.template import FREEFORM_SAMPLE, coverage
from handwrite.web import ACCEPTED, app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_arayuz_aciliyor(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    # Kullanıcının ilk ekranda görmesi gereken şeyler.
    assert "Fotoğrafı buraya sürükleyin" in body
    assert "Kamerayla çek" in body
    assert "kurşun kalem" in body, "kalem uyarısı arayüzde yok"


def test_kabul_edilen_turler_arayuzde_yaziyor(client):
    """Kullanıcı hangi dosyanın işe yaradığını denemeyle bulmamalı."""
    body = client.get("/").text
    for suffix in ("jpg", "png", "webp"):
        assert suffix in body


def test_anahtar_durumu_bastan_bildiriliyor(client):
    """Anahtar eksikse 20 saniyelik yüklemenin sonunda öğrenilmemeli."""
    payload = client.get("/api/status").json()
    assert "ai" in payload
    if not payload["ai"]:
        assert payload["reason"]


def test_ornek_metin_tum_karakterleri_kapsar(client):
    """Yazacak metni olmayan kullanıcıya verilen metin eksik harf bırakmamalı."""
    lines = client.get("/api/sample").json()["lines"]
    assert lines == FREEFORM_SAMPLE

    counts = coverage(lines, CHARSET)
    assert not [ch for ch, n in counts.items() if n == 0], "örnek metinde eksik karakter var"

    lower = TURKISH_LOWER + FOREIGN_LOWER
    thin = {ch: n for ch, n in counts.items() if ch in lower and n < 3}
    assert not thin, f"küçük harfler az geçiyor: {thin}"


def test_ornek_metin_tek_sayfaya_sigar():
    """Satırlar el yazısıyla bir A4'e sığacak kadar kısa olmalı."""
    assert len(FREEFORM_SAMPLE) <= 24
    assert max(len(line) for line in FREEFORM_SAMPLE) <= 46


def test_bozuk_dosya_ne_yapilacagini_soyluyor(client):
    """Hata mesajı sorunu değil, çözümü anlatmalı."""
    response = client.post(
        "/api/read",
        data={"family": "Test", "api_key": "sahte-anahtar"},
        files={"photos": ("not.jpg", b"bu bir goruntu degil", "image/jpeg")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "açılamadı" in detail
    assert ".jpg" in detail and ".png" in detail
    assert "HEIC" in detail, "iPhone kullanıcısına yol gösterilmiyor"


def test_desteklenen_turler_tutarli():
    """Sunucunun kabul ettiğiyle arayüzün gösterdiği aynı olmalı."""
    assert ".jpg" in ACCEPTED.values()
    assert "image/jpeg" in ACCEPTED and "image/png" in ACCEPTED


def test_uretilmemis_font_indirilemez(client):
    assert client.get("/api/font/olmayanoturum.ttf").status_code == 404
    assert client.get("/api/preview/olmayanoturum.png").status_code == 404


def test_yazisiz_gorsel_anlasilir_hata_verir(client):
    """Boş bir kağıt fotoğrafı çökme değil, açıklama üretmeli."""
    import time

    blank = Image.fromarray(np.full((900, 700), 245, np.uint8))
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")

    started = client.post(
        "/api/read",
        data={"family": "Test", "api_key": "sahte-anahtar"},
        files={"photos": ("bos.png", buffer.getvalue(), "image/png")},
    )
    assert started.status_code == 200
    job = started.json()["job"]

    for _ in range(200):
        state = client.get(f"/api/job/{job}").json()
        if state["done"]:
            break
        time.sleep(0.1)

    assert state["done"], "iş bitmedi"
    assert state.get("error"), "boş sayfada hata bildirilmedi"
    assert "result" not in state


def test_uzun_is_ilerleme_bildirir(client):
    """Kullanıcı 'takıldı mı, çalışıyor mu' sorusunu sormak zorunda kalmamalı."""
    import time

    blank = Image.fromarray(np.full((900, 700), 245, np.uint8))
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")

    job = client.post(
        "/api/read",
        data={"family": "Test", "api_key": "sahte"},
        files={"photos": ("bos.png", buffer.getvalue(), "image/png")},
    ).json()["job"]

    state = client.get(f"/api/job/{job}").json()
    assert "stage" in state and state["stage"]
    assert "elapsed" in state and isinstance(state["elapsed"], (int, float))
    assert "done" in state

    for _ in range(200):
        state = client.get(f"/api/job/{job}").json()
        if state["done"]:
            break
        time.sleep(0.1)
    assert state["elapsed"] >= 0


def test_olmayan_is_sorgulanamaz(client):
    assert client.get("/api/job/olmayan").status_code == 404


def test_uyarilar_ozetten_ayri_tasiniyor():
    """Uyarı metin dökümünün içinde kaybolmamalı; ayrı alanda gelmeli.

    "Fotoğraf çok küçük" gibi uyarılar kullanıcının tek hamlede düzeltebileceği
    şeyler; log'un ortasına gömülürse okunmuyor ve kötü fontun sebebi
    anlaşılmıyor.
    """
    from types import SimpleNamespace

    from handwrite.pipeline import Diagnostics
    from handwrite.web import _diagnostics_payload

    diagnostics = Diagnostics()
    diagnostics.warnings.append("Yazı fotoğrafta çok küçük görünüyor")
    result = SimpleNamespace(
        diagnostics=diagnostics,
        build=SimpleNamespace(characters=10, glyph_count=12),
        synthesis=None,
    )

    payload = _diagnostics_payload(result)
    assert payload["warnings"] == ["Yazı fotoğrafta çok küçük görünüyor"]
    assert not any("çok küçük" in row for row in payload["summary"]), (
        "uyarı hem özette hem uyarı alanında tekrarlanıyor"
    )


def test_uyari_arayuzde_gosteriliyor(client):
    """Uyarı alanı sayfada var ve dolduruluyor olmalı."""
    page = client.get("/").text
    assert 'id="warn"' in page
    assert "d.warnings" in page
