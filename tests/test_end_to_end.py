"""Uçtan uca testler: sentetik el yazısından gerçek bir font dosyasına.

Bu testler yavaştır (sentez pahalı) ama boru hattının tamamını tek parça
halinde doğrulayan tek yer burasıdır. Küçük bir sayfa kullanılır: amaç font
kalitesini ölçmek değil, zincirin her halkasının bağlı olduğunu göstermek.

Kalite ölçümü ayrı bir iştir ve `handwrite.bench` ile yapılır.
"""

from __future__ import annotations

import numpy as np
import pytest

from handwrite.bench import evaluate
from handwrite.config import Config
from handwrite.pipeline import build_from_images
from handwrite.preprocess import prepare_page
from handwrite.synth import HandStyle, synthesize_photo, synthesize_sheet
from handwrite.template import build_sheet, build_sheets

SHORT_CORPUS = [
    "abcçdefgğhıijklmnoöprsştuüvyz",
    "anne memnun sessiz kelimeler",
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
    "0123456789 ve noktalama .,;:!?",
]


@pytest.fixture(scope="module")
def small_sheet():
    _, spec = build_sheet(SHORT_CORPUS)
    return spec


@pytest.fixture(scope="module")
def synthesized(small_sheet):
    return synthesize_photo(small_sheet, HandStyle(), seed=7)


def test_on_isleme_el_yazisini_kaybetmez(small_sheet, synthesized):
    """Basılı içerik tamamen elenmeli, el yazısının tamamı kalmalı."""
    page = prepare_page(synthesized.photo, small_sheet, Config().preprocess)
    truth = synthesized.ink
    recovered = page.ink

    overlap = int((truth & recovered).sum())
    assert overlap / truth.sum() > 0.95, "el yazısı mürekkebi kayboldu"

    # Basılı örnek metnin bulunduğu bantlarda hiç mürekkep olmamalı.
    leak = sum(
        int(recovered[b.ref_rect[1] : b.ref_rect[3], b.ref_rect[0] : b.ref_rect[2]].sum())
        for b in small_sheet.bands
    )
    assert leak == 0, f"basılı içerik mürekkep sanıldı ({leak} piksel)"


def test_sayfa_kaydi_dort_isareti_de_bulur(small_sheet, synthesized):
    page = prepare_page(synthesized.photo, small_sheet, Config().preprocess)
    assert sorted(page.found_markers) == sorted(small_sheet.marker_ids)


def test_egim_dogru_olculur(small_sheet):
    """Bilinen eğimle üretilen sayfada tahmin o değeri bulmalı."""
    page = synthesize_photo(small_sheet, HandStyle(slant_deg=14.0), seed=3)
    report = evaluate(page, small_sheet, Config())
    assert report.slant == pytest.approx(14.0, abs=3.0)


def test_segmentasyon_dogrulugu(small_sheet, synthesized):
    """Piksel düzeyinde ölçüm: gliflere doğru mürekkep atanıyor mu?"""
    report = evaluate(synthesized, small_sheet, Config())
    summary = report.summary()
    assert summary["değerlendirilen"] > 80
    assert summary["saflık_medyan"] > 0.85
    assert summary["kapsama_medyan"] > 0.85
    assert summary["kötü_<0.40"] < 0.25


def test_font_uretimi(small_sheet, synthesized, tmp_path):
    """Zincirin tamamı: fotoğraf → kurulabilir bir .ttf."""
    from fontTools.ttLib import TTFont

    from handwrite.template import SheetSet

    result = build_from_images(
        [("test.jpg", synthesized.photo)], SheetSet([small_sheet]), Config()
    )

    assert result.build.characters > 40
    assert not result.build.empty_outlines

    path = tmp_path / "Test-Regular.ttf"
    result.font.save(str(path))
    assert path.stat().st_size > 10_000

    font = TTFont(str(path))
    cmap = font.getBestCmap()
    for ch in "abcdefgorstuz":
        assert ord(ch) in cmap, f"{ch} fontta yok"
    for ch in "çğıöşü":
        assert ord(ch) in cmap, f"Türkçe {ch} fontta yok"

    # Varyantlar ve onları döndüren özellik yerinde mi?
    assert "GSUB" in font
    tags = {r.FeatureTag for r in font["GSUB"].table.FeatureList.FeatureRecord}
    assert "calt" in tags
    assert any(".alt" in name for name in font.getGlyphOrder())


def test_glif_olculeri_tipografik_olarak_tutarli(small_sheet, synthesized, tmp_path):
    """x-yüksekliği harfleri, ascender'lar ve büyük harfler doğru boyda olmalı."""
    from fontTools.ttLib import TTFont

    from handwrite.template import SheetSet

    result = build_from_images(
        [("test.jpg", synthesized.photo)], SheetSet([small_sheet]), Config()
    )
    path = tmp_path / "m.ttf"
    result.font.save(str(path))

    font = TTFont(str(path))
    glyf = font["glyf"]

    def top(name: str) -> int:
        glyph = glyf[name]
        glyph.recalcBounds(glyf)
        return glyph.yMax

    xheight = np.median([top(n) for n in ("a", "o", "n", "e") if n in glyf])
    ascender = np.median([top(n) for n in ("b", "d", "l", "k") if n in glyf])
    capital = np.median([top(n) for n in ("A", "B", "H", "K") if n in glyf])

    cfg = Config().font
    assert xheight == pytest.approx(cfg.x_height, rel=0.20)
    assert ascender > xheight * 1.15, "ascender'lar x-yüksekliğinden yeterince uzun değil"
    # Salt büyük harfli satırda x-yüksekliği ölçümünün şişmesi bu testi düşürür.
    assert capital > xheight * 1.15, "büyük harfler küçük kalmış"


def test_sayfa_tanima_fotograf_sirasindan_bagimsiz():
    """Farklı sayfaların işaretleri farklı olduğu için sıra önemsiz olmalı."""
    from handwrite.pipeline import identify_page

    _, sheets = build_sheets(SHORT_CORPUS * 3)
    assert len(sheets.sheets) >= 2

    for spec in sheets.sheets:
        page = synthesize_sheet(spec, HandStyle(), seed=1)
        found = identify_page(page.canonical, sheets)
        assert found is not None
        assert found.page_index == spec.page_index
