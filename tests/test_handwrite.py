"""handwrite testleri.

Testlerin çoğu geliştirme sırasında gerçekten yapılmış hataları kilitler:
makaslama işaret konvansiyonu, potrace'in mürekkep yönü, salt büyük harfli
satırlarda x-yüksekliği ölçümü, varyant seçiminin hasarlı örnekleri tercih
etmesi. Hepsinin ortak özelliği sessizce yanlış çalışmalarıydı — kod hata
vermiyor, sadece kötü font üretiyordu.
"""

from __future__ import annotations

import numpy as np
import pytest

from handwrite.config import CHARSET, FOREIGN_LOWER, TURKISH_LOWER, Config
from handwrite.glyph import (
    Glyph,
    _select_variants,
    is_plausible,
    measure_xheights,
)
from handwrite.lines import estimate_slant, shear_about_baseline
from handwrite.segment import CharBox, Priors, _tokenize, split_span
from handwrite.template import (
    DEFAULT_CORPUS,
    SheetSet,
    build_sheet,
    build_sheets,
    coverage,
    missing_chars,
    render_template,
    writable_mask,
)
from handwrite.vectorize import trace_fidelity, trace_mask


# --------------------------------------------------------------------------
# Korpus ve çalışma sayfası
# --------------------------------------------------------------------------


def test_korpus_tum_karakterleri_kapsar():
    assert missing_chars(DEFAULT_CORPUS, CHARSET, 1) == []


def test_korpus_yeterli_ornek_icerir():
    """Tek örnekli bir harf için varyant üretilemez; font sahte görünür."""
    counts = coverage(DEFAULT_CORPUS, CHARSET)
    lower = TURKISH_LOWER + FOREIGN_LOWER
    thin_lower = {c: n for c, n in counts.items() if c in lower and n < 5}
    thin_other = {c: n for c, n in counts.items() if c not in lower and n < 3}
    assert not thin_lower, f"küçük harfler az geçiyor: {thin_lower}"
    assert not thin_other, f"diğer karakterler az geçiyor: {thin_other}"


def test_korpus_satirlari_yazi_alanina_sigar():
    """5 mm x-yüksekliğinde el yazısı satır başına ~38 karakter alır."""
    assert max(len(line) for line in DEFAULT_CORPUS) <= 42


def test_sayfa_geometrisi_json_turu_atlar():
    import json
    import tempfile
    from pathlib import Path

    _, sheets = build_sheets(DEFAULT_CORPUS[:12])
    text = sheets.to_json()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        path.write_text(text, encoding="utf-8")
        loaded = SheetSet.load(path)

    assert len(loaded.sheets) == len(sheets.sheets)
    first, other = sheets.sheets[0], loaded.sheets[0]
    assert first.marker_ids == other.marker_ids
    assert [b.text for b in first.bands] == [b.text for b in other.bands]
    assert first.bands[0].baseline_y == other.bands[0].baseline_y
    assert json.loads(text)["sheets"][0]["dpi"] == first.dpi


def test_her_sayfa_farkli_isaret_kullanir():
    """Fotoğrafın hangi sayfa olduğu görüntüden okunabilmeli."""
    _, sheets = build_sheets()
    seen: set[int] = set()
    for spec in sheets.sheets:
        assert not (seen & set(spec.marker_ids)), "işaret kimlikleri çakışıyor"
        seen |= set(spec.marker_ids)


def test_yazilabilir_maske_isaretleri_disarida_birakir():
    """Köşe işaretleri mürekkep sanılırsa ilk satırın hizalaması bozulur."""
    _, spec = build_sheet(DEFAULT_CORPUS[:5])
    mask = writable_mask(spec)
    for corners in spec.marker_corners:
        xs = [int(p[0]) for p in corners]
        ys = [int(p[1]) for p in corners]
        assert not mask[min(ys) : max(ys), min(xs) : max(xs)].any()


def test_basili_sablon_yazi_alanini_bos_birakmaz():
    """Şablon çıkarma için basılı kılavuzların gerçekten çizilmiş olması gerekir."""
    _, spec = build_sheet(DEFAULT_CORPUS[:5])
    template = render_template(spec)
    band = spec.bands[0]
    row = template[band.baseline_y, band.write_rect[0] : band.write_rect[2]]
    assert (row < 250).any(), "taban çizgisi basılmamış"


# --------------------------------------------------------------------------
# Eğim (slant) — işaret konvansiyonu
# --------------------------------------------------------------------------


def _slanted_bar(angle_deg: float, baseline: int = 80) -> np.ndarray:
    ink = np.zeros((100, 120), bool)
    for y in range(20, baseline):
        x = int(55 + (baseline - y) * np.tan(np.radians(angle_deg)))
        ink[y, x : x + 4] = True
    return ink


def test_makaslama_ustu_sola_tasir():
    """Pozitif açı üstü sola taşımalı; ters olsaydı eğim ikiye katlanırdı."""
    ink = np.zeros((100, 60), bool)
    ink[20:80, 28:32] = True
    sheared = shear_about_baseline(ink, 10.0, 80.0)
    top = np.flatnonzero(sheared[22]).mean()
    bottom = np.flatnonzero(sheared[78]).mean()
    assert top < bottom


def test_makaslama_taban_cizgisini_sabit_tutar():
    """Ölçtüğümüz ilerleme genişlikleri buna dayanıyor."""
    ink = np.zeros((100, 60), bool)
    ink[20:81, 28:32] = True
    sheared = shear_about_baseline(ink, 15.0, 80.0)
    assert np.flatnonzero(sheared[80]).mean() == pytest.approx(
        np.flatnonzero(ink[80]).mean(), abs=1.0
    )


@pytest.mark.parametrize("angle", [-15.0, 0.0, 12.0, 25.0])
def test_egim_tahmini_duzeltme_acisi_dondurur(angle):
    """Dönen değer eğimin kendisi değil, doğrudan uygulanacak düzeltmedir."""
    ink = _slanted_bar(angle)
    correction = estimate_slant(ink, 80.0, limit=40.0, step=1.0)
    fixed = shear_about_baseline(ink, correction, 80.0)
    top = np.flatnonzero(fixed[22]).mean()
    bottom = np.flatnonzero(fixed[78]).mean()
    assert abs(top - bottom) < 4.0, f"düzeltme sonrası hâlâ eğik ({angle}°)"


# --------------------------------------------------------------------------
# Kısıtlı bölme DP'si
# --------------------------------------------------------------------------


def test_bolme_bosluklara_oturur():
    """Kesim maliyeti boşlukta sıfırsa DP kesimi oraya koymalı."""
    cut_cost = np.ones(101) * 5.0
    cut_cost[30] = 0.0
    cut_cost[70] = 0.0
    bounds = split_span(0, 100, np.array([1.0, 1.0, 1.0]), cut_cost, Config().segment)
    assert bounds is not None
    assert bounds[0] == 0 and bounds[-1] == 100
    assert bounds[1] == 30 and bounds[2] == 70


def test_bolme_genislik_onselini_kullanir():
    """Mürekkep ipucu yoksa parçalar önsellerle orantılı bölünmeli."""
    cut_cost = np.zeros(121)
    bounds = split_span(0, 120, np.array([1.0, 2.0, 1.0]), cut_cost, Config().segment)
    assert bounds is not None
    widths = np.diff(bounds)
    assert widths[1] == pytest.approx(2 * widths[0], rel=0.15)


def test_bolme_sinir_sayisi_dogru():
    cut_cost = np.zeros(201)
    for count in (1, 2, 5, 9):
        bounds = split_span(0, 200, np.ones(count), cut_cost, Config().segment)
        assert bounds is not None and len(bounds) == count + 1


def test_belirtec_ayirma_bosluklari_birlestirir():
    assert _tokenize("ab  cd") == [("ab", False), ("  ", True), ("cd", False)]
    assert _tokenize(" a") == [(" ", True), ("a", False)]


def test_onseller_tabloya_dusuyor():
    priors = Priors(table={"m": 3.0})
    assert priors.width("m") == 3.0
    assert priors.width("i") < priors.width("a")


# --------------------------------------------------------------------------
# Vektörleştirme
# --------------------------------------------------------------------------


def test_iz_surme_murekkebi_dogru_yonde_okur():
    """potracer koyu pikselleri mürekkep sayar; ters verilirse sayfa glif olur."""
    mask = np.zeros((60, 60), bool)
    mask[15:45, 20:40] = True
    contours = trace_mask(mask, Config().font)
    assert contours, "kontur bulunamadı"

    xs = [p[0] for c in contours for p in c.points()]
    ys = [p[1] for c in contours for p in c.points()]
    # Kontur, mürekkebin sınırında olmalı — tuvalin tamamında değil.
    assert min(xs) > 10 and max(xs) < 50
    assert min(ys) > 5 and max(ys) < 55


def test_iz_surme_sekli_korur():
    mask = np.zeros((80, 60), bool)
    mask[10:70, 15:45] = True
    mask[30:50, 25:35] = False  # delik: "a" ve "o"nun gözü gibi
    assert trace_fidelity(mask, Config().font) > 0.93


def test_bos_maske_kontur_uretmez():
    assert trace_mask(np.zeros((20, 20), bool), Config().font) == []


# --------------------------------------------------------------------------
# Glif seçimi ve ölçüm
# --------------------------------------------------------------------------


def _glyph(char: str, width: int, height: int, ascent: float, scale: float = 10.0) -> Glyph:
    return Glyph(
        char=char,
        mask=np.ones((height, width), bool),
        scale=scale,
        bearing_px=1.0,
        ascent_px=ascent,
        advance_px=width + 2,
        line_index=0,
        char_index=0,
    )


def test_akla_yatkinlik_komsu_yutan_glifi_eler():
    """x-yüksekliğinde olması gereken bir harf iki katı boyda olamaz."""
    font = Config().font
    normal = _glyph("a", 5, 50, 50.0, scale=font.x_height / 50)
    swollen = _glyph("a", 5, 130, 130.0, scale=font.x_height / 50)
    assert is_plausible(normal, font)
    assert not is_plausible(swollen, font)


def test_akla_yatkinlik_ascender_ve_descenderi_kabul_eder():
    font = Config().font
    scale = font.x_height / 50
    assert is_plausible(_glyph("l", 4, 75, 75.0, scale), font)
    assert is_plausible(_glyph("p", 6, 78, 50.0, scale), font)


def test_x_yuksekligi_salt_buyuk_harfli_satirda_sismiyor():
    """Alfabe satırında profil platosu cap-height'tır; ölçüm oradan yapılamaz."""
    boxes = []
    for index, ch in enumerate("aecnorsu"):
        boxes.append(_box(ch, line=0, height=40, index=index))
    for index, ch in enumerate("ABCDEFGH"):
        boxes.append(_box(ch, line=1, height=56, index=index))

    measured, document = measure_xheights(boxes, min_samples=4)
    assert measured[0] == pytest.approx(40, abs=1)
    # 1. satırda x-yüksekliği harfi yok; belge medyanına düşmeli, 56'ya değil.
    assert measured.get(1, document) == pytest.approx(40, abs=1)


def _box(char: str, line: int, height: int, index: int) -> CharBox:
    return CharBox(
        line_index=line,
        char_index=index,
        char=char,
        mask=np.ones((height, 20), bool),
        x0=0,
        y0=0,
        cut_left=0,
        cut_right=24,
        baseline=float(height),
        xheight=float(height),
    )


def test_varyant_secimi_aykiri_ornegi_tercih_etmez():
    """Çeşitlilik, çekirdek havuzun *içinden* aranmalı.

    Sekiz benzer örnek ve bir bozuk örnek: "en uzak nokta" örneklemesi bozuğu
    seçerdi ve fontta hasarlı bir varyant çıkardı.
    """
    n = 9
    distances = np.full((n, n), 1.0)
    np.fill_diagonal(distances, 0.0)
    distances[8, :] = 20.0
    distances[:, 8] = 20.0
    distances[8, 8] = 0.0

    picks = _select_variants(distances, count=3, core_percentile=55.0)
    assert 8 not in picks, "hasarlı örnek varyant olarak seçildi"
    assert len(set(picks)) == 3


def test_varyant_secimi_az_ornekte_cokmez():
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    picks = _select_variants(distances, count=3, core_percentile=55.0)
    assert 1 <= len(picks) <= 2
    assert all(0 <= p < 2 for p in picks)
