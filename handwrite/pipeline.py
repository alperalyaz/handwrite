"""Uçtan uca boru hattı: fotoğraflardan font dosyasına.

    fotoğraf → kayıt → mürekkep → satır → eğim → karakter → glif → kontur → TTF

Her adım kendi modülünde; burada yapılan iş onları sırayla çağırmak, sayfalar
arası ortak kararları (eğim gibi) tek bir yerden vermek ve kullanıcıya
gösterilecek teşhisi toplamaktır.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import Config
from .fontbuild import BuildReport, build_font, save_font
from .glyph import GlyphLibrary, apply_xheights, build_library, measure_xheights, normalize_all
from .lines import Line, deslant_line, detect_lines, lines_from_bands, page_slant
from .preprocess import RegistrationError, detect_markers, load_gray, prepare_page
from .segment import DocumentSegmentation, segment_document
from .synthesize import SynthesisReport, fill_missing
from .template import SheetSet, SheetSpec


@dataclass
class PageResult:
    """Tek bir sayfanın işlenme sonucu."""

    source: str
    page_index: int | None
    lines: int
    ink_pixels: int
    error: str | None = None


@dataclass
class Diagnostics:
    """Kullanıcıya gösterilecek teşhis bilgisi."""

    pages: list[PageResult] = field(default_factory=list)
    slant: float = 0.0
    stroke_width: float = 0.0
    #: Ölçülen belge x-yüksekliği (piksel).
    xheight_px: float = 0.0
    #: (sayfa, satır) → eleme sebebi.
    rejected_lines: dict[tuple[int, int], str] = field(default_factory=dict)
    #: Hiç örnek bulunamayan, dolayısıyla fonta giremeyen karakterler.
    missing_characters: list[str] = field(default_factory=list)
    #: Örneklerinin çoğu elenmiş, gözden geçirilmesi iyi olur.
    weak_characters: list[tuple[str, int, int]] = field(default_factory=list)
    total_samples: int = 0
    #: Serbest modda modelin okuduğu satırlar: (indeks, metin, güven).
    transcriptions: list[tuple[int, str, float]] = field(default_factory=list)
    #: Hizalama maliyeti yüksek olduğu için atılan glif sayısı.
    dropped_glyphs: int = 0
    #: Sayfada hiç geçmediği için üretilen karakterler.
    synthetic_characters: list[str] = field(default_factory=list)
    #: Glif denetiminin sonucu (denetim yapıldıysa).
    verification: object = None
    #: Kullanıcıya gösterilecek uyarılar (sonucu bozan ama düzeltilebilir
    #: durumlar).
    warnings: list[str] = field(default_factory=list)

    def summary_lines(self, include_warnings: bool = True) -> list[str]:
        """Özet satırları. Uyarıları ayrı gösteren arayüzler bunları isteyip
        tekrar etmemek için ``include_warnings=False`` geçer."""
        rows = list(self.warnings) if include_warnings else []
        rows += [
            f"sayfa: {len([p for p in self.pages if p.error is None])}/{len(self.pages)} okundu",
            f"eğim: {self.slant:.1f}°   kalem: {self.stroke_width:.1f} px   x-yüksekliği: {self.xheight_px:.0f} px",
            f"toplam karakter örneği: {self.total_samples}",
        ]
        if self.transcriptions:
            rows.append(f"okunan satır: {len(self.transcriptions)}")
        if self.rejected_lines:
            rows.append(f"elenen satır: {len(self.rejected_lines)}")
        if self.dropped_glyphs:
            rows.append(f"hizalaması şüpheli bulunup atılan glif: {self.dropped_glyphs}")
        if self.verification is not None:
            rows.extend(self.verification.summary_lines())
        if self.synthetic_characters:
            rows.append(
                f"üretilen karakter ({len(self.synthetic_characters)}): "
                + " ".join(self.synthetic_characters)
            )
        if self.missing_characters:
            rows.append("fontta olmayan karakter: " + " ".join(self.missing_characters))
        if self.weak_characters:
            weak = ", ".join(f"{c} ({k}/{s})" for c, s, k in self.weak_characters[:10])
            rows.append(f"az örnekli karakter: {weak}")
        return rows


@dataclass
class PipelineResult:
    """Boru hattının tam çıktısı."""

    font: object
    library: GlyphLibrary
    build: BuildReport
    diagnostics: Diagnostics
    segmentation: DocumentSegmentation | None = None
    synthesis: SynthesisReport | None = None


def identify_page(gray: np.ndarray, sheets: SheetSet) -> SheetSpec | None:
    """Fotoğraftaki köşe işaretlerinden hangi sayfa olduğunu bulur.

    Her sayfa farklı işaret kimlikleri taşıdığı için fotoğrafların sırası ya da
    dosya adı önemli değildir; kullanıcı hepsini bir kerede, karışık atabilir.
    """
    found = set(detect_markers(gray))
    if not found:
        return None
    best, best_hits = None, 0
    for spec in sheets.sheets:
        hits = len(found & set(spec.marker_ids))
        if hits > best_hits:
            best, best_hits = spec, hits
    return best if best_hits >= 3 else None


def collect_lines(
    images: list[tuple[str, np.ndarray]],
    sheets: SheetSet,
    cfg: Config,
) -> tuple[list[Line], dict[int, tuple[int, int]], Diagnostics]:
    """Bütün sayfalardan satırları toplar.

    Satırlar sayfalar arasında yeniden numaralandırılır: her sayfa kendi
    bantlarını 0'dan saydığı için, birleştirilmeden önce numaralar çakışırdı.
    Dönen eşleme, teşhiste satırı tekrar (sayfa, bant) olarak gösterebilmek
    içindir.
    """
    diagnostics = Diagnostics()
    lines: list[Line] = []
    origin: dict[int, tuple[int, int]] = {}
    strokes: list[float] = []

    for name, image in images:
        spec = identify_page(image, sheets)
        if spec is None:
            diagnostics.pages.append(
                PageResult(
                    source=name,
                    page_index=None,
                    lines=0,
                    ink_pixels=0,
                    error="sayfa köşe işaretleri okunamadı — dört köşenin de "
                    "kadrajda ve net olduğundan emin olun",
                )
            )
            continue

        try:
            page = prepare_page(image, spec, cfg.preprocess)
        except RegistrationError as exc:
            diagnostics.pages.append(
                PageResult(source=name, page_index=spec.page_index, lines=0, ink_pixels=0, error=str(exc))
            )
            continue

        page_lines = lines_from_bands(page, cfg.line)
        for line in page_lines:
            index = len(lines)
            origin[index] = (spec.page_index, line.index)
            lines.append(replace(line, index=index))

        strokes.append(page.stroke_width)
        diagnostics.pages.append(
            PageResult(
                source=name,
                page_index=spec.page_index,
                lines=len(page_lines),
                ink_pixels=int(page.ink.sum()),
            )
        )

    diagnostics.stroke_width = float(np.median(strokes)) if strokes else 1.0
    return lines, origin, diagnostics


def build_from_freeform(
    images: list[tuple[str, np.ndarray]],
    transcriber,
    cfg: Config | None = None,
    min_confidence: float = 0.35,
    progress=None,
    debug=None,
    verifier=None,
) -> PipelineResult:
    """Herhangi bir el yazısı sayfasından font üretir — şablon gerekmeden.

    Çalışma sayfası yolundan tek farkı, metnin nereden geldiğidir. Orada metni
    biz dayatıyorduk; burada sayfada ne yazdığını bir görsel model okuyor.
    Geri kalan her şey aynı deterministik zincirden geçer.

    Modelin hata payı boru hattının içinde soğurulur: okunan metin yanlışsa o
    kelimenin hizalama maliyeti yükselir ve glifleri atılır (bkz.
    `segment.drop_misaligned_words`). Yani model yanılabilir, font bozulmaz.
    """
    cfg = cfg or Config()
    diagnostics = Diagnostics()
    report = progress if callable(progress) else (lambda *a, **k: None)

    lines: list[Line] = []
    origin: dict[int, tuple[int, int]] = {}
    strokes: list[float] = []

    for page_index, (name, image) in enumerate(images):
        report(f"Sayfa {page_index + 1}/{len(images)} temizleniyor")
        if debug is not None:
            debug.raw(image, page_index)
            debug.paper(image, page_index)
        page = prepare_page(image, None, cfg.preprocess)
        page_lines = detect_lines(page, cfg.line)
        if debug is not None:
            debug.ink(page, page_index)
            debug.lines(page, page_lines, page_index)
            debug.strips(page_lines, page_index)
        for line in page_lines:
            index = len(lines)
            origin[index] = (page_index, line.index)
            lines.append(replace(line, index=index))
        strokes.append(page.stroke_width)
        diagnostics.pages.append(
            PageResult(
                source=name,
                page_index=page_index,
                lines=len(page_lines),
                ink_pixels=int(page.ink.sum()),
            )
        )

    if not lines:
        raise ValueError(
            "Sayfalarda yazı satırı bulunamadı. Fotoğrafın net ve yazının "
            "kağıttan belirgin şekilde koyu olduğundan emin olun."
        )

    _check_resolution(lines, diagnostics)

    diagnostics.stroke_width = float(np.median(strokes)) if strokes else 1.0

    # Okuma, eğim düzeltmesinden *önce* yapılır: model yazıyı doğal haliyle
    # daha iyi okur, dikleştirilmiş hali ona tanıdık gelmez.
    report(f"{len(lines)} satır bulundu, yazı okunuyor")
    if hasattr(transcriber, "on_progress"):
        transcriber.on_progress = lambda done, total: report(
            f"Yazı okunuyor — {done}/{total} satır"
        )
    readings = transcriber.transcribe_lines([line.ink for line in lines])
    diagnostics.transcriptions = [(r.index, r.text, r.confidence) for r in readings]
    if debug is not None:
        debug.transcription(lines, readings)

    usable: list[Line] = []
    for line, reading in zip(lines, readings):
        if not reading.usable or reading.confidence < min_confidence:
            diagnostics.rejected_lines[origin.get(line.index, (0, line.index))] = (
                "satır okunamadı" if not reading.usable
                else f"okuma güveni düşük ({reading.confidence:.2f})"
            )
            continue
        line.text = reading.text
        usable.append(line)

    if not usable:
        raise ValueError("Hiçbir satır okunamadı; fotoğrafın okunaklı olduğundan emin olun.")

    return _finish(usable, origin, diagnostics, cfg, report, debug, verifier)


def build_from_images(
    images: list[tuple[str, np.ndarray]],
    sheets: SheetSet,
    cfg: Config | None = None,
) -> PipelineResult:
    """Doldurulmuş çalışma sayfası fotoğraflarından font üretir."""
    cfg = cfg or Config()

    lines, origin, diagnostics = collect_lines(images, sheets, cfg)
    if not lines:
        raise ValueError(
            "Hiçbir sayfadan satır çıkarılamadı. Fotoğraflarda köşe işaretleri "
            "görünüyor mu ve yazı kılavuz çizgilerinin üstünde mi?"
        )
    return _finish(lines, origin, diagnostics, cfg)


def _finish(
    lines: list[Line],
    origin: dict[int, tuple[int, int]],
    diagnostics: Diagnostics,
    cfg: Config,
    progress=None,
    debug=None,
    verifier=None,
) -> PipelineResult:
    """Satırlardan fonta giden ortak kuyruk.

    Metnin nereden geldiği (şablondan mı, modelden mi) buradan itibaren önemsiz
    olur; iki yol da aynı deterministik zinciri kullanır.
    """
    report = progress if callable(progress) else (lambda *a, **k: None)

    # Eğim bütün sayfalardan ortak hesaplanır: aynı elin yazısı olduğu için
    # sayfa başına ayrı düzeltmek tutarsızlık üretirdi.
    report("Yazı eğimi ölçülüyor")
    diagnostics.slant = page_slant(lines, limit=cfg.line.slant_limit, coarse=cfg.line.slant_step)
    straight = [deslant_line(line, diagnostics.slant) for line in lines]

    report("Harfler ayrıştırılıyor")
    segmentation = segment_document(straight, diagnostics.stroke_width, cfg.segment)
    diagnostics.rejected_lines.update(
        {origin.get(index, (-1, index)): reason for index, reason in segmentation.rejected.items()}
    )
    diagnostics.dropped_glyphs = segmentation.dropped_words

    # x-yüksekliği, segmentasyondan sonra gerçek gliflerden ölçülür; satırın
    # mürekkep profilinden yapılan ilk tahmin salt büyük harfli satırlarda
    # yanılır ve o satırların glifleri küçük kalırdı.
    if debug is not None:
        debug.glyphs(segmentation.boxes)

    # Denetim: segmentasyon bir harfin neye benzediğini bilmez, bu yüzden
    # ürettiği her adayın gerçekten o harf olup olmadığı dışarıdan sorulur.
    # Geçemeyen fonta girmez; yeterli örneği kalmayan karakter üretilir.
    boxes = segmentation.boxes
    if verifier is not None and boxes:
        from .ai.verify import apply_verification

        report("Glifler denetleniyor")
        if hasattr(verifier, "on_progress"):
            verifier.on_progress = lambda done, total: report(
                f"Glifler denetleniyor — {done}/{total} karakter"
            )
        boxes, verification = apply_verification(boxes, verifier)
        diagnostics.verification = verification

    measured, document_xheight = measure_xheights(boxes)
    apply_xheights(boxes, measured, document_xheight)
    diagnostics.xheight_px = document_xheight

    glyphs = normalize_all(boxes, cfg.font)
    diagnostics.total_samples = len(glyphs)

    space = _measure_space(segmentation, measured, document_xheight, cfg)
    library = build_library(glyphs, cfg.glyph, cfg.font, space_advance=space)

    from .config import CHARSET

    # Tek bir doğal sayfa küçük harfleri bol verir ama büyük harflerin çoğunu,
    # rakamları ve noktalamayı vermez. Eksikleri kullanıcıdan ikinci bir sayfa
    # istemek yerine üretiyoruz.
    if cfg.synthesize_missing:
        report("Eksik karakterler üretiliyor")
        synthesis = fill_missing(library, cfg.font, cfg.glyph)
    else:
        synthesis = None

    present = set(library.characters)
    diagnostics.missing_characters = [ch for ch in CHARSET if ch not in present]
    diagnostics.synthetic_characters = sorted(
        ch for ch, entry in library.characters.items() if entry.synthetic
    )
    diagnostics.weak_characters = [
        (char, entry.seen, entry.kept)
        for char, entry in sorted(library.characters.items())
        if entry.kept < 3
    ]

    report("Font dosyası yazılıyor")
    font, build = build_font(library, cfg.font, cfg.glyph)
    return PipelineResult(
        synthesis=synthesis,
        font=font,
        library=library,
        build=build,
        diagnostics=diagnostics,
        segmentation=segmentation,
    )


#: Segmentasyonun güvenilir çalıştığı en küçük x-yüksekliği (piksel).
#: Altında harfler birbirinden ayırt edilemeyecek kadar az piksele düşer.
MIN_XHEIGHT_PX = 26.0


def _check_resolution(lines: list[Line], diagnostics: Diagnostics) -> None:
    """Yazının piksel olarak yeterince büyük olup olmadığını denetler.

    Çözünürlük, kullanıcının kolayca düzeltebileceği ama sonucu belirleyen bir
    etken. 1,4 MP'lik bir fotoğrafta A4 sayfa 4,8 piksel/mm'ye düşer ve tipik
    bir el yazısının x-yüksekliği 14 piksel eder — bir harfin gövdesine 14
    piksel düştüğünde kesim yerini birkaç piksel şaşırmak harfin yarısını
    götürür. Bunu sessizce kötü bir font üreterek geçiştirmek yerine söylüyoruz.
    """
    heights = [line.xheight for line in lines if line.xheight > 0]
    if not heights:
        return
    measured = float(np.median(heights))
    diagnostics.xheight_px = measured
    if measured < MIN_XHEIGHT_PX:
        diagnostics.warnings.append(
            f"Yazı fotoğrafta çok küçük görünüyor (x-yüksekliği {measured:.0f} piksel, "
            f"gereken en az {MIN_XHEIGHT_PX:.0f}). Harfler ayrıştırılamayacak kadar az "
            "piksele düşüyor. Daha yüksek çözünürlükte çekin ya da sayfaya daha "
            "yakından, kağıdı kadraja tam sığdıracak şekilde fotoğraflayın."
        )


def _measure_space(
    segmentation: DocumentSegmentation,
    xheights: dict[int, float],
    document_xheight: float,
    cfg: Config,
) -> float | None:
    """Boşluk karakterinin genişliğini ölçülen değerlerden em birimine çevirir.

    Boşluk, kişinin yazı ritminin en görünür parçalarından biri: kimi sıkışık
    kimi ferah yazar. Varsayılan bir değer koymak yerine ölçmek, fontla yazılan
    metnin el yazısına benzemesini belirgin şekilde artırır.

    Ölçek olarak gliflerin normalize edildiği x-yüksekliğinin *aynısı*
    kullanılır; başka bir ölçek boşlukları harflerle tutarsız hale getirirdi.
    """
    values: list[float] = []
    for index, scale in segmentation.scales.items():
        if scale <= 0 or index in segmentation.rejected:
            continue
        xheight = xheights.get(index, document_xheight)
        if xheight <= 0:
            continue
        space_px = segmentation.priors.space * scale
        values.append(space_px * (cfg.font.x_height / xheight))
    return float(np.median(values)) if values else None


# --------------------------------------------------------------------------
# Dosya tabanlı kolaylık sarmalayıcıları
# --------------------------------------------------------------------------


def build_from_paths(
    paths: list[str | Path],
    sheets_path: str | Path,
    cfg: Config | None = None,
) -> PipelineResult:
    """Dosya yollarından font üretir."""
    sheets = SheetSet.load(sheets_path)
    images = [(str(Path(p).name), load_gray(p)) for p in paths]
    return build_from_images(images, sheets, cfg)


def write_font(result: PipelineResult, path: str | Path) -> Path:
    save_font(result.font, path)
    return Path(path)
