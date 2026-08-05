"""Segmentasyon doğruluğunun ölçümü.

Sentetik sayfa üretirken her mürekkep pikselinin hangi karakter tarafından
çizildiğini kaydediyoruz. Bu sayede segmentasyonu göz kararı değil piksel
sayarak değerlendirebiliyoruz:

* **saflık (purity)** — bir glife atanan mürekkebin yüzde kaçı gerçekten o
  harfe ait? Düşükse kesim komşu harften parça kopartıyor demektir.
* **kapsama (coverage)** — o harfin mürekkebinin yüzde kaçı glife girdi?
  Düşükse kesim harfin bir parçasını komşusuna bırakıyor demektir.

İkisini birden bildirmek gerekir: yalnız saflığa bakılsa çok dar kesmek,
yalnız kapsamaya bakılsa çok geniş kesmek "iyi" görünürdü.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .lines import Line, deslant_line, deslant_padding, lines_from_bands, page_slant, shear_array
from .preprocess import prepare_page
from .segment import CharBox, DocumentSegmentation, segment_document
from .synth import SynthPage
from .template import SheetSpec


@dataclass
class SegmentationReport:
    """Bir sayfanın segmentasyon değerlendirmesi."""

    total_predicted: int = 0
    total_truth: int = 0
    #: Metni tamamlanmadan biten bantlar (kullanıcı satırı bitirmemiş sayılır);
    #: bunlar ölçüm dışı bırakılır çünkü hizalama zaten imkânsızdır.
    truncated_bands: int = 0
    evaluated: int = 0
    purity: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)
    per_char: dict[str, list[float]] = field(default_factory=dict)
    slant: float = 0.0
    #: Segmentasyonun kendi teşhisiyle güvenilmez bulup elediği satırlar.
    rejected_lines: dict[int, str] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        purity = np.array(self.purity) if self.purity else np.zeros(1)
        coverage = np.array(self.coverage) if self.coverage else np.zeros(1)
        score = np.minimum(purity, coverage)
        return {
            "değerlendirilen": float(self.evaluated),
            "kesilen_bant": float(self.truncated_bands),
            "saflık_ort": float(purity.mean()),
            "saflık_medyan": float(np.median(purity)),
            "kapsama_ort": float(coverage.mean()),
            "kapsama_medyan": float(np.median(coverage)),
            "iyi_>0.80": float((score > 0.80).mean()),
            "iyi_>0.60": float((score > 0.60).mean()),
            "kötü_<0.40": float((score < 0.40).mean()),
            "eğim": self.slant,
        }

    def worst_chars(self, limit: int = 12) -> list[tuple[str, float, int]]:
        """En kötü sonuç veren karakterleri (karakter, ortalama skor, örnek) döndürür."""
        rows = [(ch, float(np.mean(v)), len(v)) for ch, v in self.per_char.items() if v]
        rows.sort(key=lambda r: r[1])
        return rows[:limit]


def segment_page(
    photo: np.ndarray, spec: SheetSpec, cfg: Config
) -> tuple[list[Line], list[Line], DocumentSegmentation, float, float]:
    """Bir fotoğrafı karakter kutularına kadar işler.

    Döndürür: (ham satırlar, eğimi düzeltilmiş satırlar, segmentasyon, eğim, kalem).
    Ham satırlar da döndürülür çünkü ölçüm, gerçek etiket haritasını aynı
    dönüşümden geçirmek için kırpma geometrisine ihtiyaç duyar.
    """
    page = prepare_page(photo, spec, cfg.preprocess)
    raw_lines = lines_from_bands(page, cfg.line)
    slant = page_slant(raw_lines, limit=cfg.line.slant_limit, coarse=cfg.line.slant_step)
    lines = [deslant_line(line, slant) for line in raw_lines]

    result = segment_document(lines, page.stroke_width, cfg.segment)
    return raw_lines, lines, result, slant, page.stroke_width


def _deslant_labels(labels: np.ndarray, raw: Line, slant: float) -> np.ndarray:
    """Gerçek etiket haritasını, satırın eğim düzeltilmiş çerçevesine taşır.

    Aynı `shear_array` çağrısı kullanıldığı için hizalama tanım gereği
    piksel piksel doğrudur; koordinat dönüşümünü elle türetmeye çalışmak
    işaret hatasına davetiye çıkarırdı.
    """
    x0, y0 = raw.origin
    height, width = raw.ink.shape

    window = np.full((height, width), -1, dtype=np.int32)
    src = labels[y0 : y0 + height, x0 : x0 + width]
    window[: src.shape[0], : src.shape[1]] = src

    if abs(slant) < 0.05:
        return window

    extra = deslant_padding(height, slant)
    padded = np.full((height, width + 2 * extra), -1, dtype=np.int32)
    padded[:, extra : extra + width] = window
    return shear_array(padded, slant, raw.baseline, fill=-1)


def evaluate(page: SynthPage, spec: SheetSpec, cfg: Config) -> SegmentationReport:
    """Sentetik bir sayfa üzerinde segmentasyonu ölçer."""
    assert page.labels is not None, "ölçüm için etiket haritası gerekli"

    raw_lines, lines, result, slant, _ = segment_page(page.photo, spec, cfg)
    boxes = result.boxes
    report = SegmentationReport(
        total_predicted=len(boxes), total_truth=len(page.truth), slant=slant
    )
    report.rejected_lines = dict(result.rejected)

    truth_index = {(c.band, c.index): i for i, c in enumerate(page.truth)}
    written_per_band: dict[int, int] = {}
    for c in page.truth:
        written_per_band[c.band] = written_per_band.get(c.band, 0) + 1

    boxes_by_line: dict[int, list[CharBox]] = {}
    for box in boxes:
        boxes_by_line.setdefault(box.line_index, []).append(box)

    for raw, line in zip(raw_lines, lines):
        expected = sum(1 for ch in line.text if ch != " ")
        if written_per_band.get(line.index, 0) < expected:
            # Sentezleyici satırı sığdıramamış: sayfada yazılan metin, hizalanması
            # istenen metinden kısa. Bu bir segmentasyon hatası değil.
            report.truncated_bands += 1
            continue

        label_map = _deslant_labels(page.labels, raw, slant)

        # Bu satırdaki her gerçek karakterin toplam mürekkep alanı.
        totals: dict[int, int] = {}
        ids, counts = np.unique(label_map[label_map >= 0], return_counts=True)
        for identifier, count in zip(ids.tolist(), counts.tolist()):
            totals[identifier] = count

        for box in boxes_by_line.get(line.index, []):
            key = (line.index, box.char_index)
            if key not in truth_index:
                continue
            target = truth_index[key]

            window = label_map[
                box.y0 : box.y0 + box.height, box.x0 : box.x0 + box.width
            ]
            if window.shape != box.mask.shape:
                continue

            own = int(((window == target) & box.mask).sum())
            labeled = int(((window >= 0) & box.mask).sum())
            total = totals.get(target, 0)
            if total == 0:
                continue

            purity = own / labeled if labeled else 0.0
            coverage = own / total
            report.purity.append(purity)
            report.coverage.append(coverage)
            report.per_char.setdefault(box.char, []).append(min(purity, coverage))
            report.evaluated += 1

    return report
