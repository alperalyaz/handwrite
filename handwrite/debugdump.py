"""Ara adımların diske dökülmesi — tahmin yerine bakmak için.

Bir font kötü çıktığında sebebi çıktıya bakarak anlamak neredeyse imkânsızdır:
mürekkep mi yanlış ayrıldı, satırlar mı yanlış bulundu, model mi yanlış okudu,
kesim mi kaydı? Hepsi aynı sonucu verir — bozuk glifler.

Bu modül her aşamayı görünür kılar. Kullanıcının tek yapması gereken çıkan
klasördeki birkaç görüntüyü paylaşmak; hangi adımın bozulduğu bir bakışta
anlaşılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .lines import Line
from .preprocess import Page, detect_paper


@dataclass
class DebugDump:
    """Ara adımları verilen klasöre yazar."""

    folder: Path
    written: list[str]

    @classmethod
    def create(cls, folder: str | Path) -> "DebugDump":
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        return cls(folder=path, written=[])

    # -- yardımcılar -------------------------------------------------------

    def _save(self, name: str, image: np.ndarray) -> None:
        cv2.imwrite(str(self.folder / name), image)
        self.written.append(name)

    @staticmethod
    def _as_image(mask: np.ndarray) -> np.ndarray:
        """Boolean maskeyi bakılabilir bir görüntüye çevirir (mürekkep siyah)."""
        if mask.dtype == bool:
            return (~mask * 255).astype(np.uint8)
        return mask

    # -- aşamalar ----------------------------------------------------------

    def raw(self, image: np.ndarray, index: int = 0) -> None:
        self._save(f"{index}-1_ham.png", image)

    def paper(self, image: np.ndarray, index: int = 0) -> None:
        """Kağıt algılamanın ne kırptığını gösterir."""
        cropped = detect_paper(image)
        if cropped is None:
            note = image.copy()
            self._save(f"{index}-2_kagit-BULUNAMADI.png", note)
        else:
            self._save(f"{index}-2_kagit.png", cropped)

    def ink(self, page: Page, index: int = 0) -> None:
        self._save(f"{index}-3_murekkep.png", self._as_image(page.ink))

    def lines(self, page: Page, lines: list[Line], index: int = 0) -> None:
        """Bulunan satırları kutularla ve taban çizgileriyle işaretler.

        Bakılacak şey: her yazı satırına bir kutu düşmüş mü? İki satır tek
        kutuya girdiyse o satıra yazılandan az karakter atanır ve her glif
        birkaç harfin birleşimi olur. Bir satır ikiye bölündüyse tersi olur.
        """
        canvas = cv2.cvtColor(self._as_image(page.ink), cv2.COLOR_GRAY2BGR)
        for line in lines:
            x0, y0 = line.origin
            h, w = line.ink.shape
            cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (0, 150, 0), 2)
            baseline = int(y0 + line.baseline)
            cv2.line(canvas, (x0, baseline), (x0 + w, baseline), (255, 120, 120), 1)
            cv2.putText(
                canvas, str(line.index), (max(0, x0 - 34), baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2,
            )
        self._save(f"{index}-4_satirlar.png", canvas)

    def strips(self, lines: list[Line], index: int = 0, limit: int = 8) -> None:
        """Modele gönderilen şeritleri, gönderildikleri hâliyle yazar."""
        for line in lines[:limit]:
            self._save(f"{index}-5_serit-{line.index:02d}.png", self._as_image(line.ink))

    def transcription(self, lines: list[Line], readings, index: int = 0) -> None:
        """Okunan metni, satır genişlikleriyle birlikte kaydeder.

        Karakter sayısı ile mürekkep genişliğinin oranı en açık ipucudur:
        bütün satırlarda benzer olmalı. Bir satırda belirgin şekilde düşükse
        model o satırı eksik okumuş, glifler de birkaç harfin birleşimi olmuş
        demektir.
        """
        rows = ["# satır | karakter | genişlik(px) | px/karakter | güven | metin", ""]
        for line, reading in zip(lines, readings):
            text = getattr(reading, "text", "")
            confidence = getattr(reading, "confidence", 0.0)
            count = max(len(text), 1)
            width = line.ink.shape[1]
            rows.append(
                f"{line.index:3d} | {len(text):3d} | {width:5d} | "
                f"{width / count:6.1f} | {confidence:.2f} | {text}"
            )
        (self.folder / f"{index}-6_okunan.txt").write_text("\n".join(rows), encoding="utf-8")
        self.written.append(f"{index}-6_okunan.txt")

    def glyphs(self, boxes, index: int = 0, per_row: int = 16, cell: int = 90) -> None:
        """Çıkarılan glifleri tek bir sayfada gösterir.

        Bakılacak şey: her hücrede tek bir harf mi var? Birden çok harf varsa
        kesim yanlış yerlerde, hiçbir şey yoksa kesim boşluğa denk gelmiş.
        """
        if not boxes:
            return
        rows: list[np.ndarray] = []
        row: list[np.ndarray] = []
        for box in boxes[: per_row * 20]:
            tile = np.full((cell, cell), 255, np.uint8)
            mask = self._as_image(box.mask)
            h, w = mask.shape
            scale = min((cell - 16) / max(h, 1), (cell - 16) / max(w, 1), 1.0)
            if scale < 1.0:
                mask = cv2.resize(mask, (max(1, int(w * scale)), max(1, int(h * scale))))
                h, w = mask.shape
            tile[2 : 2 + h, 2 : 2 + w] = mask
            cv2.putText(tile, box.char, (3, cell - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 100, 1)
            row.append(tile)
            if len(row) == per_row:
                rows.append(np.hstack(row))
                row = []
        if row:
            while len(row) < per_row:
                row.append(np.full((cell, cell), 255, np.uint8))
            rows.append(np.hstack(row))
        self._save(f"{index}-7_glifler.png", np.vstack(rows))

    def summary(self) -> str:
        listing = "\n".join(f"  {name}" for name in self.written)
        return f"Teşhis dosyaları → {self.folder}\n{listing}"
