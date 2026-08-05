"""Bitmap gliflerin kontur vektörlerine dönüştürülmesi.

Fontlar piksel değil eğri saklar. Bu modül, segmentasyondan çıkan mürekkep
maskesini kübik Bézier konturlarına çevirir.

İki incelik var:

* **Merdiven etkisi.** Tarama çözünürlüğünde bir harfin kenarı basamaklıdır.
  Doğrudan iz sürmek o basamakları eğriye kodlar ve font büyütüldüğünde tırtıklı
  görünür. Maske önce büyütülüp yumuşatılır, sonra iz sürülür; maliyeti sadece
  işlem süresidir, çıktı dosyası büyümez.

* **potrace'in mürekkep yönü.** potracer, *koyu* pikselleri (blacklevel'ın
  altındakileri) mürekkep sayar. Boolean maskemizde True mürekkeptir, yani
  maskeyi ters çevirmeden vermek sayfanın tamamını glif sanmasına yol açar.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import potrace

from .config import FontConfig

Point = tuple[float, float]


@dataclass
class Contour:
    """Kapalı bir kontur: bir başlangıç noktası ve ardışık parçalar.

    Parçalar ya `("line", uç)` ya da `("curve", kontrol1, kontrol2, uç)`
    biçimindedir.
    """

    start: Point
    segments: list[tuple]

    def points(self) -> list[Point]:
        """Konturun tüm düğüm noktaları (kaba sınır kutusu hesabı için)."""
        result = [self.start]
        for segment in self.segments:
            result.extend(segment[1:])
        return result


def _smooth_mask(mask: np.ndarray, upscale: int) -> np.ndarray:
    """Maskeyi büyütüp yumuşatarak basamakları giderir."""
    if upscale <= 1:
        return mask
    big = cv2.resize(
        mask.astype(np.float32),
        (mask.shape[1] * upscale, mask.shape[0] * upscale),
        interpolation=cv2.INTER_LINEAR,
    )
    big = cv2.GaussianBlur(big, (0, 0), upscale * 0.42)
    return big > 0.5


def trace_mask(mask: np.ndarray, cfg: FontConfig) -> list[Contour]:
    """Bir mürekkep maskesini kontur listesine çevirir.

    Dönen koordinatlar maskenin piksel uzayındadır: x sütun, y satır (aşağı
    pozitif). Font birimlerine çevirme işi çağırana aittir.
    """
    if not mask.any():
        return []

    upscale = max(1, cfg.trace_upscale)
    big = _smooth_mask(mask, upscale)

    # Kenara değen mürekkep konturun kapanmasını engeller; bir piksel çerçeve
    # bırakıp sonra ofseti geri alıyoruz.
    padded = np.zeros((big.shape[0] + 2, big.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = big

    # potracer koyu pikselleri mürekkep sayar; maskeyi ters veriyoruz.
    path = potrace.Bitmap(~padded).trace(
        turdsize=cfg.trace_turdsize * upscale,
        alphamax=cfg.trace_alphamax,
        opticurve=True,
        opttolerance=0.2,
    )

    scale = 1.0 / upscale
    offset = 1.0

    contours: list[Contour] = []
    for curve in path:
        start = ((curve.start_point.x - offset) * scale, (curve.start_point.y - offset) * scale)
        segments: list[tuple] = []
        for segment in curve.segments:
            end = ((segment.end_point.x - offset) * scale, (segment.end_point.y - offset) * scale)
            if segment.is_corner:
                corner = ((segment.c.x - offset) * scale, (segment.c.y - offset) * scale)
                segments.append(("line", corner))
                segments.append(("line", end))
            else:
                c1 = ((segment.c1.x - offset) * scale, (segment.c1.y - offset) * scale)
                c2 = ((segment.c2.x - offset) * scale, (segment.c2.y - offset) * scale)
                segments.append(("curve", c1, c2, end))
        if segments:
            contours.append(Contour(start=start, segments=segments))
    return contours


# --------------------------------------------------------------------------
# Doğrulama
# --------------------------------------------------------------------------


def rasterize(contours: list[Contour], shape: tuple[int, int], samples: int = 12) -> np.ndarray:
    """Konturları tekrar bitmap'e çevirir.

    Yalnızca doğrulama için: iz sürmenin şekli koruduğunu ölçmenin tek dürüst
    yolu, çıkan eğrileri geri çizip orijinal maskeyle karşılaştırmaktır.
    Konturlar iç içe olabildiği için (ör. "a"nın gözü) çift-tek dolgu kuralı
    veren `cv2.fillPoly` kullanılır.
    """
    polygons = []
    for contour in contours:
        points = [contour.start]
        current = contour.start
        for segment in contour.segments:
            if segment[0] == "line":
                points.append(segment[1])
                current = segment[1]
            else:
                _, c1, c2, end = segment
                for i in range(1, samples + 1):
                    t = i / samples
                    u = 1.0 - t
                    x = (
                        u**3 * current[0]
                        + 3 * u**2 * t * c1[0]
                        + 3 * u * t**2 * c2[0]
                        + t**3 * end[0]
                    )
                    y = (
                        u**3 * current[1]
                        + 3 * u**2 * t * c1[1]
                        + 3 * u * t**2 * c2[1]
                        + t**3 * end[1]
                    )
                    points.append((x, y))
                current = end
        polygons.append(np.array(points, dtype=np.float32).round().astype(np.int32))

    canvas = np.zeros(shape, np.uint8)
    if polygons:
        cv2.fillPoly(canvas, polygons, 1)
    return canvas > 0


def trace_fidelity(mask: np.ndarray, cfg: FontConfig) -> float:
    """İz sürmenin şekli ne kadar koruduğunu IoU olarak döndürür."""
    contours = trace_mask(mask, cfg)
    if not contours:
        return 0.0
    redrawn = rasterize(contours, mask.shape)
    union = (mask | redrawn).sum()
    return float((mask & redrawn).sum() / union) if union else 0.0
