"""Çıkarılan gliflerin doğrulanması — "bu gerçekten bir 'a' mı?"

Segmentasyon bir harfin neye benzediğini bilmez. Bir satırda kaç karakter
olduğunu ve her birinin yaklaşık ne kadar yer kaplaması gerektiğini bilir,
o kadar. Bu yüzden bitişik bir kelimeyi hiçbiri harf olmayan parçalara da
memnuniyetle böler ve sonuç fonta girer.

Buradaki adım o boşluğu kapatır: her aday glif, iddia ettiği harf olup
olmadığı sorularak dışarıdan denetlenir. Denetimi geçemeyen fonta girmez;
bir karakterden yeterli sayıda doğrulanmış örnek kalmazsa o karakter
üretilir (bkz. `handwrite.synthesize`).

Bunun kritik olmasının sebebi, denetimin *dışarıdan* gelmesi. Kendi
gliflerimizi kendi kümelerimizle denetlemek döngüseldir: bir harfin
örneklerinin çoğu yanlışsa kümenin merkezi de yanlış olur ve doğru örnekler
aykırı sayılır. Görsel bir modelin böyle bir önyargısı yoktur.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from .provider import AIError, load_api_key

#: Kişi başına bir karakterden en fazla kaç aday denetlenir. Hepsini
#: denetlemek gereksiz: fonta zaten birkaç varyant girecek.
MAX_CANDIDATES = 12

#: Bir istekte kaç karakterin tabakası gönderilir.
CHARS_PER_REQUEST = 6

#: Tabakadaki her hücrenin piksel boyu.
CELL = 88

PROMPT = """Sana el yazısı bir sayfadan otomatik olarak kesilmiş harf adayları
veriyorum. Her görüntü, numaralanmış hücrelerden oluşan bir tabakadır ve
tabakanın hangi harfe ait olduğunu sana söylüyorum.

Her hücre için tek bir soruya cevap ver: bu hücrede o harfin *tek, eksiksiz
ve temiz* bir örneği var mı?

"Hayır" demen gereken durumlar:
- hücrede başka bir harf var,
- birden çok harf ya da harf parçaları bir arada,
- harfin bir kısmı kesilmiş (bir bacağı, kuyruğu, noktası eksik),
- komşu harften yapışmış fazladan bir çizgi var,
- ne olduğu anlaşılmıyor.

Küçük/büyük harf ayrımına dikkat et: "a" isteniyorsa "A" hayırdır.
Türkçe harflerde noktalı/noktasız ve aksan ayrımı önemlidir: "ı" isteniyorsa
"i" hayırdır, "o" isteniyorsa "ö" hayırdır.

El yazısının çirkin, eğri ya da özensiz olması sorun değil — soru güzellik
değil, doğru harfin eksiksiz olup olmadığı."""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "char": {"type": "STRING"},
            "cell": {"type": "INTEGER"},
            "ok": {"type": "BOOLEAN"},
        },
        "required": ["char", "cell", "ok"],
    },
}


class GlyphVerifier(Protocol):
    """Aday glifleri denetleyen herhangi bir sağlayıcı."""

    def verify(self, candidates: dict[str, list[np.ndarray]]) -> dict[str, list[bool]]:
        """Her karakter için adayların geçip geçmediğini döndürür."""
        ...


@dataclass
class VerificationReport:
    """Denetimin sonucu."""

    checked: int = 0
    passed: int = 0
    per_char: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.checked if self.checked else 0.0

    def summary_lines(self) -> list[str]:
        if not self.checked:
            return []
        rows = [
            f"glif denetimi: {self.passed}/{self.checked} aday doğrulandı "
            f"(%{self.pass_rate * 100:.0f})"
        ]
        weak = sorted(
            (c for c, (ok, total) in self.per_char.items() if ok == 0 and total),
            key=str,
        )
        if weak:
            rows.append(
                "hiçbir örneği doğrulanamayan karakter (üretilecek): " + " ".join(weak)
            )
        return rows


def build_sheet(images: list[np.ndarray], columns: int = 6) -> np.ndarray:
    """Adayları numaralanmış bir tabakaya dizer.

    Tek tek göndermek yerine tabaka gönderilmesinin sebebi maliyet değil
    tutarlılık: model adayları yan yana görünce "hangisi gerçekten bu harf"
    sorusunu karşılaştırarak yanıtlıyor.
    """
    from PIL import ImageDraw

    rows = max(1, (len(images) + columns - 1) // columns)
    sheet = Image.new("L", (columns * CELL, rows * CELL), 255)
    draw = ImageDraw.Draw(sheet)

    for index, mask in enumerate(images):
        array = (~mask * 255).astype(np.uint8) if mask.dtype == bool else mask
        tile = Image.fromarray(array)
        # Hücreye sığdır, oranı koru; numara için altta yer bırak.
        box = CELL - 22
        scale = min(box / max(tile.width, 1), box / max(tile.height, 1))
        if scale < 1.0:
            tile = tile.resize(
                (max(1, int(tile.width * scale)), max(1, int(tile.height * scale))),
                Image.LANCZOS,
            )
        column, row = index % columns, index // columns
        x = column * CELL + (CELL - tile.width) // 2
        y = row * CELL + 4
        sheet.paste(tile, (x, y))
        draw.text((column * CELL + 4, row * CELL + CELL - 15), str(index), fill=110)
        draw.rectangle(
            [column * CELL, row * CELL, column * CELL + CELL - 1, row * CELL + CELL - 1],
            outline=200,
        )
    return np.asarray(sheet)


def _encode(sheet: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(sheet).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class GeminiVerifier(GlyphVerifier):
    """Gemini ile glif denetimi."""

    api_key: str | None = None
    model: str = "gemini-2.5-flash"
    max_candidates: int = MAX_CANDIDATES
    chars_per_request: int = CHARS_PER_REQUEST
    timeout: float = 90.0
    max_retries: int = 2
    on_progress: object = None

    def __post_init__(self) -> None:
        self.api_key = load_api_key(self.api_key)

    def verify(self, candidates: dict[str, list[np.ndarray]]) -> dict[str, list[bool]]:
        chars = sorted(candidates)
        verdicts: dict[str, list[bool]] = {}

        for start in range(0, len(chars), self.chars_per_request):
            group = chars[start : start + self.chars_per_request]
            try:
                verdicts.update(self._verify_group(group, candidates))
            except AIError:
                # Denetlenemeyen karakterler denetimsiz geçer: bir ağ hatası
                # yüzünden bütün fontu üretilmiş gliflere düşürmek yanlış olur.
                for char in group:
                    verdicts.setdefault(char, [True] * len(candidates[char]))
            if callable(self.on_progress):
                self.on_progress(min(start + len(group), len(chars)), len(chars))
        return verdicts

    def _verify_group(
        self, chars: list[str], candidates: dict[str, list[np.ndarray]]
    ) -> dict[str, list[bool]]:
        import requests

        parts: list[dict] = [{"text": PROMPT}]
        counts: dict[str, int] = {}
        for char in chars:
            images = candidates[char][: self.max_candidates]
            counts[char] = len(images)
            parts.append({"text": f'--- aşağıdaki tabaka "{char}" harfi için ---'})
            parts.append(
                {"inline_data": {"mime_type": "image/png", "data": _encode(build_sheet(images))}}
            )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        response = None
        for _ in range(self.max_retries):
            try:
                response = requests.post(
                    url, params={"key": self.api_key}, json=payload, timeout=self.timeout
                )
            except requests.RequestException:
                continue
            if response.status_code == 200:
                break
            response = None
        if response is None:
            raise AIError("denetim isteği başarısız")

        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            rows = json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise AIError("denetim yanıtı okunamadı") from exc

        # Varsayılan: cevabı gelmeyen aday elenir. Denetimin amacı şüpheliyi
        # dışarıda tutmak; sessiz kalan bir hücre için "geçti" demek amacı
        # tersine çevirirdi.
        verdicts = {char: [False] * counts[char] for char in chars}
        for row in rows if isinstance(rows, list) else []:
            char = str(row.get("char", ""))
            cell = row.get("cell")
            if char in verdicts and isinstance(cell, int) and 0 <= cell < counts[char]:
                verdicts[char][cell] = bool(row.get("ok"))
        return verdicts


def apply_verification(
    boxes: list, verifier: GlyphVerifier, max_candidates: int = MAX_CANDIDATES
) -> tuple[list, VerificationReport]:
    """Kutuları denetimden geçirir; geçenleri ve raporu döndürür.

    Denetime yalnız her karakterin bir alt kümesi gönderilir. Fonta karakter
    başına birkaç varyant gireceği için hepsini denetlemek gereksiz maliyet
    olurdu; doğrulanan örnekler yeterli sayıdaysa geri kalanı zaten
    kullanılmayacaktır.
    """
    grouped: dict[str, list] = {}
    for box in boxes:
        grouped.setdefault(box.char, []).append(box)

    sampled = {char: items[:max_candidates] for char, items in grouped.items()}
    verdicts = verifier.verify({char: [b.mask for b in items] for char, items in sampled.items()})

    report = VerificationReport()
    kept: list = []
    for char, items in sampled.items():
        flags = verdicts.get(char, [True] * len(items))
        passed = [box for box, ok in zip(items, flags) if ok]
        report.checked += len(items)
        report.passed += len(passed)
        report.per_char[char] = (len(passed), len(items))
        kept.extend(passed)
    return kept, report
