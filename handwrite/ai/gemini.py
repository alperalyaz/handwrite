"""Google Gemini ile el yazısı satırlarının okunması.

Tasarımın iki kilit kararı:

**Sayfa değil, satır gönderilir.** Bütün sayfayı gönderip "satır satır yaz"
demek daha az çağrı eder ama hangi metnin hangi satıra ait olduğu belirsizleşir:
model iki satırı birleştirir, birini atlar, sırayı kaydırır. Bizim için bu
ölümcül, çünkü hizalama satır-metin eşleşmesinin doğru olduğunu varsayar.
Satırları biz geometrik olarak buluyoruz (bu iş güvenilir), modele yalnızca
"bu şeritte ne yazıyor?" diye soruyoruz. Eşleşme tanım gereği doğru olur.

**Harfi harfine okuma istenir.** Model, yazım hatası düzeltmeye ya da metni
normalleştirmeye çok meyillidir. Bir tek harf eklemesi ya da düşürmesi bütün
satırın hizalamasını kaydırır, yani o satırdaki her glif yanlış harfe atanır.
İstem bunu açıkça yasaklar, ayrıca çıktı uzunluğu mürekkep genişliğine karşı
denetlenir.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

from .provider import AIError, LineTranscription, Transcriber, load_api_key

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: Varsayılan model. Görsel okuma gücü ile maliyet arasında denge.
DEFAULT_MODEL = "gemini-2.5-flash"

PROMPT = """Sana bir el yazısı sayfasından kesilmiş, TEK SATIRLIK şeritler veriyorum.
Her şeridi harfi harfine oku.

Kurallar:
- Gördüğünü aynen yaz. Yazım hatası varsa DÜZELTME, olduğu gibi bırak.
- Hiçbir harf ekleme, hiçbir harf düşürme. Metnin karakter dizisi
  el yazısındakiyle birebir aynı olmalı.
- Türkçe harflere dikkat: ı ile i, ş ile s, ğ ile g, ç ile c, ö ile o, ü ile u
  farklı harflerdir. Noktalı/noktasız ayrımını gördüğün gibi yaz.
- Büyük/küçük harfleri ve noktalama işaretlerini yazıldığı gibi koru.
- Satırın başındaki ve sonundaki boşlukları yazma.
- Bir şeridi okuyamıyorsan text alanını boş bırak ve confidence'ı 0 yap.
- Kelime aralarına tek boşluk koy.

Her şerit için sırayla bir nesne döndür. index alanı şeridin sırasıdır (0'dan
başlar). confidence, o satırı ne kadar emin okuduğun (0 ile 1 arası)."""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "text": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["index", "text", "confidence"],
    },
}


def encode_line(image: np.ndarray, height: int = 64) -> str:
    """Satır görüntüsünü base64 PNG'ye çevirir.

    Şeritler modele siyah-beyaz gönderilir: kağıt dokusu, gölge ve renk okumaya
    katkı sağlamaz, yalnız veri boyutunu büyütür.

    Ölçekleme **yalnız küçültme yönünde** yapılır. Sabit bir yüksekliğe
    "normalize etmek" ilk bakışta düzenli görünür ama tipik bir satır maskesi
    zaten 50-60 piksel yüksekliğindedir; onu 96'ya çıkarmak hiçbir bilgi
    katmadan görüntü alanını üç katına çıkarır. Bu, istek gövdesini şişirip
    modelin işini ağırlaştırır ve gerçek ölçümde zaman aşımına yol açtı: 10
    satırlık bir grup 330 KB'a çıkıyordu.
    """
    array = image
    if array.dtype == bool:
        array = (~array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    picture = Image.fromarray(array)
    if 0 < height < picture.height:
        scale = height / picture.height
        picture = picture.resize(
            (max(1, int(picture.width * scale)), height), Image.LANCZOS
        )

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class GeminiTranscriber(Transcriber):
    """Gemini tabanlı satır okuyucu."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    #: Tek istekte gönderilecek satır sayısı. Büyütmek çağrı sayısını azaltır
    #: ama isteği ağırlaştırır, modelin sırayı kaçırma ihtimalini artırır ve
    #: ilerleme geri bildirimini seyrekleştirir.
    batch_size: int = 5
    #: Tek bir isteğin zaman aşımı. Yüksek tutmak "takıldı mı, çalışıyor mu"
    #: belirsizliğini uzatır; hızlı başarısız olup tekrar denemek yeğdir.
    timeout: float = 75.0
    max_retries: int = 3
    #: Bütün okuma işi için üst sınır. Tek tek zaman aşımları çarpılıp
    #: kullanıcıyı on beş dakika bekletebiliyordu; bu sınır buna izin vermez.
    total_deadline: float = 300.0
    #: Şeritlerin modele gönderileceği en büyük piksel yüksekliği (yalnız
    #: küçültme yönünde uygulanır).
    strip_height: int = 64
    #: İlerleme bildirimi için isteğe bağlı geri çağrı: (biten, toplam).
    on_progress: object = None

    def __post_init__(self) -> None:
        self.api_key = load_api_key(self.api_key)

    # -- genel arayüz ------------------------------------------------------

    def transcribe_lines(self, images: list[np.ndarray]) -> list[LineTranscription]:
        results: list[LineTranscription] = []
        self._started = time.monotonic()
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            for item in self._transcribe_batch(chunk):
                results.append(
                    LineTranscription(
                        index=start + item.index, text=item.text, confidence=item.confidence
                    )
                )
            if callable(self.on_progress):
                self.on_progress(min(start + self.batch_size, len(images)), len(images))
        return results

    def _remaining(self) -> float:
        """İş için kalan süre; bittiğinde daha fazla deneme yapılmaz."""
        started = getattr(self, "_started", None)
        if started is None or self.total_deadline <= 0:
            return float("inf")
        return self.total_deadline - (time.monotonic() - started)

    # -- iç işleyiş --------------------------------------------------------

    def _transcribe_batch(self, images: list[np.ndarray]) -> list[LineTranscription]:
        parts: list[dict] = [{"text": PROMPT}]
        for order, image in enumerate(images):
            parts.append({"text": f"--- şerit {order} ---"})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": encode_line(image, self.strip_height),
                    }
                }
            )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                # Okuma işi yaratıcılık istemez; sıcaklık sıfırda tutulur ki
                # aynı sayfa her koşuda aynı okunsun.
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
            },
        }

        data = self._post(payload)
        return self._parse(data, len(images))

    def _post(self, payload: dict) -> dict:
        url = ENDPOINT.format(model=self.model)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            if self._remaining() <= 0:
                raise AIError(
                    f"Okuma {self.total_deadline:.0f} saniyede tamamlanamadı. "
                    "Gemini yanıt vermiyor olabilir; biraz sonra tekrar deneyin "
                    "ya da daha az sayıda fotoğrafla başlayın."
                )
            try:
                response = requests.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
            except requests.Timeout as exc:
                last = AIError(
                    f"Gemini {self.timeout:.0f} saniyede yanıt vermedi "
                    f"(deneme {attempt + 1}/{self.max_retries})"
                )
                _ = exc
                continue
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2**attempt, max(0.0, self._remaining())))
                continue

            if response.status_code == 200:
                return response.json()

            # 429 ve 5xx geçicidir; diğerleri tekrar denemeye değmez.
            if response.status_code == 429 or response.status_code >= 500:
                last = AIError(f"HTTP {response.status_code}: {response.text[:200]}")
                time.sleep(min(2**attempt * 2, max(0.0, self._remaining())))
                continue

            raise AIError(f"Gemini isteği reddetti (HTTP {response.status_code}): {response.text[:300]}")

        raise AIError(f"Gemini'ye ulaşılamadı: {last}")

    @staticmethod
    def _parse(data: dict, expected: int) -> list[LineTranscription]:
        try:
            candidates = data["candidates"]
            text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            reason = data.get("promptFeedback", {}).get("blockReason")
            raise AIError(f"Gemini beklenmedik yanıt döndürdü (blockReason={reason})") from exc

        try:
            rows = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AIError(f"Gemini geçerli JSON döndürmedi: {text[:200]}") from exc

        # Model bir şeridi atlarsa ya da sırayı karıştırırsa, indeksi olmayan
        # satırlar boş kalır. Sessizce kaydırmak, o satırdaki bütün glifleri
        # yanlış harfe atamak demek olurdu.
        result = [LineTranscription(index=i, text="", confidence=0.0) for i in range(expected)]
        for row in rows if isinstance(rows, list) else []:
            try:
                index = int(row["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < expected:
                result[index] = LineTranscription(
                    index=index,
                    text=str(row.get("text", "")).strip(),
                    confidence=float(row.get("confidence", 1.0) or 0.0),
                )
        return result
