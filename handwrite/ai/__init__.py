"""Yapay zekâ destekli adımlar.

Boru hattının geri kalanı geometriktir ve deterministik kalır. Buradaki modüller
yalnızca *anlamsal* işleri üstlenir — dil modellerinin gerçekten iyi olduğu
işler:

* **okuma** (`transcribe`) — sayfada ne yazdığını çıkarmak. Bu, sistemin
  önceden bilinen bir metin dayatmasını ortadan kaldırır: kullanıcı elindeki
  herhangi bir yazıyı yükleyebilir.
* **ayıklama** (`review`) — çıkarılan gliflerden hangilerinin bozuk olduğuna
  bakmak.

Bilinçli olarak yapılmayan şey, modele piksel koordinatı sordurmak. Görsel
modeller sınır kutusu tahmininde güvenilmez; font ise em biriminde çalışır ve
"yaklaşık" bir kutu glifin ayağını keser. Bu yüzden iş bölümü nettir:

    model *ne* yazdığını söyler, dinamik programlama *nerede* olduğunu bulur.
"""

from .provider import (
    AIError,
    LineTranscription,
    Transcriber,
    load_api_key,
)

__all__ = ["AIError", "LineTranscription", "Transcriber", "load_api_key"]
