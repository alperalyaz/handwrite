"""Sağlayıcıdan bağımsız arayüz ve anahtar yönetimi."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class AIError(RuntimeError):
    """Sağlayıcı çağrısı başarısız olduğunda atılır."""


@dataclass
class LineTranscription:
    """Tek bir satırın okunması."""

    index: int
    text: str
    #: Modelin okunabilirlik değerlendirmesi (0–1). Düşükse satır şüphelidir.
    confidence: float = 1.0

    @property
    def usable(self) -> bool:
        return bool(self.text.strip())


class Transcriber(Protocol):
    """El yazısı satırlarını metne çeviren herhangi bir sağlayıcı."""

    def transcribe_lines(self, images: list[np.ndarray]) -> list[LineTranscription]:
        """Satır görüntülerini sırayla metne çevirir.

        Dönen listenin uzunluğu girdiyle aynı olmalıdır; okunamayan satırlar
        boş metinle döner.
        """
        ...


#: Anahtarın aranacağı ortam değişkenleri.
ENV_KEYS = ("GOOGLE_AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Ortam değişkeni yoksa bakılacak dosya. Depo dışında tutulur ki yanlışlıkla
#: sürüm kontrolüne girmesin.
KEY_FILE = Path.home() / ".config" / "handwrite" / "env"


def load_api_key(explicit: str | None = None) -> str:
    """API anahtarını bulur: parametre → ortam değişkeni → yerel dosya.

    Anahtar hiçbir zaman depoya yazılmaz ve günlüklere basılmaz.
    """
    if explicit:
        return explicit
    for name in ENV_KEYS:
        value = os.environ.get(name)
        if value:
            return value.strip()

    if KEY_FILE.exists():
        for raw in KEY_FILE.read_text(encoding="utf-8").splitlines():
            if "=" not in raw or raw.lstrip().startswith("#"):
                continue
            name, _, value = raw.partition("=")
            if name.strip() in ENV_KEYS and value.strip():
                return value.strip()

    raise AIError(
        "API anahtarı bulunamadı. Şunlardan birini yapın:\n"
        f"  export {ENV_KEYS[0]}=...\n"
        f"  ya da {KEY_FILE} dosyasına '{ENV_KEYS[0]}=...' satırını ekleyin."
    )
