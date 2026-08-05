"""Üretilen fontun örnek sayfası (specimen) çizimi.

Fontun iyi olup olmadığına bakmanın tek gerçek yolu onunla yazılmış bir metne
bakmaktır. Bu modül hem kullanıcıya gösterilecek önizlemeyi hem de geliştirme
sırasında kullanılan görsel kontrolü üretir.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_SAMPLE = [
    "Pijamalı hasta yağız şoföre çabucak güvendi.",
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ",
    "abcçdefgğhıijklmnoöprsştuüvyz",
    "0123456789  .,;:!?'\"()-",
]

PARAGRAPH = (
    "Bu yazı, kağıda kendi elimle yazdığım harflerden üretildi. Her harfin "
    "birden çok biçimi var ve yan yana geldiklerinde sırayla kullanılıyorlar; "
    "bu yüzden aynı harf iki kez üst üste aynı görünmüyor. Noktalama, rakam ve "
    "Türkçe karakterler de aynı yazıdan geliyor: 1453, %20, (parantez içi)."
)


def _load(path: str | Path, size: int) -> ImageFont.FreeTypeFont:
    """Fontu, OpenType özelliklerini uygulayacak yerleşim motoruyla yükler.

    Raqm yoksa `calt` uygulanmaz ve bütün varyantlar devre dışı kalır — yazı
    yine çıkar ama her harf hep aynı biçimde görünür.
    """
    try:
        return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return ImageFont.truetype(str(path), size)


def render_text(
    font_path: str | Path,
    text: str,
    size: int = 48,
    width: int = 1100,
    padding: int = 30,
    features: list[str] | None = None,
) -> Image.Image:
    """Tek bir metni verilen fontla çizer (satır kaydırmalı)."""
    font = _load(font_path, size)
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) > width - 2 * padding and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    spacing = int(size * 1.55)
    image = Image.new("L", (width, padding * 2 + spacing * len(lines)), 255)
    canvas = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        canvas.text(
            (padding, padding + i * spacing), line, font=font, fill=20, features=features
        )
    return image


def render_specimen(
    font_path: str | Path,
    title: str = "handwrite",
    width: int = 1200,
) -> Image.Image:
    """Boyut örnekleri, alfabe ve paragraf içeren tam bir örnek sayfası üretir."""
    padding = 40
    blocks: list[Image.Image] = []

    header = Image.new("L", (width, 70), 255)
    ImageDraw.Draw(header).text(
        (padding, 20), f"{title} — üretilen font örneği", fill=110,
        font=ImageFont.load_default(size=22),
    )
    blocks.append(header)

    for line, size in zip(DEFAULT_SAMPLE, (54, 40, 40, 40)):
        blocks.append(render_text(font_path, line, size=size, width=width, padding=padding))

    for size in (18, 24, 32):
        blocks.append(render_text(font_path, PARAGRAPH, size=size, width=width, padding=padding))

    total = sum(b.height for b in blocks)
    sheet = Image.new("L", (width, total + 20), 255)
    y = 10
    for block in blocks:
        sheet.paste(block, (0, y))
        y += block.height
    return sheet


def render_variant_check(font_path: str | Path, size: int = 56, width: int = 1200) -> Image.Image:
    """Varyant döngüsünün çalıştığını gösteren karşılaştırma.

    Aynı metin iki kez çizilir: üstte `calt` kapalı (her harf tek biçim), altta
    açık. İki satır birbirinin aynıysa varyantlar devreye girmiyor demektir.
    """
    text = "aaaa eeee nnnn mmmm  anne  memnun  sessiz"
    # Dikkat: boş liste "özellik yok" demek değil, "varsayılanlar" demektir ve
    # `calt` varsayılan olarak açıktır. Kapatmak için özelliğin başına eksi
    # konur; aksi halde iki satır da aynı çıkar ve test hiçbir şey ölçmez.
    off = render_text(font_path, text, size=size, width=width, features=["-calt"])
    on = render_text(font_path, text, size=size, width=width, features=["+calt"])

    label = ImageFont.load_default(size=18)
    image = Image.new("L", (width, off.height + on.height + 70), 255)
    draw = ImageDraw.Draw(image)
    draw.text((30, 8), "calt kapalı — tek biçim", fill=120, font=label)
    image.paste(off, (0, 30))
    draw.text((30, off.height + 42), "calt açık — varyantlar dönüyor", fill=120, font=label)
    image.paste(on, (0, off.height + 64))
    return image
