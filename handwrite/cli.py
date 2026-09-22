"""Komut satırı arayüzü.

Son kullanıcı için tek komut yeter; gerisi tarayıcıda olur:

    handwrite serve                    # arayüzü açar

Geri kalan komutlar geliştirme ve toplu iş içindir:

    handwrite read  yazim.jpg -o Benim.ttf      # şablonsuz, model okur
    handwrite sheets -o calisma/                # basılı çalışma sayfası üret
    handwrite build  calisma/sheets.json *.jpg  # doldurulmuş sayfalardan üret
    handwrite preview Benim.ttf                 # örnek sayfa çiz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CHARSET, Config
from .template import DEFAULT_CORPUS, build_sheets, coverage


def _resolve_photos(patterns: list[str]) -> tuple[list[Path], list[str]]:
    """Dosya listesini çözer, gerekirse joker karakterleri kendisi genişletir.

    Unix kabuğu `foto*.jpg` gibi kalıpları komuta ulaşmadan genişletir, ama
    PowerShell yerleşik olmayan komutlar için bunu yapmaz ve kalıbı olduğu gibi
    aktarır. Genişletmeyi burada da yapmak, aynı komut satırının iki platformda
    da çalışmasını sağlar.
    """
    resolved: list[Path] = []
    missing: list[str] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.exists():
            resolved.append(candidate)
            continue
        if any(ch in pattern for ch in "*?["):
            base = candidate.parent if candidate.parent != Path("") else Path(".")
            matches = sorted(base.glob(candidate.name))
            if matches:
                resolved.extend(matches)
                continue
        missing.append(pattern)
    return resolved, missing


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


def _report_missing(missing: list[str], patterns: list[str]) -> int:
    """Bulunamayan dosyaları, aynı klasördeki gerçek adlarla birlikte bildirir.

    Kuru bir "bulunamadı" kullanıcıyı yalnız bırakır: dosya adını mı yanlış
    yazdı, yanlış klasörde mi, uzantı mı farklı — bilemez. Klasördeki görüntü
    dosyalarını listelemek bu üç sorunun da cevabını bir bakışta verir.
    """
    print("Bulunamayan dosya: " + ", ".join(missing), file=sys.stderr)

    folders = {Path(p).parent if Path(p).parent != Path("") else Path(".") for p in patterns}
    found: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            found.extend(
                sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            )

    if found:
        print("\nBu klasördeki görüntü dosyaları:", file=sys.stderr)
        for path in found[:20]:
            print(f"  {path}", file=sys.stderr)
        print("\nKomutu bu adlardan biriyle tekrar çalıştırın.", file=sys.stderr)
    else:
        print(
            "\nBu klasörde hiç görüntü dosyası yok. El yazısı fotoğrafını buraya "
            "kopyalayın ya da tam yolunu verin:\n"
            "  handwrite read \"C:\\Users\\ad\\Pictures\\yazim.jpg\" -o Benim.ttf",
            file=sys.stderr,
        )
    return 2


def _cmd_sheets(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    lines = DEFAULT_CORPUS
    if args.text:
        lines = [line.rstrip() for line in Path(args.text).read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line.strip()]

    images, sheets = build_sheets(lines, dpi=args.dpi)
    for index, image in enumerate(images):
        image.save(out / f"sayfa{index + 1}.png")
    sheets.save(out / "sheets.json")

    counts = coverage(lines, CHARSET)
    thin = sorted(ch for ch, n in counts.items() if n < 3)

    print(f"{len(images)} sayfa üretildi → {out}")
    print(f"  sheets.json  (font üretirken bu dosya gerekli, saklayın)")
    print()
    print("Sırada:")
    print("  1. Sayfaları ölçeklendirmeden, %100 boyutta yazdırın (A4).")
    print("  2. Basılı örnek metni kendi el yazınızla, taban çizgisini takip")
    print("     ederek alttaki boşluğa yazın.")
    print("  3. İyi ışıkta, sayfanın dört köşesi de kadraja girecek şekilde")
    print("     fotoğraflayın ya da tarayın.")
    print("  4. handwrite build ile fontu üretin.")
    if thin:
        print()
        print("  Not: şu karakterler metinde 3'ten az geçiyor, fontta zayıf")
        print("  kalabilirler: " + " ".join(thin))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from .pipeline import build_from_paths, write_font

    cfg = Config()
    if args.family:
        cfg.font.family_name = args.family
    if args.variants:
        cfg.glyph.variants_per_char = args.variants

    photos, missing = _resolve_photos(args.photos)
    if missing:
        return _report_missing(missing, args.photos)
    if not photos:
        print("Hiç fotoğraf verilmedi.", file=sys.stderr)
        return 2

    result = build_from_paths(photos, args.sheets, cfg)

    print("Teşhis:")
    for line in result.diagnostics.summary_lines():
        print("  " + line)
    for page in result.diagnostics.pages:
        if page.error:
            print(f"  ! {page.source}: {page.error}")
    for (page_index, band), reason in sorted(result.diagnostics.rejected_lines.items()):
        print(f"  ! sayfa {page_index + 1}, satır {band + 1} atlandı: {reason}")

    output = Path(args.output)
    write_font(result, output)
    size_kb = output.stat().st_size / 1024
    print()
    print(f"Font yazıldı: {output}  ({result.build.characters} karakter, "
          f"{result.build.glyph_count} glif, {size_kb:.0f} KB)")

    if args.preview:
        from .specimen import render_specimen

        preview = output.with_suffix(".onizleme.png")
        render_specimen(output, title=cfg.font.family_name).save(preview)
        print(f"Önizleme: {preview}")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    """Şablonsuz mod: herhangi bir el yazısı sayfasından font üret."""
    from .ai.gemini import GeminiTranscriber
    from .ai.provider import AIError
    from .pipeline import build_from_freeform, write_font
    from .preprocess import load_gray

    cfg = Config()
    if args.family:
        cfg.font.family_name = args.family
    if args.no_synth:
        cfg.synthesize_missing = False

    photos, missing = _resolve_photos(args.photos)
    if missing:
        return _report_missing(missing, args.photos)
    if not photos:
        print("Hiç fotoğraf verilmedi.", file=sys.stderr)
        return 2

    try:
        transcriber = GeminiTranscriber(api_key=args.api_key, model=args.model)
    except AIError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    dump = None
    if args.debug:
        from .debugdump import DebugDump

        dump = DebugDump.create(args.debug)

    verifier = None
    if not args.no_verify:
        from .ai.verify import GeminiVerifier

        verifier = GeminiVerifier(api_key=args.api_key, model=args.model)

    images = [(p.name, load_gray(p)) for p in photos]
    try:
        result = build_from_freeform(
            images, transcriber, cfg, debug=dump, verifier=verifier
        )
    except AIError as exc:
        print(f"Okuma başarısız: {exc}", file=sys.stderr)
        return 1

    print("Teşhis:")
    for line in result.diagnostics.summary_lines():
        print("  " + line)
    if args.show_text:
        print()
        print("Okunan metin:")
        for _, text, confidence in result.diagnostics.transcriptions:
            print(f"  [{confidence:.2f}] {text}")
    if result.synthesis:
        print()
        print("Eksik karakterler:")
        for line in result.synthesis.summary_lines():
            print("  " + line)

    output = Path(args.output)
    write_font(result, output)
    print()
    print(f"Font yazıldı: {output}  ({result.build.characters} karakter, "
          f"{output.stat().st_size / 1024:.0f} KB)")

    if args.preview:
        from .specimen import render_specimen

        target = output.with_suffix(".onizleme.png")
        render_specimen(output, title=cfg.font.family_name).save(target)
        print(f"Önizleme: {target}")
    if dump is not None:
        print()
        print(dump.summary())
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from .specimen import render_specimen, render_text, render_variant_check

    font = Path(args.font)
    if args.text:
        image = render_text(font, args.text, size=args.size)
        target = Path(args.output or font.with_suffix(".metin.png"))
    elif args.variants:
        image = render_variant_check(font)
        target = Path(args.output or font.with_suffix(".varyantlar.png"))
    else:
        image = render_specimen(font, title=font.stem)
        target = Path(args.output or font.with_suffix(".onizleme.png"))
    image.save(target)
    print(f"Yazıldı: {target}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Web arayüzü için ek paketler gerekli:\n"
            '    pip install "handwrite[web]"',
            file=sys.stderr,
        )
        return 2
    from .ai.provider import AIError, load_api_key
    from .web import app

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"

    try:
        load_api_key()
        key_note = "API anahtarı bulundu."
    except AIError:
        key_note = (
            "API anahtarı bulunamadı — arayüzde elle girebilirsiniz\n"
            "   (ya da GOOGLE_AI_API_KEY ortam değişkenini tanımlayın)."
        )

    print(f"handwrite açık:  {url}")
    print(f"   {key_note}")
    print("   Durdurmak için Ctrl+C.")

    if not args.no_open:
        # Sunucu başlamadan tarayıcı açılırsa boş sayfa gelir; kısa bir gecikme
        # ile açmak, tek komutla çalışan bir deneyim için yeterli.
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="handwrite",
        description="El yazısı taramalarından OpenType font üretir.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sheets = sub.add_parser("sheets", help="doldurulacak çalışma sayfalarını üret")
    sheets.add_argument("-o", "--output", default="calisma", help="çıktı klasörü")
    sheets.add_argument("--dpi", type=int, default=200, help="basım çözünürlüğü")
    sheets.add_argument("--text", help="kendi metniniz (satır satır bir dosya)")
    sheets.set_defaults(func=_cmd_sheets)

    build = sub.add_parser("build", help="doldurulmuş sayfa fotoğraflarından font üret")
    build.add_argument("sheets", help="sayfa üretiminde oluşan sheets.json")
    build.add_argument("photos", nargs="+", help="doldurulmuş sayfaların fotoğrafları")
    build.add_argument("-o", "--output", default="Handwrite-Regular.ttf", help="çıktı .ttf")
    build.add_argument("--family", help="font ailesi adı")
    build.add_argument("--variants", type=int, help="karakter başına varyant sayısı")
    build.add_argument("--preview", action="store_true", help="örnek sayfa da üret")
    build.set_defaults(func=_cmd_build)

    read = sub.add_parser(
        "read",
        help="şablonsuz: herhangi bir el yazısı sayfasını okuyup font üret",
    )
    read.add_argument("photos", nargs="+", help="el yazısı sayfalarının fotoğrafları")
    read.add_argument("-o", "--output", default="Handwrite-Regular.ttf", help="çıktı .ttf")
    read.add_argument("--family", help="font ailesi adı")
    read.add_argument("--model", default="gemini-2.5-flash", help="kullanılacak model")
    read.add_argument("--api-key", help="API anahtarı (yoksa ortamdan okunur)")
    read.add_argument("--show-text", action="store_true", help="okunan metni yazdır")
    read.add_argument(
        "--no-synth",
        action="store_true",
        help="eksik karakterleri üretme (font eksik ama tamamen gerçek kalır)",
    )
    read.add_argument("--preview", action="store_true", help="örnek sayfa da üret")
    read.add_argument(
        "--no-verify",
        action="store_true",
        help="glif denetimini atla (daha hızlı ama yanlış harfler fonta girebilir)",
    )
    read.add_argument(
        "--debug",
        metavar="KLASOR",
        help="ara adımları bu klasöre yaz (hangi adımın bozulduğunu görmek için)",
    )
    read.set_defaults(func=_cmd_read)

    preview = sub.add_parser("preview", help="bir fontun örnek sayfasını çiz")
    preview.add_argument("font", help=".ttf dosyası")
    preview.add_argument("-o", "--output", help="çıktı görüntüsü")
    preview.add_argument("--text", help="örnek sayfa yerine bu metni çiz")
    preview.add_argument("--size", type=int, default=48, help="--text ile punto")
    preview.add_argument(
        "--variants", action="store_true", help="varyant döngüsü karşılaştırması çiz"
    )
    preview.set_defaults(func=_cmd_preview)

    serve = sub.add_parser("serve", help="web arayüzünü başlat (önerilen kullanım)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-open", action="store_true", help="tarayıcıyı otomatik açma")
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
