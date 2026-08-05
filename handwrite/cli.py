"""Komut satırı arayüzü.

Üç komut, kullanıcının üç adımına karşılık gelir:

    handwrite sheets  -o calisma/      # 1. çalışma sayfalarını üret ve yazdır
    ...                                # 2. sayfaları elle doldur, fotoğrafla
    handwrite build   calisma/sheets.json foto*.jpg -o Benim.ttf
    handwrite preview Benim.ttf        # 3. sonucu gör
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CHARSET, Config
from .template import DEFAULT_CORPUS, build_sheets, coverage


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

    photos = [Path(p) for p in args.photos]
    missing = [p for p in photos if not p.exists()]
    if missing:
        print("Bulunamayan dosya: " + ", ".join(str(p) for p in missing), file=sys.stderr)
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
            "    pip install 'handwrite[web]'",
            file=sys.stderr,
        )
        return 2
    from .web import app

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

    preview = sub.add_parser("preview", help="bir fontun örnek sayfasını çiz")
    preview.add_argument("font", help=".ttf dosyası")
    preview.add_argument("-o", "--output", help="çıktı görüntüsü")
    preview.add_argument("--text", help="örnek sayfa yerine bu metni çiz")
    preview.add_argument("--size", type=int, default=48, help="--text ile punto")
    preview.add_argument(
        "--variants", action="store_true", help="varyant döngüsü karşılaştırması çiz"
    )
    preview.set_defaults(func=_cmd_preview)

    serve = sub.add_parser("serve", help="web arayüzünü başlat")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
