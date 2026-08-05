"""Sürükle-bırak web arayüzü.

Tek sayfalık bir akış: çalışma sayfalarını indir → doldur → fotoğrafları at →
fontu ve önizlemesini al. Oturum durumu bellekte tutulur; kalıcı depolama yok,
çünkü el yazısı kişisel veridir ve saklamamak en iyi saklama biçimidir.
"""

from __future__ import annotations

import io
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import Config
from .pipeline import build_from_images
from .specimen import render_specimen
from .template import SheetSet, build_sheets

app = FastAPI(title="handwrite")


@dataclass
class Session:
    """Bir kullanıcının çalışma sayfası takımı ve ürettiği font."""

    sheets: SheetSet
    font_bytes: bytes | None = None
    preview_png: bytes | None = None


SESSIONS: dict[str, Session] = {}


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/sheets")
def create_sheets() -> JSONResponse:
    """Yeni bir çalışma sayfası takımı üretir ve oturum kimliği döndürür."""
    images, sheets = build_sheets()
    token = uuid.uuid4().hex
    SESSIONS[token] = Session(sheets=sheets)
    return JSONResponse({"session": token, "pages": len(images)})


@app.get("/api/sheets/{token}.zip")
def download_sheets(token: str) -> Response:
    """Çalışma sayfalarını ve geometri dosyasını bir zip içinde verir."""
    session = SESSIONS.get(token)
    if session is None:
        raise HTTPException(404, "oturum bulunamadı")

    images, _ = build_sheets([b.text for s in session.sheets.sheets for b in s.bands])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images):
            archive.writestr(f"sayfa{index + 1}.png", _png(image))
        archive.writestr("sheets.json", session.sheets.to_json())
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="handwrite-sayfalar.zip"'},
    )


@app.post("/api/build")
async def build(
    session: str = Form(...),
    family: str = Form("Handwrite"),
    photos: list[UploadFile] = File(...),
) -> JSONResponse:
    """Yüklenen fotoğraflardan fontu üretir."""
    entry = SESSIONS.get(session)
    if entry is None:
        raise HTTPException(404, "oturum bulunamadı — sayfaları yeniden üretin")

    images: list[tuple[str, np.ndarray]] = []
    for upload in photos:
        raw = np.frombuffer(await upload.read(), np.uint8)
        decoded = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise HTTPException(400, f"{upload.filename} okunamadı")
        images.append((upload.filename or "foto", decoded))

    cfg = Config()
    cfg.font.family_name = family or "Handwrite"

    try:
        result = build_from_images(images, entry.sheets, cfg)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    buffer = io.BytesIO()
    result.font.save(buffer)
    entry.font_bytes = buffer.getvalue()

    # Önizleme, fontu gerçekten yükleyerek çizilir — üretilen dosyanın bir yazı
    # tipi motoru tarafından açılabildiğini de dolaylı olarak doğrular.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "font.ttf"
        path.write_bytes(entry.font_bytes)
        entry.preview_png = _png(render_specimen(path, title=cfg.font.family_name))

    diagnostics = result.diagnostics
    return JSONResponse(
        {
            "ok": True,
            "summary": diagnostics.summary_lines(),
            "characters": result.build.characters,
            "glyphs": result.build.glyph_count,
            "missing": diagnostics.missing_characters,
            "weak": [f"{c} ({k}/{s})" for c, s, k in diagnostics.weak_characters],
            "rejected": [
                f"sayfa {p + 1}, satır {b + 1}: {reason}"
                for (p, b), reason in sorted(diagnostics.rejected_lines.items())
            ],
            "page_errors": [
                f"{p.source}: {p.error}" for p in diagnostics.pages if p.error
            ],
        }
    )


@app.get("/api/font/{token}.ttf")
def download_font(token: str) -> Response:
    session = SESSIONS.get(token)
    if session is None or session.font_bytes is None:
        raise HTTPException(404, "font henüz üretilmedi")
    return Response(
        session.font_bytes,
        media_type="font/ttf",
        headers={"Content-Disposition": 'attachment; filename="Handwrite-Regular.ttf"'},
    )


@app.get("/api/preview/{token}.png")
def preview(token: str) -> Response:
    session = SESSIONS.get(token)
    if session is None or session.preview_png is None:
        raise HTTPException(404, "önizleme yok")
    return Response(session.preview_png, media_type="image/png")


INDEX_HTML = """
<!doctype html>
<html lang="tr">
<meta charset="utf-8">
<title>handwrite — el yazınızdan font</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; --fg:#1a1a1a; --bg:#faf9f7; --muted:#6b6b6b;
          --line:#dcd8d2; --accent:#2f6f4f; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e6e3; --bg:#1c1b19; --muted:#a09c96; --line:#3a3835; --accent:#7fc9a3; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
         color:var(--fg); background:var(--bg); }
  main { max-width: 820px; margin: 0 auto; padding: 48px 24px 96px; }
  h1 { font-size: 30px; margin: 0 0 6px; letter-spacing:-0.01em; }
  .sub { color: var(--muted); margin: 0 0 40px; }
  .step { border:1px solid var(--line); border-radius:12px; padding:22px 24px; margin-bottom:20px; }
  .step h2 { font-size:16px; margin:0 0 10px; display:flex; align-items:center; gap:10px; }
  .num { display:inline-grid; place-items:center; width:24px; height:24px; border-radius:50%;
         background:var(--accent); color:var(--bg); font-size:13px; font-weight:600; }
  button { font:inherit; padding:9px 16px; border-radius:8px; border:1px solid var(--accent);
           background:var(--accent); color:var(--bg); cursor:pointer; }
  button.ghost { background:transparent; color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  input[type=text] { font:inherit; padding:8px 10px; border-radius:8px;
                     border:1px solid var(--line); background:transparent; color:var(--fg); }
  #drop { border:2px dashed var(--line); border-radius:10px; padding:32px; text-align:center;
          color:var(--muted); cursor:pointer; }
  #drop.hot { border-color:var(--accent); color:var(--fg); }
  ul { margin:8px 0 0; padding-left:20px; color:var(--muted); }
  .warn { color:#b45309; }
  @media (prefers-color-scheme: dark) { .warn { color:#f0b357; } }
  pre { white-space:pre-wrap; font:13px/1.6 ui-monospace,monospace; color:var(--muted); margin:10px 0 0; }
  img { max-width:100%; border:1px solid var(--line); border-radius:8px; margin-top:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
</style>
<main>
  <h1>handwrite</h1>
  <p class="sub">El yazınızı gerçek bir yazı tipine çevirir. Kutu doldurmak yok —
     normal bir metin gibi yazın.</p>

  <div class="step">
    <h2><span class="num">1</span> Çalışma sayfalarını al</h2>
    <p class="sub" style="margin:0 0 12px">A4, %100 ölçekte yazdırın. Ölçeklendirme
       yapılırsa köşe işaretleri okunmaz.</p>
    <div class="row">
      <button id="mk">Sayfaları üret</button>
      <a id="dl" style="display:none"><button class="ghost">zip indir</button></a>
    </div>
  </div>

  <div class="step">
    <h2><span class="num">2</span> Doldur ve fotoğrafla</h2>
    <p class="sub" style="margin:0">Basılı örnek metni kendi elinizle, taban çizgisini
       takip ederek alttaki boşluğa yazın. Sonra iyi ışıkta, <b>dört köşe de
       kadrajda</b> olacak şekilde fotoğraflayın.</p>
  </div>

  <div class="step">
    <h2><span class="num">3</span> Fotoğrafları yükle</h2>
    <div class="row" style="margin-bottom:12px">
      <label>Font adı <input type="text" id="family" value="Handwrite"></label>
    </div>
    <div id="drop">Fotoğrafları buraya sürükleyin veya tıklayın
      <input type="file" id="file" multiple accept="image/*" hidden>
    </div>
    <div class="row" style="margin-top:12px">
      <button id="go" disabled>Fontu üret</button>
      <span id="count" class="sub" style="margin:0"></span>
    </div>
    <pre id="log"></pre>
  </div>

  <div class="step" id="out" style="display:none">
    <h2><span class="num">4</span> Sonuç</h2>
    <div class="row">
      <a id="ttf"><button>.ttf indir</button></a>
    </div>
    <img id="prev" alt="font önizlemesi">
  </div>
</main>
<script>
let session = null, files = [];
const $ = id => document.getElementById(id);

$('mk').onclick = async () => {
  $('mk').disabled = true; $('mk').textContent = 'üretiliyor…';
  const r = await fetch('/api/sheets', {method:'POST'});
  const d = await r.json();
  session = d.session;
  $('dl').href = `/api/sheets/${session}.zip`;
  $('dl').style.display = 'inline';
  $('mk').textContent = `${d.pages} sayfa hazır`;
  sync();
};

$('drop').onclick = () => $('file').click();
$('file').onchange = e => { files = [...e.target.files]; sync(); };
['dragenter','dragover'].forEach(ev => $('drop').addEventListener(ev, e => {
  e.preventDefault(); $('drop').classList.add('hot');
}));
['dragleave','drop'].forEach(ev => $('drop').addEventListener(ev, e => {
  e.preventDefault(); $('drop').classList.remove('hot');
}));
$('drop').addEventListener('drop', e => {
  files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
  sync();
});

function sync() {
  $('count').textContent = files.length ? `${files.length} fotoğraf seçildi` : '';
  $('go').disabled = !(session && files.length);
}

$('go').onclick = async () => {
  $('go').disabled = true; $('log').textContent = 'işleniyor…';
  const form = new FormData();
  form.append('session', session);
  form.append('family', $('family').value || 'Handwrite');
  files.forEach(f => form.append('photos', f));
  const r = await fetch('/api/build', {method:'POST', body: form});
  if (!r.ok) {
    const err = await r.json().catch(() => ({detail:'bilinmeyen hata'}));
    $('log').innerHTML = `<span class="warn">Hata: ${err.detail}</span>`;
    $('go').disabled = false; return;
  }
  const d = await r.json();
  let text = d.summary.join('\\n');
  if (d.page_errors.length) text += '\\n\\n! ' + d.page_errors.join('\\n! ');
  if (d.rejected.length)    text += '\\n\\n! ' + d.rejected.join('\\n! ');
  if (d.missing.length)     text += '\\n\\nfontta olmayan: ' + d.missing.join(' ');
  if (d.weak.length)        text += '\\nzayıf: ' + d.weak.join(', ');
  $('log').textContent = text;
  $('ttf').href = `/api/font/${session}.ttf`;
  $('prev').src = `/api/preview/${session}.png?t=` + Date.now();
  $('out').style.display = 'block';
  $('go').disabled = false;
};
</script>
</html>
"""
