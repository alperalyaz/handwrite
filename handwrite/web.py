"""Web arayüzü — kullanıcının gördüğü tek yüz.

Komut satırı geliştirme içindir; son kullanıcı `handwrite serve` deyip
tarayıcıda çalışır. Arayüz üç şeyi kapsar:

* **Fotoğraf alma** — dosya seçme, sürükle-bırak ve doğrudan kamera. Telefonda
  `capture` özniteliği kamerayı doğrudan açar; masaüstünde `getUserMedia` ile
  webcam kullanılır. Kullanıcıdan bir klasöre dosya kopyalamasını istemek
  arayüz sayılmaz.
* **Yönlendirme** — nasıl bir fotoğrafın işe yaradığı *önceden* söylenir.
  Kalem tipi, satır sayısı, ışık ve kağıt hakkındaki ölçütler sonradan hata
  mesajı olarak değil, baştan kontrol listesi olarak verilir.
* **Yazısı olmayan için metin** — eline kalem almamış biri için kopyalanacak
  bir metin ve yazdırılabilir hâli.

Oturum durumu yalnız bellekte tutulur. El yazısı kişisel veridir; saklamamak
en iyi saklama biçimidir.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .ai.provider import AIError, load_api_key
from .config import Config
from .pipeline import build_from_freeform, build_from_images
from .specimen import render_specimen
from .template import FREEFORM_SAMPLE, SheetSet, build_sheets

app = FastAPI(title="handwrite")

#: Kabul edilen görüntü türleri. Arayüz de aynı listeyi gösterir; kullanıcının
#: neyin işe yaradığını denemeyle bulması gerekmemeli.
ACCEPTED = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


@dataclass
class Session:
    """Bir kullanıcının ürettiği font ve (şablon modundaysa) sayfa takımı."""

    sheets: SheetSet | None = None
    font_bytes: bytes | None = None
    preview_png: bytes | None = None
    family: str = "Handwrite"
    #: Ara adım görüntülerinin zip'i. Bir font kötü çıktığında hangi adımın
    #: bozulduğunu anlamanın tek yolu ara adımlara bakmak; bunu kullanıcıya
    #: komut satırı ödevi olarak vermek yerine tek tıkla indirilebilir yapıyoruz.
    debug_zip: bytes | None = None


SESSIONS: dict[str, Session] = {}


@dataclass
class Job:
    """Arka planda süren bir font üretimi.

    Üretim yarım dakikayı bulabiliyor; bunu tek bir uzun HTTP isteğiyle yapmak
    kullanıcıyı ekranda "bir şey oluyor mu?" diye bırakır ve tarayıcı/vekil
    zaman aşımlarına takılır. İş arka plana alınıp durumu sorgulanabilir hale
    getirildi: arayüz hangi aşamada olduğunu ve ne kadar süre geçtiğini
    gösterebiliyor.
    """

    stage: str = "başlıyor"
    started: float = field(default_factory=time.monotonic)
    done: bool = False
    error: str | None = None
    payload: dict | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


JOBS: dict[str, Job] = {}


def _zip_folder(folder: str) -> bytes:
    """Klasörü belleğe zipler."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return buffer.getvalue()


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode(upload: UploadFile, raw: bytes) -> np.ndarray:
    """Yüklenen dosyayı gri tonlamalı diziye çevirir, olmazsa anlaşılır hata verir."""
    array = np.frombuffer(raw, np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise HTTPException(
            400,
            f"'{upload.filename}' bir görüntü olarak açılamadı. "
            f"Desteklenen türler: {', '.join(sorted(set(ACCEPTED.values())))}. "
            "iPhone'dan HEIC geldiyse, paylaşırken 'En Uyumlu' biçimi seçin ya "
            "da ekran görüntüsü alın.",
        )
    return decoded


def _render_preview(session: Session) -> None:
    """Fontu gerçekten yükleyip örnek sayfa çizer.

    Önizlemenin yan faydası: üretilen dosyanın bir yazı tipi motoru tarafından
    açılabildiğini de doğrulamış oluruz.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "font.ttf"
        path.write_bytes(session.font_bytes or b"")
        session.preview_png = _png(render_specimen(path, title=session.family))


def _diagnostics_payload(result) -> dict:
    diagnostics = result.diagnostics
    payload = {
        "ok": True,
        # Uyarılar özetten ayrı taşınır: metin dökümünün içinde kaybolduklarında
        # kimse okumuyor, oysa çoğu ("fotoğraf çok küçük") kullanıcının tek
        # hamlede düzeltebileceği ve sonucu belirleyen şeyler.
        "warnings": diagnostics.warnings,
        "summary": diagnostics.summary_lines(include_warnings=False),
        "characters": result.build.characters,
        "glyphs": result.build.glyph_count,
        "missing": diagnostics.missing_characters,
        "synthetic": diagnostics.synthetic_characters,
        "rejected": [
            f"sayfa {p + 1}, satır {b + 1}: {reason}"
            for (p, b), reason in sorted(diagnostics.rejected_lines.items())
        ],
        "page_errors": [f"{p.source}: {p.error}" for p in diagnostics.pages if p.error],
        "transcriptions": [
            {"text": text, "confidence": confidence}
            for _, text, confidence in diagnostics.transcriptions
        ],
    }
    if result.synthesis:
        payload["synthesis"] = result.synthesis.summary_lines()
    return payload


# --------------------------------------------------------------------------
# Sayfalar
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status")
def status() -> JSONResponse:
    """Arayüzün anahtar durumunu baştan bilmesi için.

    Anahtar yoksa kullanıcı 20 saniyelik bir yüklemenin *sonunda* öğrenmemeli.
    """
    try:
        load_api_key()
        return JSONResponse({"ai": True})
    except AIError as exc:
        return JSONResponse({"ai": False, "reason": str(exc)})


@app.get("/api/sample")
def sample() -> JSONResponse:
    """Elinde yazı olmayanlar için kopyalanacak metin."""
    return JSONResponse({"lines": FREEFORM_SAMPLE})


# --------------------------------------------------------------------------
# Serbest mod (yapay zekâ okur)
# --------------------------------------------------------------------------


@app.post("/api/read")
async def read(
    family: str = Form("Handwrite"),
    api_key: str = Form(""),
    photos: list[UploadFile] = File(...),
) -> JSONResponse:
    """Font üretimini arka planda başlatır ve iş kimliği döndürür."""
    from .ai.gemini import GeminiTranscriber

    try:
        transcriber = GeminiTranscriber(api_key=api_key.strip() or None)
    except AIError as exc:
        raise HTTPException(400, str(exc)) from exc

    images: list[tuple[str, np.ndarray]] = []
    for upload in photos:
        images.append((upload.filename or "foto", _decode(upload, await upload.read())))

    cfg = Config()
    cfg.font.family_name = family.strip() or "Handwrite"

    job_id = uuid.uuid4().hex
    job = Job()
    JOBS[job_id] = job

    def run() -> None:
        from .ai.verify import GeminiVerifier
        from .debugdump import DebugDump

        folder = tempfile.mkdtemp(prefix="handwrite-teshis-")
        dump = DebugDump.create(folder)
        try:
            verifier = GeminiVerifier(api_key=api_key.strip() or None)
        except AIError:
            verifier = None

        try:
            result = build_from_freeform(
                images,
                transcriber,
                cfg,
                progress=lambda stage: setattr(job, "stage", stage),
                debug=dump,
                verifier=verifier,
            )
        except AIError as exc:
            job.error = f"Okuma başarısız: {exc}"
        except ValueError as exc:
            job.error = str(exc)
        except Exception:  # beklenmeyen hatayı da kullanıcıya bir şey söyleyerek bitir
            job.error = "Beklenmeyen bir hata oluştu:\n" + traceback.format_exc(limit=3)
        else:
            token = uuid.uuid4().hex
            entry = Session(family=cfg.font.family_name)
            buffer = io.BytesIO()
            result.font.save(buffer)
            entry.font_bytes = buffer.getvalue()
            job.stage = "önizleme çiziliyor"
            _render_preview(entry)
            entry.debug_zip = _zip_folder(folder)
            SESSIONS[token] = entry

            payload = _diagnostics_payload(result)
            payload["session"] = token
            job.payload = payload
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            job.stage = "bitti"
            job.done = True

    threading.Thread(target=run, daemon=True).start()
    return JSONResponse({"job": job_id})


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    """İşin hangi aşamada olduğunu ve ne kadar süredir sürdüğünü bildirir."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "iş bulunamadı")

    body: dict = {"stage": job.stage, "elapsed": round(job.elapsed, 1), "done": job.done}
    if job.error:
        body["error"] = job.error
    if job.payload:
        body["result"] = job.payload
    return JSONResponse(body)


# --------------------------------------------------------------------------
# Şablon modu (basılı çalışma sayfası)
# --------------------------------------------------------------------------


@app.post("/api/sheets")
def create_sheets() -> JSONResponse:
    images, sheets = build_sheets()
    token = uuid.uuid4().hex
    SESSIONS[token] = Session(sheets=sheets)
    return JSONResponse({"session": token, "pages": len(images)})


@app.get("/api/sheets/{token}.zip")
def download_sheets(token: str) -> Response:
    session = SESSIONS.get(token)
    if session is None or session.sheets is None:
        raise HTTPException(404, "oturum bulunamadı")

    images, _ = build_sheets(
        [band.text for spec in session.sheets.sheets for band in spec.bands]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images):
            archive.writestr(f"sayfa{index + 1}.png", _png(image))
        archive.writestr("sheets.json", session.sheets.to_json())
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="handwrite-sayfalar.zip"'},
    )


@app.post("/api/build")
async def build(
    session: str = Form(...),
    family: str = Form("Handwrite"),
    photos: list[UploadFile] = File(...),
) -> JSONResponse:
    entry = SESSIONS.get(session)
    if entry is None or entry.sheets is None:
        raise HTTPException(404, "oturum bulunamadı — sayfaları yeniden üretin")

    images: list[tuple[str, np.ndarray]] = []
    for upload in photos:
        images.append((upload.filename or "foto", _decode(upload, await upload.read())))

    cfg = Config()
    cfg.font.family_name = family.strip() or "Handwrite"
    entry.family = cfg.font.family_name

    try:
        result = build_from_images(images, entry.sheets, cfg)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    buffer = io.BytesIO()
    result.font.save(buffer)
    entry.font_bytes = buffer.getvalue()
    _render_preview(entry)

    payload = _diagnostics_payload(result)
    payload["session"] = session
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# Çıktılar
# --------------------------------------------------------------------------


@app.get("/api/font/{token}.ttf")
def download_font(token: str) -> Response:
    session = SESSIONS.get(token)
    if session is None or session.font_bytes is None:
        raise HTTPException(404, "font henüz üretilmedi")
    name = session.family.replace(" ", "") or "Handwrite"
    return Response(
        session.font_bytes,
        media_type="font/ttf",
        headers={"Content-Disposition": f'attachment; filename="{name}-Regular.ttf"'},
    )


@app.get("/api/debug/{token}.zip")
def download_debug(token: str) -> Response:
    """Ara adım görüntülerini indirir.

    Font beklenenden kötü çıktığında bakılacak yer burası: kağıt algılama,
    mürekkep maskesi, satır kutuları, modele giden şeritler, okunan metin ve
    çıkarılan glifler.
    """
    session = SESSIONS.get(token)
    if session is None or session.debug_zip is None:
        raise HTTPException(404, "teşhis paketi yok")
    return Response(
        session.debug_zip,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="handwrite-teshis.zip"'},
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
  :root {
    color-scheme: light dark;
    --fg:#1b1a18; --dim:#6c6862; --bg:#faf9f7; --card:#fff; --line:#e2ded7;
    --accent:#2f6f4f; --accent-fg:#fff; --warn:#9a5b00; --bad:#a33;
  }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eae7e2; --dim:#a5a099; --bg:#1a1918; --card:#232120; --line:#3a3734;
            --accent:#6fbf95; --accent-fg:#12211a; --warn:#e0a458; --bad:#e08585; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:760px; margin:0 auto; padding:40px 20px 100px; }
  h1 { font-size:30px; margin:0 0 4px; letter-spacing:-.02em; }
  .sub { color:var(--dim); margin:0 0 32px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:22px; margin-bottom:16px; }
  h2 { font-size:17px; margin:0 0 14px; display:flex; align-items:center; gap:10px; }
  .n { display:grid; place-items:center; width:25px; height:25px; border-radius:50%;
       background:var(--accent); color:var(--accent-fg); font-size:13px; font-weight:700; flex:none; }
  button { font:inherit; font-weight:500; padding:11px 18px; border-radius:9px;
           border:1px solid var(--accent); background:var(--accent); color:var(--accent-fg);
           cursor:pointer; }
  button.ghost { background:transparent; color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  input[type=text],input[type=password] { font:inherit; padding:10px 12px; border-radius:9px;
    border:1px solid var(--line); background:transparent; color:var(--fg); width:100%; }
  label { display:block; font-size:13px; color:var(--dim); margin-bottom:5px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:560px){ .grid2{ grid-template-columns:1fr; } }
  #drop { border:2px dashed var(--line); border-radius:12px; padding:28px 20px;
          text-align:center; color:var(--dim); cursor:pointer; transition:.15s; }
  #drop.hot { border-color:var(--accent); color:var(--fg); background:color-mix(in srgb,var(--accent) 8%,transparent); }
  #shots { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
  #shots figure { margin:0; position:relative; }
  #shots img { width:96px; height:120px; object-fit:cover; border-radius:8px; border:1px solid var(--line); }
  #shots button { position:absolute; top:-7px; right:-7px; width:24px; height:24px; padding:0;
                  border-radius:50%; font-size:14px; line-height:1; }
  ul.check { list-style:none; padding:0; margin:0; font-size:14px; color:var(--dim); }
  ul.check li { padding-left:24px; position:relative; margin-bottom:7px; }
  ul.check li::before { content:"✓"; position:absolute; left:4px; color:var(--accent); font-weight:700; }
  ul.check li.no::before { content:"✕"; color:var(--bad); }
  pre { white-space:pre-wrap; font:13px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;
        color:var(--dim); margin:12px 0 0; }
  #sample { font:15px/2.1 ui-monospace,monospace; background:var(--bg); border:1px solid var(--line);
            border-radius:10px; padding:16px; white-space:pre-wrap; margin-top:12px; }
  img.preview { max-width:100%; border:1px solid var(--line); border-radius:10px; margin-top:14px; }
  .bad { color:var(--bad); } .warn { color:var(--warn); }
  #warn { border:1px solid var(--warn); border-radius:10px; padding:14px 16px; margin-top:14px;
          background:color-mix(in srgb,var(--warn) 10%,transparent); font-size:14px; line-height:1.6; }
  #warn p { margin:0 0 8px; } #warn p:last-child { margin-bottom:0; }
  #warn b { color:var(--warn); }
  .muted { color:var(--dim); font-size:14px; }
  video { width:100%; max-width:420px; border-radius:10px; border:1px solid var(--line); }
  .hidden { display:none !important; }
  .bar { height:5px; background:var(--line); border-radius:3px; overflow:hidden; margin-top:14px; }
  .bar i { display:block; height:100%; width:35%; background:var(--accent);
           animation:slide 1.3s ease-in-out infinite; }
  @keyframes slide { 0%{margin-left:-35%} 100%{margin-left:100%} }
  .tabs { display:flex; gap:6px; margin-bottom:18px; }
  .tabs button { background:transparent; color:var(--dim); border-color:transparent; padding:8px 14px; }
  .tabs button.on { background:var(--card); color:var(--fg); border-color:var(--line); }
</style>
<main>
  <h1>handwrite</h1>
  <p class="sub">El yazınızı gerçek bir yazı tipine çevirir. Kutu doldurmak yok:
     bir kağıda normal yazın, fotoğrafını çekin.</p>

  <div id="keywarn" class="card hidden" style="border-color:var(--warn)">
    <h2><span class="n" style="background:var(--warn)">!</span> API anahtarı gerekli</h2>
    <p class="muted" style="margin:0 0 12px">Yazıyı okumak için bir Google AI
       anahtarı lazım. Sunucuda tanımlı değil; buraya girebilirsiniz (yalnız bu
       oturumda bellekte tutulur, diske yazılmaz).</p>
    <label for="key">GOOGLE_AI_API_KEY</label>
    <input type="password" id="key" placeholder="AQ...">
  </div>

  <div class="tabs">
    <button id="tabA" class="on">Yazım hazır</button>
    <button id="tabB">Ne yazacağımı bilmiyorum</button>
  </div>

  <div id="paneB" class="card hidden">
    <h2><span class="n">?</span> Şu metni kağıda geçirin</h2>
    <p class="muted" style="margin:0">Aşağıdaki metin bütün Türkçe harfleri,
       rakamları ve noktalama işaretlerini kapsıyor. Kendi el yazınızla, doğal
       hızınızda yazın — güzel yazmaya çalışmayın, her zamanki gibi yazın.</p>
    <div id="sample">yükleniyor…</div>
    <div class="row" style="margin-top:14px">
      <button class="ghost" id="copy">Metni kopyala</button>
      <button class="ghost" id="print">Yazdır</button>
    </div>
  </div>

  <div class="card">
    <h2><span class="n">1</span> Fotoğrafı verin</h2>
    <ul class="check" style="margin-bottom:16px">
      <li>Tükenmez ya da jel kalem — kurşun kalem soluk kalır</li>
      <li>En az 10–15 satır yazı; ne kadar çok olursa font o kadar iyi</li>
      <li>Düz ışık, gölge yok, kağıt kadraja tam otursun</li>
      <li>Çizgili defter olur — çizgiler otomatik temizlenir</li>
      <li class="no">Fotoğrafa filtre/efekt uygulamayın</li>
    </ul>

    <div id="drop">
      <b>Fotoğrafı buraya sürükleyin</b> ya da tıklayıp seçin
      <div class="muted" style="margin-top:6px">jpg · png · webp · bmp · tif</div>
      <input type="file" id="file" multiple accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff" hidden>
    </div>
    <div class="row" style="margin-top:12px">
      <button class="ghost" id="camBtn">Kamerayla çek</button>
      <input type="file" id="mobileCam" accept="image/*" capture="environment" hidden>
      <span class="muted" id="count"></span>
    </div>

    <div id="camBox" class="hidden" style="margin-top:14px">
      <video id="video" playsinline autoplay muted></video>
      <div class="row" style="margin-top:10px">
        <button id="snap">Çek</button>
        <button class="ghost" id="camClose">Kapat</button>
      </div>
    </div>

    <div id="shots"></div>
  </div>

  <div class="card">
    <h2><span class="n">2</span> Fontu üretin</h2>
    <div class="grid2" style="margin-bottom:14px">
      <div><label for="family">Font adı</label>
        <input type="text" id="family" value="Benim Yazım"></div>
    </div>
    <button id="go" disabled>Fontu üret</button>
    <div id="prog" class="hidden" style="margin-top:16px">
      <div class="row" style="justify-content:space-between">
        <b id="stage">başlıyor…</b>
        <span class="muted" id="elapsed">0 sn</span>
      </div>
      <div class="bar"><i></i></div>
      <div class="muted" id="hint" style="margin-top:8px"></div>
    </div>
    <div id="warn" class="hidden"></div>
    <pre id="log"></pre>
  </div>

  <div class="card hidden" id="out">
    <h2><span class="n">3</span> Hazır</h2>
    <div class="row">
      <a id="ttf" download><button>.ttf indir</button></a>
      <a id="dbg" download><button class="ghost">teşhis paketi indir</button></a>
    </div>
    <p class="muted" style="margin:10px 0 0">
      İndirip çift tıklayın, sisteme kurulur. Sonuç beklediğiniz gibi değilse
      teşhis paketi hangi adımın bozulduğunu gösterir — paylaşırsanız sorun
      tahminle değil bakılarak bulunur.</p>
    <img class="preview" id="prev" alt="font önizlemesi">
  </div>

  <p class="muted" style="text-align:center;margin-top:32px">
    Yüklediğiniz görüntüler yalnız bellekte işlenir, diske kaydedilmez.
  </p>
</main>
<script>
const $ = id => document.getElementById(id);
let shots = [], stream = null, aiReady = true;

// -- anahtar durumu --------------------------------------------------------
fetch('/api/status').then(r => r.json()).then(d => {
  aiReady = d.ai;
  if (!d.ai) $('keywarn').classList.remove('hidden');
});

// -- sekmeler --------------------------------------------------------------
$('tabA').onclick = () => { $('tabA').classList.add('on'); $('tabB').classList.remove('on');
  $('paneB').classList.add('hidden'); };
$('tabB').onclick = () => { $('tabB').classList.add('on'); $('tabA').classList.remove('on');
  $('paneB').classList.remove('hidden'); loadSample(); };

let sampleLines = null;
async function loadSample() {
  if (sampleLines) return;
  const d = await (await fetch('/api/sample')).json();
  sampleLines = d.lines;
  $('sample').textContent = d.lines.join('\\n');
}
$('copy').onclick = () => navigator.clipboard.writeText(sampleLines.join('\\n'))
  .then(() => { $('copy').textContent = 'Kopyalandı'; setTimeout(()=>$('copy').textContent='Metni kopyala',1500); });
$('print').onclick = () => {
  const w = window.open('', '_blank');
  w.document.write('<pre style="font:16px/2.4 monospace;padding:40px">'
    + sampleLines.map(l => l.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))).join('\\n')
    + '</pre>');
  w.document.close(); w.print();
};

// -- dosya seçme -----------------------------------------------------------
const OK_TYPES = ['image/jpeg','image/png','image/webp','image/bmp','image/tiff'];

function addFiles(list) {
  const rejected = [];
  for (const f of list) {
    if (!f.type.startsWith('image/')) { rejected.push(f.name); continue; }
    if (!OK_TYPES.includes(f.type) && f.type !== '') { rejected.push(f.name); continue; }
    shots.push(f);
  }
  if (rejected.length) {
    $('log').innerHTML = '<span class="bad">Desteklenmeyen dosya: ' + rejected.join(', ')
      + '<br>Kabul edilenler: jpg, png, webp, bmp, tif. iPhone HEIC gönderiyorsa '
      + 'Ayarlar → Kamera → Biçimler → "En Uyumlu" seçin.</span>';
  }
  render();
}

function render() {
  $('shots').innerHTML = '';
  shots.forEach((f, i) => {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    const del = document.createElement('button');
    del.textContent = '×'; del.title = 'kaldır';
    del.onclick = e => { e.stopPropagation(); shots.splice(i,1); render(); };
    fig.append(img, del); $('shots').append(fig);
  });
  $('count').textContent = shots.length ? shots.length + ' fotoğraf' : '';
  $('go').disabled = shots.length === 0;
}

$('drop').onclick = () => $('file').click();
$('file').onchange = e => addFiles(e.target.files);
['dragenter','dragover'].forEach(ev => $('drop').addEventListener(ev, e => {
  e.preventDefault(); $('drop').classList.add('hot'); }));
['dragleave','drop'].forEach(ev => $('drop').addEventListener(ev, e => {
  e.preventDefault(); $('drop').classList.remove('hot'); }));
$('drop').addEventListener('drop', e => addFiles(e.dataTransfer.files));

// -- kamera ----------------------------------------------------------------
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
$('mobileCam').onchange = e => addFiles(e.target.files);

$('camBtn').onclick = async () => {
  // Telefonda yerleşik kamera uygulaması hem daha iyi hem de odak/pozlama
  // kontrolü kullanıcıda kalıyor; masaüstünde webcam akışı açılır.
  if (isMobile) { $('mobileCam').click(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode:'environment', width:{ideal:2560}, height:{ideal:1440} } });
  } catch (err) {
    $('log').innerHTML = '<span class="bad">Kamera açılamadı: ' + err.message
      + '<br>Tarayıcı kamera iznini engellemiş olabilir; dosya seçerek de yükleyebilirsiniz.</span>';
    return;
  }
  $('video').srcObject = stream;
  $('camBox').classList.remove('hidden');
};
$('camClose').onclick = () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; $('camBox').classList.add('hidden');
};
$('snap').onclick = () => {
  const v = $('video');
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob(b => {
    shots.push(new File([b], 'kamera' + (shots.length+1) + '.png', {type:'image/png'}));
    render();
  }, 'image/png');
};

// -- üretim ----------------------------------------------------------------
// Uzun süren işi tek bir HTTP isteğiyle beklemek yerine iş kimliği alınıp
// durumu sorgulanır: kullanıcı hangi aşamada olduğunu ve kaç saniye geçtiğini
// görür. Sessizce bekleyen bir çubuk, takılmayla çalışmayı ayırt ettirmiyordu.
let timer = null;

function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

function showProgress(on) {
  $('prog').classList.toggle('hidden', !on);
  if (!on) stopTimer();
}

$('go').onclick = async () => {
  const key = $('key') ? $('key').value.trim() : '';
  if (!aiReady && !key) {
    $('log').innerHTML = '<span class="bad">Önce API anahtarını girin.</span>';
    return;
  }
  $('go').disabled = true; $('log').textContent = '';
  showProgress(true);
  $('stage').textContent = 'fotoğraf yükleniyor…';
  $('elapsed').textContent = '0 sn'; $('hint').textContent = '';

  const form = new FormData();
  form.append('family', $('family').value || 'Handwrite');
  if (key) form.append('api_key', key);
  shots.forEach(f => form.append('photos', f));

  let r;
  try { r = await fetch('/api/read', {method:'POST', body:form}); }
  catch (err) {
    $('log').innerHTML = '<span class="bad">Sunucuya ulaşılamadı: ' + err.message + '</span>';
    $('go').disabled = false; showProgress(false); return;
  }
  if (!r.ok) {
    const e = await r.json().catch(() => ({detail:'bilinmeyen hata'}));
    $('log').innerHTML = '<span class="bad">' + e.detail + '</span>';
    $('go').disabled = false; showProgress(false); return;
  }

  const { job } = await r.json();
  const d = await poll(job);
  $('go').disabled = false; showProgress(false);
  if (!d) return;

  // Uyarı varsa font yine üretilir ama beklenenden kötü olur; kullanıcının
  // bunu indirmeden önce görmesi gerek.
  if (d.warnings && d.warnings.length) {
    $('warn').innerHTML = '<p><b>Dikkat</b></p>'
      + d.warnings.map(w => '<p>' + w.replace(/[<&]/g, c => c === '<' ? '&lt;' : '&amp;') + '</p>').join('');
    $('warn').classList.remove('hidden');
  } else {
    $('warn').classList.add('hidden');
  }

  let out = d.summary.join('\\n');
  if (d.page_errors.length) out += '\\n\\n! ' + d.page_errors.join('\\n! ');
  if (d.rejected.length)    out += '\\n\\nAtlanan satırlar:\\n  ' + d.rejected.join('\\n  ');
  if (d.transcriptions.length) {
    out += '\\n\\nOkunan metin:\\n' + d.transcriptions
      .map(t => '  [' + t.confidence.toFixed(2) + '] ' + t.text).join('\\n');
  }
  if (d.synthesis) out += '\\n\\nEksik karakterler:\\n  ' + d.synthesis.join('\\n  ');
  $('log').textContent = out;

  $('ttf').href = '/api/font/' + d.session + '.ttf';
  $('dbg').href = '/api/debug/' + d.session + '.zip';
  $('prev').src = '/api/preview/' + d.session + '.png?t=' + Date.now();
  $('out').classList.remove('hidden');
  $('out').scrollIntoView({behavior:'smooth', block:'nearest'});
};

async function poll(job) {
  return new Promise(resolve => {
    timer = setInterval(async () => {
      let s;
      try { s = await (await fetch('/api/job/' + job)).json(); }
      catch { return; }

      $('stage').textContent = s.stage;
      $('elapsed').textContent = Math.round(s.elapsed) + ' sn';
      // Beklemenin normal olup olmadığını kullanıcı bilmeli.
      $('hint').textContent = s.elapsed > 90
        ? 'Alışılmadık şekilde uzun sürüyor. Gemini yoğun olabilir; 5 dakikada '
          + 'tamamlanmazsa kendiliğinden durur ve size haber verir.'
        : 'Tipik süre: sayfa başına 20–40 saniye.';

      if (!s.done) return;
      stopTimer();
      if (s.error) {
        $('log').innerHTML = '<span class="bad">' + s.error.replace(/\\n/g,'<br>') + '</span>';
        resolve(null);
      } else {
        resolve(s.result);
      }
    }, 900);
  });
}
</script>
</html>
"""
