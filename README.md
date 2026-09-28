# handwrite

🇬🇧 English · 🇹🇷 [Türkçe](README.tr.md)

**Turn a photo of your handwriting into a real OpenType font.**
No boxes, no grid template: write normally on any sheet of paper, take a photo, get a `.ttf`.

```bash
handwrite serve
```

Your browser opens. Drop in a photo (or use the camera) and download your font.
If you don't have anything written yet, the UI gives you a short text that covers every character.

Measured: a single notebook page (14 lines, ~500 characters) becomes a complete
87-glyph font in **~30 seconds**.

> The web UI and the default sample text are currently Turkish. The pipeline itself is
> language-agnostic for Latin script and covers the full English alphabet.

Prefer the command line?

```bash
export GOOGLE_AI_API_KEY=...
handwrite read notebook.jpg -o Mine.ttf --preview      # free-form page, the model reads it

handwrite sheets -o work                               # printable worksheet (no AI needed)
handwrite build work/sheets.json photo*.jpg -o Mine.ttf
```

---

## Where exactly the AI is used

The split is deliberate and fits in one sentence:

> **The model says *what* is written; dynamic programming finds *where* it is.**

| job | who | why |
|---|---|---|
| Read what the page says | Gemini | A semantic task: context helps, and it knows the language |
| Find the pixels of each letter | DP | Bounding boxes from vision models are unreliable; fonts work in em units and an "approximate" box cuts off a glyph's foot |
| Discard broken glyphs | clustering | Measurable and cheap |
| Create letters that never appeared | 3-tier synthesis | See below |

Asking the model for pixel coordinates is tempting but wrong. Measured: Gemini 2.5 Flash
reads synthetic Turkish handwriting at **CER 0.0038** (4 of 6 lines exactly right). The
remaining errors are single-character insertions or deletions.

### Robust to model mistakes

Reading doesn't need to be perfect. Segmentation is two-level (words first, then
characters inside each word), so a one-letter reading error only affects **its own word**;
the rest of the line still falls into place. With single-level alignment the same error
would shift every glyph on the line. The few damaged glyphs never reach the font: they
are outliers within their own letter's cluster and get filtered out.

## Missing characters: one page isn't enough, so we fill the gaps

Natural prose gives you plenty of lowercase letters, but you rarely write "Q" or "%".
The real distribution on a 500-character page:

| group | status |
|---|---|
| lowercase | plenty (`a` ~155, `e` ~115 samples); the only real gap is `j` |
| uppercase | mostly missing: only sentence starts |
| digits, punctuation | only what the text happens to contain |
| `q w x` | absent |

**~44 of 86 characters never appear on a single page.** The gap is filled in three tiers,
ordered by reliability:

1. **Composition.** Built from the writer's *real* strokes: `Ç` = `C` + the cedilla of `ç`,
   `İ` = `I` + the dot of `i`. Every part comes from the same hand, so the result is
   near-perfect.
2. **Case transfer.** Some capitals are just scaled lowercase (`c/C`, `o/O`, `s/S`, `v/V`…).
   The writer's own letter is scaled to cap height and the pen weight is compensated.
   Letters whose shape truly changes (`a/A`, `e/E`, `g/G`) are excluded.
3. **Reference shape + your pen.** For the rest, a reference typeface's letter skeleton is
   used, with your measured stroke weight, proportions, corner roundness and hand jitter
   applied. Widths are calibrated against your own letters (measured correction ~0.89:
   printed letters are wider than handwriting).

Every generated character is clearly flagged in the diagnostics screen. Presenting a
synthesized letter as the user's own handwriting would be dishonest.

Direct image generation ("draw a Q in this hand") was deliberately not used: today's models
produce things that *look like* handwriting, but the letter skeleton comes out wrong, stroke
weight drifts, and they can't imitate a specific hand. A font glyph forgives far less than
a picture.

---

## Why no boxes

The boxes in tools like Calligraphr aren't an arbitrary design choice; they sidestep a real
problem: **knowing which letter a scribble is, is hard.** The box makes the user answer
that question. The box's position *is* the label.

handwrite never asks. The user writes a text the system *already knows*, which turns
"recognition" into "alignment":

> Given a text of N characters and a photo of it written out, where should the ink be
> cut, left to right, into exactly N pieces?

That's a solved class of problem: constrained optimal segmentation. No trained model,
no training data, no GPU. Dynamic programming is enough.

## Pipeline

```
photo → page registration → ink separation → lines → slant correction
      → character segmentation → glyph normalization → variant selection
      → vectorization → TTF
```

Each stage lives in its own module and is testable on its own.

### Page registration (`template.py`, `preprocess.py`)

In template mode, the worksheet has ArUco markers in its four corners. They solve three
things at once:

- **Perspective.** A skewed phone photo is mapped to canonical page coordinates via
  homography. The skew is solved, not guessed.
- **Line-to-text matching.** Which band holds which sentence is known exactly.
- **Page identity.** Each page carries different marker IDs, so photos can be uploaded
  in any order, unnamed.

### Ink separation

One rule removes the printed guide lines and sample text:

> A pixel is handwriting if it is clearly darker than what the template says should be there.

Because we generate the template, we know every pixel's nominal gray value. How the
print-and-scan chain shifts those values (`observed ≈ a·nominal + b`) is measured from the
page itself, so it self-calibrates across printers, paper and lighting. When a stroke
crosses the baseline, it's darker than the printed line, so the letter survives. Crude
rectangle masking can't do that.

### Character segmentation (`segment.py`): the core

Two costs are balanced:

**Cut cost.** A cut isn't a straight vertical line but a *seam* that can drift sideways as
it goes down. Cutting ink near the baseline is discounted: in joined-up writing that's
exactly where the connecting stroke is, and where the cut *should* happen.

**Width cost.** "m" is wide, "i" is narrow. Where the ink gives no hint (fully cursive
writing), this prior puts the cut in the right place. After one pass the priors are
re-estimated from the writer's *own* hand. Some people write a narrow "m", some a wide "a".

Mixed writing goes through the same mechanism: if letters are separate, cut cost is zero
in the gap and the cut lands there; if they're joined, the width prior picks the cut that
crosses the least ink.

### Consistency (`glyph.py`)

Put raw crops straight into a font and you get a ransom note. So:

- **One slant per page**, measured and removed. Per-line correction creates inconsistency.
- **Measured x-height**, from the actual x-height letters *after* segmentation. Estimating
  it from a line's ink profile mistakes cap height for x-height on all-caps lines.
- **Plausibility filter.** An "a" twice the x-height has stolen a piece of its neighbour.
  It's dropped before clustering.
- **Vertical alignment.** On paper, vertical jitter is random and invisible. In a font the
  same glyph repeats, so jitter becomes *systematic*: an "a" that sat too high sits too high
  everywhere. Vertical position is therefore mostly normalized.

### Natural look: variant cycling (`fontbuild.py`)

The number one reason handwriting fonts look fake is that every "a" is pixel-identical.
Several real samples of each character go into the font and are cycled with OpenType `calt`:

```fea
sub @BASE @BASE' by @ALT1;    # after a base letter -> variant 1
sub @ALT1 @BASE' by @ALT2;    # after variant 1     -> variant 2
```

No third rule is needed: after ALT2 nothing matches and the letter stays in its base form,
so the cycle `BASE→ALT1→ALT2→BASE` falls out naturally.

Variants are picked from the cluster *core*. Searching for diversity as "the sample
farthest from the center" selects exactly the damaged samples: an "o" with an "r" stuck
to it is, by definition, the farthest one.

### Measured metrics

Advance widths, side bearings and the width of the space are not guessed; they're measured
from real letter spacing on the paper. That's how a person's writing rhythm gets into the font.

## Measuring quality

`handwrite.synth` generates synthetic handwriting while recording **which character drew
every ink pixel**. Segmentation is therefore scored by counting pixels, not by eye
(`handwrite.bench`):

- **purity**: what fraction of the ink assigned to a glyph really belongs to that letter?
- **coverage**: what fraction of the letter's ink made it into the glyph?

Both are needed: purity alone rewards cutting too narrow, coverage alone rewards cutting too wide.

Current results: 4 pages, 1,138 evaluated characters, synthetic writing slanted 9° and 45%
joined, with camera degradation applied (perspective, uneven lighting, blur, noise):

| metric | median | mean |
|---|---|---|
| purity | 1.000 | 0.904 |
| coverage | 0.985 | 0.895 |

82.3% of characters score above 0.80 on both; 7.3% score below 0.40. Outlier rejection
cleans up that tail, so the glyphs that reach the font are cleaner than the mean suggests.

Parameters were tuned across different writing styles (slanted cursive / upright print /
back-slanted), not on a single page. The setting that scores best on one page scores worst
on the others.

## Known limitations

- **Synthesized letters aren't your hand.** The ~32 characters from tier 3 (mostly
  capitals, digits, punctuation) blend in, but a careful eye will notice. For full, real
  coverage use template mode or a second "fill the gaps" page.
- **Detecting reading errors is unreliable.** Alignment cost is used to flag suspicious
  words, but two-level segmentation partly absorbs errors by shifting word boundaries, so
  the cost doesn't rise as much as expected. The real safeguard is outlier rejection.
- **Fully cursive writing** is the hardest case. It works, but with a higher error rate.
- **No kerning pairs.** Measured side bearings give natural spacing, but there aren't
  enough samples for true pairwise kerning.
- **No ligatures (`liga`).** Entry and exit strokes of joined writing aren't modeled;
  letters stand apart.
- **Character set:** Turkish + English letters, digits and common punctuation (86 chars).
  Other accented letters (é, ñ, ß…) aren't included yet.
- Template mode is limited to 12 pages by the ArUco dictionary.

## Install

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[web]'
.venv/bin/handwrite serve
```

Run the tests with `.venv/bin/pip install -e '.[web,test]' && .venv/bin/pytest` (58 tests, ~4 min).

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\pip install -e ".[web]"
.venv\Scripts\handwrite serve
```

You need a free [Google AI Studio](https://aistudio.google.com/apikey) API key for free-form
mode. Set it as `GOOGLE_AI_API_KEY`, or paste it into the web UI (kept in memory only,
never written to disk). Uploaded photos are processed in memory and not stored.

## Modules

| file | job |
|---|---|
| `config.py` | all tunable constants and typographic priors |
| `template.py` | worksheet generation, page geometry |
| `preprocess.py` | perspective correction, lighting, ink separation |
| `lines.py` | line extraction, vertical metrics, slant |
| `segment.py` | constrained character segmentation (core algorithm) |
| `glyph.py` | normalization, outlier rejection, variant selection |
| `vectorize.py` | bitmap → Bézier contours |
| `fontbuild.py` | TTF output, `calt` variant cycling |
| `pipeline.py` | end-to-end flow and diagnostics |
| `ai/gemini.py` | line-by-line handwriting reading (Gemini) |
| `synthesize.py` | synthesis of missing glyphs (3 tiers) |
| `synth.py` | synthetic handwriting generator (for tests) |
| `bench.py` | segmentation accuracy benchmark |
| `specimen.py` | specimen page rendering |
| `web.py` | web UI |
| `cli.py` | command line (development and batch jobs) |

## License

[MIT](LICENSE) © 2026 Alper Alyaz
