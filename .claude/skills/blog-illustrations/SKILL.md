---
name: blog-illustrations
description: Style guide and workflow for every diagram and illustration on the blog — when to use mermaid vs generated SVG, the kb-dark palette, typography, and geometry rules. Use whenever creating or editing a figure in _posts/.
---

# Blog illustrations: style + workflow

Two kinds of figures exist on this blog. **Pick the right kind first:**

| Kind | Use for | How |
|---|---|---|
| **Mermaid fence** | anything box-and-arrow: pipelines, decision trees, training loops, comparisons of stages | ` ```mermaid ` block in the post — rendered client-side by `_includes/mermaid.html` |
| **Generated SVG** | anything mermaid can't express: curves, distributions, histograms, scatter plots, network-node diagrams, gauges, game boards | a Python generator in `_scripts/illustrations/` using `svgkit.py`, output committed to `images/` |

Never hand-author SVG coordinates for either kind. Hand-tuned arrows are how we
got arrows that stop short of boxes; generated geometry is exact by construction.

## Palette (kb-dark)

Defined once in `_scripts/illustrations/svgkit.py`; mirror of `_sass/_includes/_blog.scss`
and the mermaid theme in `_includes/mermaid.html`. Import the tokens, never inline hex.

| Token | Hex | Role |
|---|---|---|
| `BG` | `#0f1117` | canvas background |
| `CARD` / `CARD2` | `#1a1d27` / `#131620` | panels, node fills |
| `BORDER` | `#2a2d3a` | neutral borders |
| `TEXT` / `MUTED` / `FAINT` | `#e0e0e6` / `#8b8fa3` / `#565b6e` | text hierarchy, axes |
| `BLUE` | `#6c9eff` | networks, model outputs, data distributions |
| `GREEN` | `#7ddba0` | positive, outputs, expert data |
| `RED` | `#f38ba8` | negative, errors, stochastic noise |
| `GOLD` | `#e6c07b` | parameters, highlights |
| `ORANGE` | `#e8a87c` | secondary curves |
| `PURPLE` | `#b794f6` | latents, priors |
| `TEAL` | `#7fd8cf` | weights, auxiliary quantities |

Mermaid classDefs use the same hues: node fill `#1a1d27` (or tinted `#17301f` green /
`#301a22` red / `#2a2340` purple), stroke = accent, text `#e0e0e6`.

## Formality

Figures must read like publication figures, not slides:

- **Neutral ink first.** Default everything to `TEXT`/`MUTED`/`FAINT`. Accent hues are
  reserved for *encoding meaning* (a curve identity, a positive/negative signal), never
  decoration. Budget: 2–3 accent hues per figure; the same hue means the same thing
  across a post.
- **No decorative pictograms.** No emoji, ✗/✓ glyphs, or stars. A verdict is a written
  label ("no expert data here"), not a red cross; an agent is a state marker `s₀`.
  Exception: a single domain glyph that *is* the subject of the panel (e.g., the game
  being played) may stay — one per figure, drawn in the palette, never as commentary.
- **No exclamation marks, scare quotes, or first-person phrasing** in labels. Sentence
  case throughout.
- **Boxes are neutral** (`BORDER` stroke, `rx≤8`) unless the box itself is the subject
  of the figure. Titles identify panels; color does not.
- **Thin strokes**: 1.2–2px for lines and curves, arrowheads ≤9px.

## Typography

- Font: `Muli, sans-serif` (site font) — set once on the `<svg>` root.
- Sizes: figure title 15 / weight 600 · panel header 12.5–13 / weight 600 ·
  body label 11–12 · tick/small 10. Bold (700) only for single-character emphasis.
- Variables italic where practical (π, θ subscripts via `<tspan>`).
- Captions under a figure go in the *post*, not the SVG:
  `<div style="text-align:center;color:#8b8fa3;font-size:13px;margin:-0.5em 0 2em">…</div>`

## Geometry rules

- **Arrows must touch.** Use `svgkit.SVG.arrow()` / `arrow_path()` — the tip is placed
  exactly on the target edge point. Compute the edge point (box edge, circle edge at
  `cx ± r`), never eyeball it.
- **Curves are functions.** Gaussians, mixtures, and any analytic curve come from
  `fn_curve()` sampled at ≤4px steps — never freehand bezier approximations.
- **Scatter data is seeded.** `random.Random(<fixed seed>)` so regeneration is stable.
- **Baselines and axes** via `axis_x`/`axis_y`; ticks labelled in `MUTED`.

## Mermaid specifics

- Subgraphs with no connecting edges are laid out **side by side** and shrink.
  Chain them with invisible links (`A ~~~ B`) to stack them full-width.
- `linkStyle` accepts only ONE index per statement here (`linkStyle 0 stroke:#7ddba0`);
  comma lists fail to parse.
- HTML labels are enabled (`securityLevel: loose`): `<b> <i> <sub> <sup> <img>` work.
  Glyph images inside nodes: small themed SVGs in the post's image folder.

## Workflow

1. Write/edit the generator in `_scripts/illustrations/` (one script per post),
   importing from `svgkit.py`. Output path: the post's folder under `images/`.
2. Run it: `python3 _scripts/illustrations/<script>.py`
3. **Look at the result** — rasterise and inspect:
   `google-chrome --headless=new --screenshot=/tmp/x.png --window-size=<W>,<H> file://<abs>.svg`
4. Validate mermaid blocks with the Docker CLI (needs a world-writable mount dir):
   `docker run --rm -v "$PWD":/data minlag/mermaid-cli -i /data/d.mmd -o /data/d.svg`
5. `make build` and check the built page for broken refs.

## Thumbnails (featured_image)

Card thumbnails can't be mermaid. Render the post's signature diagram to PNG with the
Docker mermaid CLI (`-b "#0f1117" -w 1200 -c <theme config>`) and point `featured_image`
at the PNG; the theme config mirroring `_includes/mermaid.html` is documented there.
For SVG illustrations, the generated `.svg` itself can be the featured image.
