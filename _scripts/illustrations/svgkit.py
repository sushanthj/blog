"""Shared helpers for generating blog illustration SVGs in the site theme.

Style contract lives in .claude/skills/blog-illustrations/SKILL.md.
Every illustration SVG in images/ should be produced by a generator script in
this directory so it can be regenerated, restyled, and kept geometrically exact.
"""
import math

# ---- palette: kb-dark, mirrors _sass/_includes/_blog.scss + _includes/mermaid.html ----
BG = "#0f1117"        # page background
CARD = "#1a1d27"      # panel / node fill
CARD2 = "#131620"     # inset panel fill
BORDER = "#2a2d3a"    # neutral borders
TEXT = "#e0e0e6"      # primary text
MUTED = "#8b8fa3"     # secondary text, neutral arrows, axes
FAINT = "#565b6e"     # de-emphasised strokes, dividers

BLUE = "#6c9eff"      # networks, model outputs, data distributions
GREEN = "#7ddba0"     # positive, outputs, expert data
RED = "#f38ba8"       # negative, errors, stochastic noise
GOLD = "#e6c07b"      # parameters, highlights
ORANGE = "#e8a87c"    # secondary curves
PURPLE = "#b794f6"    # latents, priors
TEAL = "#7fd8cf"      # weights, auxiliary quantities

FONT = "Muli, sans-serif"

HEAD_LEN = 9  # arrowhead length; lines are shortened by this so the tip lands on target


class SVG:
    def __init__(self, w, h, label=""):
        self.w, self.h, self.label = w, h, label
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    # ---- primitives ----
    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=1.5, opacity=None, dash=None):
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"'
        if rx:
            s += f' rx="{rx}"'
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            s += f' opacity="{opacity}"'
        self.add(s + "/>")

    def card(self, x, y, w, h, stroke=BORDER, fill=CARD, sw=1.5, rx=10, dash=None):
        self.rect(x, y, w, h, fill, rx=rx, stroke=stroke, sw=sw, dash=dash)

    def circle(self, cx, cy, r, fill, stroke=None, sw=1.5, opacity=None):
        s = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"'
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        if opacity is not None:
            s += f' opacity="{opacity}"'
        self.add(s + "/>")

    def line(self, x1, y1, x2, y2, color=MUTED, w=1.5, dash=None, opacity=None, cap=None):
        s = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            s += f' opacity="{opacity}"'
        if cap:
            s += f' stroke-linecap="{cap}"'
        self.add(s + "/>")

    def path(self, d, stroke=None, w=2, fill="none", dash=None, opacity=None, cap=None):
        s = f'<path d="{d}" fill="{fill}"'
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{w}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            s += f' opacity="{opacity}"'
        if cap:
            s += f' stroke-linecap="{cap}"'
        self.add(s + "/>")

    def text(self, x, y, s, fill=TEXT, size=12, anchor="middle", weight=None,
             italic=False, rotate=None, family=None):
        t = f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}"'
        if weight:
            t += f' font-weight="{weight}"'
        if italic:
            t += ' font-style="italic"'
        if rotate:
            t += f' transform="rotate({rotate} {x} {y})"'
        if family:
            t += f' font-family="{family}"'
        self.add(t + f">{s}</text>")

    # ---- arrows: tip ALWAYS lands exactly on (x2, y2) ----
    def head(self, x2, y2, angle, color):
        a = math.radians(angle)
        hx, hy = x2 - HEAD_LEN * math.cos(a), y2 - HEAD_LEN * math.sin(a)
        px, py = math.sin(a) * 3.6, -math.cos(a) * 3.6
        self.add(f'<polygon points="{hx+px:.1f},{hy+py:.1f} {x2:.1f},{y2:.1f} '
                 f'{hx-px:.1f},{hy-py:.1f}" fill="{color}"/>')

    def arrow(self, x1, y1, x2, y2, color=MUTED, w=1.8, dash=None):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = math.radians(ang)
        self.line(x1, y1, x2 - (HEAD_LEN - 2) * math.cos(a), y2 - (HEAD_LEN - 2) * math.sin(a),
                  color=color, w=w, dash=dash)
        self.head(x2, y2, ang, color)

    def arrow_path(self, d, tip, angle, color=MUTED, w=1.8, dash=None):
        """Curved arrow: caller supplies the path and the tip position + direction."""
        self.path(d, stroke=color, w=w, dash=dash)
        self.head(tip[0], tip[1], angle, color)

    # ---- curves ----
    def curve(self, pts, color, w=2, dash=None, fill_opacity=None, base=None, opacity=None):
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} " + " ".join(
            f"L{x:.1f},{y:.1f}" for x, y in pts[1:])
        if fill_opacity is not None and base is not None:
            self.path(d + f" L{pts[-1][0]:.1f},{base} L{pts[0][0]:.1f},{base} Z",
                      fill=color, opacity=fill_opacity)
        self.path(d, stroke=color, w=w, dash=dash, opacity=opacity)

    # ---- axes ----
    def axis_x(self, x0, x1, y, ticks=(), color=MUTED, tick_size=10, label_dy=17):
        self.line(x0, y, x1, y, color=color, w=1.4)
        for tx, lab in ticks:
            self.line(tx, y, tx, y + 5, color=color, w=1)
            if lab:
                self.text(tx, y + label_dy, lab, fill=MUTED, size=tick_size)

    def axis_y(self, x, y0, y1, color=MUTED):
        self.line(x, y0, x, y1, color=color, w=1.4)

    def save(self, path):
        body = "\n  ".join(self.parts)
        svg = (f'<svg viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg" '
               f'role="img" aria-label="{self.label}" font-family="{FONT}">\n'
               f'  <rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{BG}"/>\n'
               f'  {body}\n</svg>\n')
        with open(path, "w") as f:
            f.write(svg)
        print("wrote", path)


# ---- math helpers ----
def gauss(x, mu, sig):
    return math.exp(-((x - mu) ** 2) / (2 * sig ** 2))


def mixture(x, comps):
    """comps: list of (mu, sigma, weight)."""
    return sum(w * gauss(x, m, s) for m, s, w in comps)


def fn_curve(x0_px, x1_px, base, amp, fn, x0_v, x1_v, step_px=4):
    """Sample fn over value range [x0_v, x1_v] mapped to pixels [x0_px, x1_px];
    normalise the peak to `amp` above `base`. Returns [(x_px, y_px)]."""
    n = max(2, int((x1_px - x0_px) / step_px))
    vals, pts = [], []
    for i in range(n + 1):
        t = i / n
        vals.append(fn(x0_v + t * (x1_v - x0_v)))
    peak = max(vals) or 1.0
    for i in range(n + 1):
        t = i / n
        pts.append((x0_px + t * (x1_px - x0_px), base - amp * vals[i] / peak))
    return pts
