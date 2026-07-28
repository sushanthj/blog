"""Regenerates the RL theory basics 1 illustrations in the site theme.
Run: python3 _scripts/illustrations/rl_basics_1.py

Follows the formality rules in .claude/skills/blog-illustrations/SKILL.md:
neutral ink first, 2-3 accent hues, no pictograms, publication-style labels.
"""
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from svgkit import (SVG, BG, CARD, CARD2, BORDER, TEXT, MUTED, FAINT, BLUE, GREEN,
                    RED, GOLD, ORANGE, PURPLE, TEAL, gauss, mixture, fn_curve)

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "images", "blog", "reinforcement-learning", "RL_theory_basics_1")

NODE_FILL = "#141721"
PI_THETA = 'Policy π<tspan font-size="10" dy="3">θ</tspan>'


def neural_net(s, layers, conn_opacity=0.3):
    """layers: list of (x, [cy...], r, stroke_color). Connections, then nodes."""
    for (x1, cys1, _, _), (x2, cys2, _, _) in zip(layers, layers[1:]):
        for cy1 in cys1:
            for cy2 in cys2:
                s.line(x1, cy1, x2, cy2, color=FAINT, w=0.7, opacity=conn_opacity)
    for x, cys, r, col in layers:
        for cy in cys:
            s.circle(x, cy, r, NODE_FILL, stroke=col, sw=1.5)


def ys(center, n, gap):
    return [center + (i - (n - 1) / 2) * gap for i in range(n)]


def obs_grid(s, x, y, cell, n, seed=3):
    """Abstract observation: an n-by-n grid of pixel intensities."""
    rng = random.Random(seed)
    for i in range(n):
        for j in range(n):
            s.rect(x + j * cell, y + i * cell, cell - 1.5, cell - 1.5, BLUE,
                   opacity=round(rng.uniform(0.06, 0.5), 2))


# ---------------------------------------------------------------- svg1: NN policy
def svg1():
    s = SVG(800, 350, "A neural network policy: robot state in, action out")
    s.card(30, 115, 145, 80, rx=8)
    s.text(102, 150, "Robot state", size=14, weight=600)
    s.text(102, 175, "(x, y, z)", fill=MUTED, size=12)

    layers = [(280, ys(155, 3, 60), 14, BLUE),
              (400, ys(155, 5, 50), 14, MUTED),
              (520, ys(155, 2, 60), 14, GREEN)]
    neural_net(s, layers)
    s.text(400, 308, PI_THETA, size=14, weight=600)

    s.arrow(180, 155, 264, 155, w=1.4)
    s.arrow(536, 155, 636, 155, w=1.4)

    s.card(638, 120, 130, 70, rx=8)
    s.text(703, 160, "Action", size=14, weight=600)
    s.save(os.path.join(OUT, "svg1_neural_net_policy.svg"))


# ---------------------------------------------------------------- svg2: discrete policy
def svg2():
    s = SVG(900, 400, "A discrete policy: observed game frame in, distribution over actions out")
    s.card(30, 105, 160, 140, rx=8)
    s.text(110, 135, "Game frame", size=13, weight=600)
    s.path("M 110,198 L 138,181 A 32,32 0 1 0 138,215 Z", fill=GOLD)
    s.circle(115, 186, 3, BG)
    for cx, cy in [(68, 198), (78, 220), (62, 180)]:
        s.circle(cx, cy, 3.5, GOLD)

    layers = [(295, ys(175, 3, 60), 13, BLUE),
              (395, ys(175, 5, 50), 13, MUTED),
              (495, ys(175, 2, 60), 13, GREEN)]
    neural_net(s, layers)
    s.text(395, 328, PI_THETA, size=14, weight=600)

    s.arrow(195, 175, 281, 175, w=1.4)
    s.arrow(509, 175, 601, 175, w=1.4)

    s.text(727, 62, "P(action | s)", size=13, weight=600)
    base = 280
    for x, p, g in [(615, 0.5, "↑"), (675, 0.2, "↓"), (735, 0.1, "←"), (795, 0.2, "→")]:
        h = p / 0.5 * 195
        s.rect(x, base - h, 45, h, GREEN, rx=2, opacity=0.55)
        s.text(x + 22.5, base - h - 8, f"{p}", fill=MUTED, size=11)
        s.text(x + 22.5, 306, g, fill=TEXT, size=16)
    s.line(605, base, 850, base, color=MUTED, w=1.4)
    s.save(os.path.join(OUT, "svg2_pacman_policy.svg"))


# ---------------------------------------------------------------- svg3: continuous policy
def svg3():
    s = SVG(800, 350, "A continuous policy: car state in, steering angle out")
    s.card(20, 105, 170, 90, rx=8)
    s.text(105, 140, "Car state", size=14, weight=600)
    s.text(105, 168, "(speed, position, heading)", fill=MUTED, size=11)

    layers = [(280, ys(150, 3, 60), 14, BLUE),
              (400, ys(150, 5, 50), 14, MUTED),
              (520, ys(150, 2, 60), 14, GREEN)]
    neural_net(s, layers)
    s.text(400, 298, PI_THETA, size=14, weight=600)

    s.arrow(195, 150, 264, 150, w=1.4)
    s.arrow(536, 150, 596, 150, w=1.4)

    s.path("M 600,220 A 80,80 0 0 1 760,220", stroke=MUTED, w=2)
    for ang, lab, lx, ly in [(135, "−45°", 607, 150), (90, "0°", 680, 124),
                             (45, "+45°", 753, 150)]:
        a = math.radians(ang)
        s.line(680 + 72 * math.cos(a), 220 - 72 * math.sin(a),
               680 + 86 * math.cos(a), 220 - 86 * math.sin(a), color=MUTED, w=1.5)
        s.text(lx, ly, lab, fill=MUTED, size=11)
    s.circle(680, 220, 3.5, MUTED)
    a = math.radians(65)
    s.line(680, 220, 680 + 70 * math.cos(a), 220 - 70 * math.sin(a),
           color=BLUE, w=2.5, cap="round")
    s.text(724, 154, "θ", fill=TEXT, size=16, italic=True)
    s.text(680, 258, "Steering angle", fill=MUTED, size=12)
    s.save(os.path.join(OUT, "svg3_steering_policy.svg"))


# ---------------------------------------------------------------- svg4: mean averaging
def x_of_deg(deg, cx=400, scale=10):
    return cx + deg * scale


def svg4():
    s = SVG(800, 400, "Bimodal expert data: the mean-squared-error fit predicts the mean, where no expert ever drove")
    s.text(400, 30, "The mean-averaging problem", size=15, weight=600)

    base = 320
    comps = [(-15, 5, 0.5), (15, 5, 0.5)]
    pts = fn_curve(100, 700, base, 118, lambda d: mixture(d, comps), -30, 30)
    s.curve(pts, BLUE, w=2, fill_opacity=0.1, base=base)

    rng = random.Random(7)
    for _ in range(60):
        mode = rng.choice([-15, 15])
        d = max(-29, min(29, rng.gauss(mode, 5)))
        x = x_of_deg(d)
        cy = base - 118 * mixture(d, comps) / mixture(-15, comps)
        y = base - (base - cy) * rng.uniform(0.15, 0.92)
        s.circle(x, y, 2.6, GREEN, opacity=0.55)

    s.axis_x(100, 700, base, ticks=[(x_of_deg(d), f"{d}°") for d in range(-30, 31, 10)],
             tick_size=10)
    s.text(400, 368, "Steering angle", fill=MUTED, size=12)

    s.line(400, 78, 400, base, color=RED, w=1.5, dash="7 5")
    s.text(412, 96, "model prediction: 0° (the mean)", fill=RED, size=11.5, anchor="start")
    s.text(412, 113, "no expert data here", fill=MUTED, size=10.5, italic=True, anchor="start")

    s.circle(108, 70, 3, GREEN, opacity=0.7)
    s.text(118, 74, "expert demonstrations (100 drivers)", fill=MUTED, size=10.5, anchor="start")
    s.line(100, 90, 116, 90, color=BLUE, w=2)
    s.text(118, 94, "true action distribution", fill=MUTED, size=10.5, anchor="start")
    s.save(os.path.join(OUT, "svg4_mean_averaging.svg"))


# ------------------------------------------------- left panel shared by 4b and 5
def nn_to_params(s, param_lines, brace_y0, brace_y1, label, label_y):
    s.card(30, 120, 130, 100, rx=8)
    s.text(95, 158, "Neural net", size=13, weight=600)
    s.text(95, 186, 'π<tspan font-size="10" dy="3">θ</tspan>', fill=MUTED, size=15)

    s.card(40, 275, 110, 40, rx=6)
    s.text(95, 300, "Robot state", fill=MUTED, size=12)
    s.arrow(95, 275, 95, 222, w=1.4)
    s.arrow(160, 170, 206, 170, w=1.4)

    mid = (brace_y0 + brace_y1) / 2
    s.path(f"M220,{brace_y0} C215,{brace_y0} 213,{brace_y0+2} 213,{brace_y0+7} "
           f"L213,{mid-7} C213,{mid-2} 211,{mid} 207,{mid} "
           f"C211,{mid} 213,{mid+2} 213,{mid+7} "
           f"L213,{brace_y1-7} C213,{brace_y1-2} 215,{brace_y1} 220,{brace_y1}",
           stroke=FAINT, w=1.3)
    for x, y, txt, size in param_lines:
        s.text(x, y, txt, fill=TEXT, size=size, italic=True)
    s.text(216, label_y, label, fill=MUTED, size=10)
    s.text(216, label_y + 12, "parameters", fill=MUTED, size=10)


def steering_axes(s, base=310, ytop=75):
    s.axis_y(360, base, ytop)
    s.axis_x(360, 760, base,
             ticks=[(560 + d * (200 / 30), f"{d}°") for d in range(-30, 31, 10)],
             tick_size=10)
    s.text(560, 351, "Steering angle", fill=MUTED, size=12)


def bimodal_curve(s, base=310, amp=210):
    comps = [(-15, 5, 0.5), (15, 5, 0.5)]
    pts = fn_curve(360, 760, base, amp, lambda d: mixture(d, comps), -30, 30, step_px=3)
    s.curve(pts, BLUE, w=2, fill_opacity=0.1, base=base)


# ---------------------------------------------------------------- svg4b: gaussian policy
def svg4b():
    s = SVG(800, 420, "A single-Gaussian policy places its peak between the two modes of bimodal expert data")
    nn_to_params(s, [(248, 158, "μ", 16), (248, 194, "σ", 16)],
                 138, 196, "Gaussian", 218)
    s.text(95, 345, "two parameters per state", fill=MUTED, size=11, italic=True)

    s.line(310, 50, 310, 370, color=FAINT, w=1, dash="4 4")
    s.text(560, 40, "Single Gaussian vs bimodal data", size=14, weight=600)

    base = 310
    steering_axes(s)
    bimodal_curve(s)
    fit = fn_curve(360, 760, base, 137, lambda d: gauss(d, 0, 13), -30, 30, step_px=3)
    s.curve(fit, RED, w=2, dash="7 5")

    s.text(560, 118, "peak falls between the modes,", fill=RED, size=11)
    s.text(560, 132, "where the experts never drove", fill=RED, size=11)
    s.line(560, 140, 560, 166, color=FAINT, w=1)
    for mx in (460, 660):
        s.line(mx, 88, mx, 97, color=FAINT, w=1.2)
        s.text(mx, 82, "mode", fill=MUTED, size=10)

    s.line(370, 56, 385, 56, color=BLUE, w=2)
    s.text(390, 60, "expert data (bimodal)", fill=MUTED, size=10, anchor="start")
    s.line(510, 56, 528, 56, color=RED, w=2, dash="4 3")
    s.text(533, 60, "Gaussian fit", fill=MUTED, size=10, anchor="start")

    s.text(400, 405, "The network outputs only μ and σ — a single bell curve cannot represent two modes",
           fill=MUTED, size=12)
    s.save(os.path.join(OUT, "svg4b_gaussian_policy.svg"))


# ---------------------------------------------------------------- svg5: GMM policy
def svg5():
    s = SVG(800, 420, "A two-component Gaussian mixture policy captures both modes of bimodal expert data")
    params = [(240, 118, "μ₁  σ₁  w₁", 13), (240, 178, "μ₂  σ₂  w₂", 13)]
    nn_to_params(s, params, 100, 200, "mixture", 224)
    s.text(95, 345, "K components: mean, spread,", fill=MUTED, size=11, italic=True)
    s.text(95, 360, "and weight for each", fill=MUTED, size=11, italic=True)

    s.line(310, 50, 310, 370, color=FAINT, w=1, dash="4 4")
    s.text(560, 40, "Mixture fit to bimodal data", size=14, weight=600)

    base = 310
    steering_axes(s)
    bimodal_curve(s)
    c1 = fn_curve(360, 640, base, 196, lambda d: gauss(d, -15, 5.5), -30, 12, step_px=3)
    s.curve(c1, RED, w=1.8, dash="6 4")
    c2 = fn_curve(480, 760, base, 178, lambda d: gauss(d, 15, 5.5), -12, 30, step_px=3)
    s.curve(c2, ORANGE, w=1.8, dash="6 4")
    s.text(437, 95, "component 1", fill=RED, size=10.5)
    s.text(680, 112, "component 2", fill=ORANGE, size=10.5)

    s.line(370, 56, 385, 56, color=BLUE, w=2)
    s.text(390, 60, "expert data", fill=MUTED, size=10, anchor="start")
    s.line(462, 56, 478, 56, color=RED, w=1.8, dash="4 3")
    s.text(483, 60, "mixture components", fill=MUTED, size=10, anchor="start")

    s.text(400, 405, "The network outputs a mean, spread, and weight per component — both modes are captured, up to K of them",
           fill=MUTED, size=12)
    s.save(os.path.join(OUT, "svg5_gmm_policy.svg"))


# --------------------------------------------------------- bin column helper
def bin_opacities(n, modes):
    vals = [mixture(i, modes) for i in range(n)]
    peak = max(vals)
    return [0.05 + 0.85 * v / peak for v in vals]


def bin_column(s, x, y, w, h, n, ops, color, pad=4, gap=2, stroke=BORDER):
    s.rect(x, y, w, h, "none", rx=4, stroke=stroke, sw=1.2)
    cell_h = (h - 2 * pad - (n - 1) * gap) / n
    for i, op in enumerate(ops):
        s.rect(x + pad, y + pad + i * (cell_h + gap), w - 2 * pad, cell_h,
               color, opacity=round(op, 2))


# ---------------------------------------------------------------- svg6a: transformer
def svg6a():
    s = SVG(800, 380, "A transformer policy outputs a distribution over discretized bins at every timestep")
    s.card(30, 110, 130, 105, rx=8)
    s.text(95, 148, "Transformer", size=14, weight=600)
    s.text(95, 170, "policy", fill=MUTED, size=12)
    s.text(95, 198, 'π<tspan font-size="12" dy="3">θ</tspan>', fill=MUTED, size=17)

    s.card(40, 270, 110, 36, rx=6)
    s.text(95, 293, "Robot state", fill=MUTED, size=12)
    s.arrow(95, 270, 95, 217, w=1.4)
    s.arrow(160, 160, 238, 160, w=1.4)

    cols = [(240, 6, "t = 1", "P(a₁ | s)"), (310, 5, "t = 2", "P(a₂ | s, a₁)"),
            (380, 4.5, "t = 3", "P(a₃ | s, a₁, a₂)")]
    for x, m1, lab, plab in cols:
        s.text(x + 22, 60, lab, fill=MUTED, size=11, weight=600)
        ops = bin_opacities(16, [(m1 - 1.5, 1.6, 0.55), (m1 + 6, 1.8, 0.45)])
        bin_column(s, x, 72, 44, 175, 16, ops, BLUE)
        s.text(x + 22, 261, plab, fill=MUTED, size=9.5)
    s.text(455, 165, "...", fill=FAINT, size=22)

    s.path("M240,48 C240,42 242,40 248,40 L346,40 C352,40 354,38 354,32 "
           "C354,38 356,40 362,40 L460,40 C466,40 468,42 468,48",
           stroke=FAINT, w=1.1)
    s.text(354, 26, "one distribution per timestep", fill=MUTED, size=10.5)

    s.arrow(472, 160, 566, 160, w=1.4, dash="4 3")
    s.text(519, 148, "detail", fill=MUTED, size=10)

    s.text(600, 48, "Single timestep output", size=12.5, weight=600)
    ops = bin_opacities(16, [(4.5, 1.6, 0.55), (11, 1.7, 0.45)])
    bin_column(s, 570, 62, 60, 232, 16, ops, BLUE, pad=5)
    for lab, yy in [("bin 1", 77), ("bin 15", 178), ("bin 30", 287)]:
        s.text(560, yy, lab, fill=MUTED, size=9, anchor="end")
    s.path("M635,62 C640,62 642,64 642,69 L642,165 C642,170 644,172 649,172 "
           "C644,172 642,174 642,179 L642,287 C642,292 640,294 635,294",
           stroke=FAINT, w=1.1)
    s.text(658, 172, "vocab_size = 30", fill=TEXT, size=11.5, anchor="start")
    s.text(658, 190, "(LLM vocabularies:", fill=MUTED, size=9.5, anchor="start")
    s.text(658, 202, "~30,000 tokens)", fill=MUTED, size=9.5, anchor="start")

    s.arrow_path("M275,268 C275,296 317,296 319,272", (319, 272), -80,
                 color=GREEN, w=1.3, dash="3 2")
    s.text(298, 310, "a₁ fed to t = 2", fill=MUTED, size=9)
    s.arrow_path("M345,268 C345,292 387,292 389,272", (389, 272), -80,
                 color=GREEN, w=1.3, dash="3 2")

    s.text(400, 340, "Autoregressive decoding: each timestep is a probability vector over discretized action bins,",
           fill=MUTED, size=11.5)
    s.text(400, 358, "conditioned on the state and all previously sampled actions",
           fill=MUTED, size=11.5)
    s.save(os.path.join(OUT, "svg6a_transformer_timesteps.svg"))


# ---------------------------------------------------------------- svg6b: histogram
def svg6b():
    s = SVG(800, 380, "Softmax bins form a histogram approximating a continuous bimodal distribution")
    s.text(400, 30, "Bins form a histogram — approximating a continuous distribution",
           size=15, weight=600)

    comps = [(7.5, 2.2, 0.52), (22, 2.4, 0.48)]
    n = 30
    vals = [mixture(i, comps) for i in range(n)]
    peak = max(vals)

    s.text(100, 60, "Softmax output", fill=MUTED, size=12, weight=600)
    ops = [0.04 + 0.85 * v / peak for v in vals]
    bin_column(s, 72, 72, 56, 248, n, ops, BLUE, gap=1.4)
    for lab, yy in [("1", 82), ("15", 198), ("30", 315)]:
        s.text(66, yy, lab, fill=MUTED, size=8.5, anchor="end")

    s.arrow(140, 195, 216, 195, w=1.4)
    s.text(168, 183, "each bin", fill=MUTED, size=10)
    s.text(168, 212, "= one bar", fill=MUTED, size=10)

    x0, x1, base, ytop = 220, 750, 310, 60
    s.axis_y(x0, base, ytop)
    bw = (x1 - x0) / n
    for i, v in enumerate(vals):
        h = 222 * v / peak
        s.rect(x0 + i * bw + 0.8, base - h, bw - 1.6, h, BLUE,
               opacity=round(0.25 + 0.6 * v / peak, 2))
    curve = fn_curve(x0, x1, base, 228, lambda b: mixture(b, comps), -0.5, n - 0.5,
                     step_px=3)
    s.curve(curve, GOLD, w=2, dash="6 3", opacity=0.8)
    s.axis_x(x0, x1, base, ticks=[(x0, "−30°"), ((x0 + x1) / 2, "0°"), (x1, "+30°")])
    s.text((x0 + x1) / 2, 348, "Steering angle (discretized into 30 bins)", fill=MUTED, size=11)
    s.text(209, 120, "P(bin)", fill=MUTED, size=10, rotate=-90)

    s.text(x0 + 8 * bw, 74, "mode 1", fill=MUTED, size=10)
    s.text(x0 + 22.5 * bw, 66, "mode 2", fill=MUTED, size=10)

    s.rect(320, 360, 14, 9, BLUE, opacity=0.6)
    s.text(340, 368, "bin probabilities (model output)", fill=MUTED, size=10, anchor="start")
    s.line(520, 364, 540, 364, color=GOLD, w=2, dash="4 2")
    s.text(546, 368, "true continuous distribution", fill=MUTED, size=10, anchor="start")
    s.save(os.path.join(OUT, "svg6b_bins_to_histogram.svg"))


# ---------------------------------------------------------------- svg8: autoregressive chain
def hist5(s, x, y, heights, color):
    for i, h in enumerate(heights):
        s.rect(x + 5 + i * 15, y + 70 - h, 12, h, color, rx=1,
               opacity=0.25 + 0.5 * h / max(heights))
    s.line(x + 3, y + 70, x + 79, y + 70, color=FAINT, w=1)


def svg8():
    s = SVG(800, 380, "Autoregressive action sampling: each action dimension is sampled conditioned on the previous ones")
    for x, lab in [(155, "Step 1"), (470, "Step 2")]:
        s.text(x, 30, lab, fill=MUTED, size=13, weight=600)

    # step 1
    s.card(95, 135, 120, 60, rx=8)
    s.text(155, 170, "Model", size=13, weight=600)
    s.text(155, 240, "sₜ", fill=TEXT, size=14, italic=True)
    s.arrow(155, 227, 155, 197, w=1.4)
    hist5(s, 115, 50, [18, 35, 50, 40, 20], BLUE)
    s.text(155, 44, "p(a₁ | sₜ)", fill=MUTED, size=11)
    s.arrow(155, 135, 155, 122, w=1.4)
    s.arrow(218, 95, 268, 113, color=GREEN, w=1.6)
    s.text(240, 86, "sample", fill=MUTED, size=10)
    s.text(283, 122, "a₁", fill=GREEN, size=14, anchor="start", italic=True)

    s.arrow_path("M300,132 C315,192 420,205 490,229", (492, 230), 18,
                 color=GREEN, w=1.5, dash="5 3")

    # step 2
    s.card(410, 135, 120, 60, rx=8)
    s.text(470, 170, "Model", size=13, weight=600)
    s.text(440, 240, "sₜ", fill=TEXT, size=14, italic=True)
    s.text(500, 240, "a₁", fill=GREEN, size=14, italic=True)
    s.arrow(440, 227, 449, 197, w=1.4)
    s.arrow(500, 227, 491, 197, color=GREEN, w=1.4)
    hist5(s, 430, 50, [22, 42, 32, 50, 35], BLUE)
    s.text(470, 44, "p(a₂ | sₜ, a₁)", fill=MUTED, size=11)
    s.arrow(470, 135, 470, 122, w=1.4)
    s.arrow(533, 95, 578, 113, color=GREEN, w=1.6)
    s.text(555, 86, "sample", fill=MUTED, size=10)
    s.text(593, 122, "a₂", fill=GREEN, size=14, anchor="start", italic=True)

    s.text(660, 170, "…", fill=FAINT, size=24)

    s.card(120, 290, 560, 45, rx=8, fill=CARD2)
    s.text(400, 318,
           'p(<tspan font-weight="600">a</tspan> | s)  =  p(a₁ | s) · p(a₂ | s, a₁) · …',
           size=14)
    s.text(400, 365,
           "Autoregressive factorization: each action dimension is sampled in sequence, conditioned on all previous",
           fill=MUTED, size=11.5)
    s.save(os.path.join(OUT, "svg8_autoregressive_chain.svg"))


if __name__ == "__main__":
    svg1(); svg2(); svg3(); svg4(); svg4b(); svg5(); svg6a(); svg6b(); svg8()
