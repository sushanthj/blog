"""Regenerates the RL theory basics 2 illustrations and the VAE/ELBO appendix
figure in the site theme. Run: python3 _scripts/illustrations/rl_basics_2_appendix.py

Follows the formality rules in .claude/skills/blog-illustrations/SKILL.md.
"""
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from svgkit import (SVG, BG, CARD, CARD2, BORDER, TEXT, MUTED, FAINT, BLUE, GREEN,
                    RED, GOLD, ORANGE, PURPLE, TEAL, gauss, mixture, fn_curve)

RL2 = os.path.join(os.path.dirname(__file__), "..", "..",
                   "images", "blog", "reinforcement-learning", "RL_theory_basics_2")
APP = os.path.join(os.path.dirname(__file__), "..", "..",
                   "images", "blog", "appendix", "VAE_and_ELBO")


# ------------------------------------------------- svg3: REINFORCE vs IL weights
def svg3():
    s = SVG(800, 380, "Imitation learning weights every action equally; REINFORCE scales the update by reward and reverses it for negative returns")
    s.text(200, 30, "Imitation learning", size=14, weight=600)
    s.text(200, 50, "every expert action, equal weight", fill=MUTED, size=11.5)

    for i in range(5):
        y = 72 + i * 46
        s.card(55, y, 130, 32, rx=6)
        s.text(120, y + 21, f"a{chr(0x2081+i)} (expert)", fill=MUTED, size=11)
        s.arrow(185, y + 16, 310, y + 16, color=GREEN, w=2)
    s.text(325, 185, "equal", fill=MUTED, size=10.5, anchor="start")
    s.text(325, 198, "weight", fill=MUTED, size=10.5, anchor="start")
    s.text(200, 320, "∇ log π(a | s)", fill=TEXT, size=12)
    s.text(200, 340, "all update vectors have the same length", fill=MUTED, size=10.5)

    s.line(400, 20, 400, 360, color=FAINT, w=1, dash="6 4")

    s.text(600, 30, "REINFORCE", size=14, weight=600)
    s.text(600, 50, "weight each trajectory by its return", fill=MUTED, size=11.5)

    rows = [("τ₁   R = +8.2", GREEN, 165, 3.5, None),
            ("τ₂   R = +3.1", GREEN, 95, 2.5, None),
            ("τ₃   R = +0.5", GREEN, 40, 1.6, None),
            ("τ₄   R = −2.4", RED, 95, 2.5, "reversed"),
            ("τ₅   R = −6.7", RED, 155, 3.5, "reversed")]
    for i, (lab, col, ln, w, note) in enumerate(rows):
        y = 72 + i * 46
        s.card(435, y, 130, 32, rx=6)
        s.text(500, y + 21, lab, fill=TEXT, size=11)
        s.arrow(565, y + 16, 565 + ln, y + 16, color=col, w=w)
        if note:
            s.text(575 + ln, y + 20, note, fill=MUTED, size=9, anchor="start", italic=True)
    s.text(600, 320, "∇ log π(a | s) · R(τ)", fill=TEXT, size=12)
    s.text(600, 340, "update length proportional to |R|; sign follows the return", fill=MUTED, size=10.5)
    s.save(os.path.join(RL2, "svg3_reinforce_vs_il.svg"))


# ------------------------------------------------- svg4: sampled trajectories
def svg4():
    s = SVG(800, 400, "Five sampled locomotion trajectories: the only positive-return sample is falling forward, so the gradient reinforces it")
    s.text(400, 28, "Five sampled trajectories — learning to walk", size=15, weight=600)
    s.text(400, 48, "r(s, a) = forward velocity", fill=MUTED, size=11.5)

    # start-state marker
    s.circle(150, 168, 5, NODE_FILL := "#141721", stroke=TEXT, sw=1.6)
    s.text(150, 150, "s₀", fill=TEXT, size=13, italic=True)

    s.line(90, 215, 580, 215, color=FAINT, w=1, dash="4 3")
    s.text(390, 228, "forward displacement →", fill=FAINT, size=10)

    s.path("M150,170 C148,175 140,190 132,210", stroke=RED, w=1.8, opacity=0.8)
    s.text(152, 240, "falls backward", fill=MUTED, size=10)
    s.text(152, 253, "R &lt; 0", fill=RED, size=10)

    s.path("M150,173 C120,183 105,200 100,210", stroke=RED, w=1.6, opacity=0.6)
    s.text(64, 240, "back, then forward", fill=MUTED, size=10)
    s.text(64, 253, "R &lt; 0", fill=RED, size=10)

    s.path("M150,165 C156,162 160,163 163,165", stroke=GOLD, w=1.8, opacity=0.8)
    s.text(170, 192, "stands still · R = 0", fill=MUTED, size=10, anchor="start")

    s.path("M150,168 C220,150 280,155 300,180 C315,200 305,210 295,215",
           stroke=RED, w=1.6, opacity=0.7)
    s.text(300, 240, "steps forward, falls", fill=MUTED, size=10)
    s.text(300, 253, "R &lt; 0", fill=RED, size=10)

    s.path("M150,162 C250,135 380,140 450,170 C480,185 500,205 510,215",
           stroke=GREEN, w=2.2)
    s.text(512, 240, "falls forward", fill=MUTED, size=10)
    s.text(512, 253, "R &gt; 0 — highest return in the batch", fill=GREEN, size=10)

    s.card(470, 78, 295, 78, rx=8)
    s.text(618, 102, "Gradient update", size=12.5, weight=600)
    s.text(618, 122, "increase the probability of the", fill=MUTED, size=11)
    s.text(618, 137, "highest-return sample — falling forward", fill=MUTED, size=11)

    s.text(400, 310, "With few samples the gradient estimate is noisy: the policy learns to fall forward, not to walk",
           fill=MUTED, size=12)
    s.text(400, 330, "the high-variance problem of policy gradients", fill=FAINT, size=11, italic=True)
    s.save(os.path.join(RL2, "svg4_humanoid_trajectories.svg"))


# ------------------------------------------------- svg5: baseline
def svg5():
    s = SVG(800, 450, "Without a baseline every non-negative reward pushes probabilities up; subtracting the average pushes below-average actions down")
    s.text(400, 28, "The baseline — relative rather than absolute reward", size=15, weight=600)

    bars = [("no-op", 0.0), ("sleeves", 0.5), ("flatten", 0.0), ("fold", 1.0)]

    s.text(200, 60, "Without baseline", size=13, weight=600)
    s.text(200, 78, "all rewards ≥ 0: every action is pushed up", fill=MUTED, size=11)
    s.axis_y(60, 310, 100)
    s.line(60, 310, 340, 310, color=MUTED, w=1.4)
    for i, (lab, r) in enumerate(bars):
        x = 78 + i * 65
        h = max(4, 130 * r)
        col = GREEN if r > 0 else FAINT
        s.rect(x, 310 - h, 48, h, col, rx=2, opacity=0.5 if r > 0 else 0.4)
        s.text(x + 24, 310 - h - 8, "↑", fill=col, size=13)
        s.text(x + 24, 328, lab, fill=MUTED, size=9.5)
        s.text(x + 24, 341, f"r = {r:g}", fill=MUTED, size=9.5)
    s.text(28, 205, "Δ probability", fill=MUTED, size=10, rotate=-90)
    s.text(200, 368, "every trajectory is reinforced,", fill=MUTED, size=11)
    s.text(200, 383, "including the no-op ones", fill=RED, size=11)

    s.line(400, 55, 400, 420, color=FAINT, w=1, dash="6 4")

    s.text(600, 60, "With baseline  (b = average reward)", size=13, weight=600)
    s.text(600, 78, "better than average up, worse than average down", fill=MUTED, size=11)
    s.axis_y(460, 310, 100)
    s.line(460, 310, 740, 310, color=MUTED, w=1.4)
    zero = 220
    s.line(460, zero, 740, zero, color=GOLD, w=1.4, dash="6 3")
    s.text(746, zero + 4, "b = 0.375", fill=GOLD, size=10, anchor="start")
    for i, (lab, r) in enumerate(bars):
        x = 478 + i * 65
        adv = r - 0.375
        h = abs(adv) * 128
        col = GREEN if adv > 0 else RED
        y = zero - h if adv > 0 else zero
        s.rect(x, y, 48, h, col, rx=2, opacity=0.45)
        glyph, gy = ("↑", y - 7) if adv > 0 else ("↓", y + h + 15)
        s.text(x + 24, gy, glyph, fill=col, size=13)
        s.text(x + 24, 328, lab, fill=MUTED, size=9.5)
        s.text(x + 24, 341, ("r − b &gt; 0" if adv > 0 else "r − b &lt; 0"), fill=MUTED, size=9.5)
    s.text(428, 205, "Δ probability", fill=MUTED, size=10, rotate=-90)
    s.text(600, 368, "below-average actions are now", fill=MUTED, size=11)
    s.text(600, 383, "actively suppressed", fill=GREEN, size=11)

    s.text(400, 418, "Subtracting a baseline reduces variance without introducing bias:",
           fill=MUTED, size=12)
    s.text(400, 436, "E[∇ log p(τ) · b] = 0 — the baseline vanishes in expectation",
           fill=FAINT, size=11)
    s.save(os.path.join(RL2, "svg5_baseline.svg"))


# ------------------------------------------------- svg8: importance weights
def svg8():
    s = SVG(750, 330, "Importance sampling reweights actions drawn from the old policy by the probability ratio of new to old")
    cx, base = 375, 230

    s.line(75, base, 675, base, color=MUTED, w=1.2)
    s.text(375, base + 25, "action", fill=MUTED, size=11)

    old = fn_curve(95, 655, base, 130, lambda v: gauss(v, -45, 88), -280, 280, step_px=3)
    s.curve(old, ORANGE, w=2, fill_opacity=0.07, base=base)
    new = fn_curve(95, 655, base, 130, lambda v: gauss(v, 100, 92), -280, 280, step_px=3)
    s.curve(new, BLUE, w=2, fill_opacity=0.07, base=base)
    s.text(235, 82, 'π<tspan font-size="9" dy="3">θ</tspan><tspan dy="-3"> (old)</tspan>',
           fill=ORANGE, size=12)
    s.text(545, 68, 'π<tspan font-size="9" dy="3">θ′</tspan><tspan dy="-3"> (new)</tspan>',
           fill=BLUE, size=12)

    samples = [(-120, 0.3), (-60, 0.6), (-30, 0.9), (0, 1.1), (20, 1.5), (50, 2.3), (130, 3.8)]
    for off, ratio in samples:
        x = cx + off
        col = RED if ratio < 0.8 else (MUTED if ratio < 1.3 else GREEN)
        h = 10 + ratio * 11
        s.circle(x, base, 3.5, GOLD, stroke=BG, sw=1, opacity=0.9)
        s.line(x, base - 5, x, base - h, color=col, w=1.8)
        s.text(x, base - h - 6, f"{ratio}×", fill=col, size=9)

    s.card(50, 265, 650, 50, fill=CARD2, rx=6)
    s.text(80, 285, "ratio &gt; 1", fill=GREEN, size=11, anchor="start")
    s.text(160, 285, "the new policy takes this action more often — the sample is upweighted",
           fill=MUTED, size=11, anchor="start")
    s.text(80, 305, "ratio &lt; 1", fill=RED, size=11, anchor="start")
    s.text(160, 305, "the new policy takes this action less often — the sample is downweighted",
           fill=MUTED, size=11, anchor="start")
    s.save(os.path.join(RL2, "svg8_importance_weights.svg"))


# ------------------------------------------------- svg13: marginal intractable
def svg13():
    s = SVG(900, 380, "A discrete latent gives a three-term sum for p(x); a continuous latent gives an intractable integral over R^d")
    s.line(450, 50, 450, 350, color=FAINT, w=1, dash="4 4")
    s.text(225, 35, "Discrete latent — tractable", size=15, weight=600)
    s.text(675, 35, "Continuous latent — intractable", size=15, weight=600)

    base = 280
    for x, p, lab in [(90, 0.5, "z_A"), (195, 0.3, "z_B"), (300, 0.2, "z_C")]:
        h = p * 200
        s.rect(x, base - h, 60, h, GREEN, opacity=0.25 + 0.6 * p)
        s.text(x + 30, base - h - 10, f"p(z = {lab[-1]}) = {p}", fill=MUTED, size=12)
        s.text(x + 30, base + 22, lab, fill=TEXT, size=13, italic=True)
    s.line(80, base, 380, base, color=MUTED, w=1.4)
    s.text(225, 335, "p(x) = Σ_z p(x | z) · p(z) — three terms, a weighted sum",
           fill=TEXT, size=12.5)

    pts = fn_curve(490, 850, base, 115, lambda v: gauss(v, 0, 55), -180, 180, step_px=3)
    s.curve(pts, PURPLE, w=2, fill_opacity=0.06, base=base)
    s.text(670, 150, "p(z) = 𝒩(0, I) over ℝ^d", fill=PURPLE, size=12.5)
    s.rect(654, 222, 32, base - 222, GOLD, opacity=0.25)
    s.text(670, 212, "the only region where", fill=MUTED, size=10.5)
    s.text(670, base + 30, "p(x | z) is non-negligible", fill=GOLD, size=10.5)
    s.line(490, base, 850, base, color=MUTED, w=1.4)
    s.text(486, base + 15, "−∞", fill=MUTED, size=11, anchor="end")
    s.text(854, base + 15, "+∞", fill=MUTED, size=11, anchor="start")

    s.text(675, 335, "p(x) = ∫ p(x | z) · p(z) dz — an integral over all of ℝ^d",
           fill=TEXT, size=12.5)
    s.text(675, 357, "no closed form; a grid needs ~10³² points; prior samples miss the spike",
           fill=MUTED, size=11.5)
    s.save(os.path.join(APP, "svg13_marginal_intractable.svg"))


if __name__ == "__main__":
    svg3(); svg4(); svg5(); svg8(); svg13()
