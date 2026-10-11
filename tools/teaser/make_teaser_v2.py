"""Short motion-graphics teaser for Agima (1280x720, 30 fps), drawn at 2x and downsampled.

usage: make_teaser_v2.py [out.mp4] [stills <dir>]

Reuses the data and drawing helpers of make_teaser.py (same inputs). Sokoban boards are re-rendered
with sprites at fractional positions so the agent and boxes slide between cells.
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_argv, sys.argv = sys.argv, sys.argv[:1]
import make_teaser as T   # noqa: E402  (loads shared data; must not see our argv)
sys.argv = _argv

from make_teaser import (DEAD, EP34, FIG5, K5, SITE, SPR, Hi, gif_frames, img, inter, rounded_path)  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "x" else f"{SITE}/video/agima_teaser_v2.mp4"
W, H, FPS, Z = 1280, 720, 30, 2
BG = (247, 245, 240)
INK, MUTED, FAINT = (22, 22, 24), (118, 116, 110), (196, 193, 185)
ACCENT, RED, GREEN = (38, 96, 224), (212, 76, 56), (18, 146, 108)
LINE = (190, 187, 178)


# ------------------------------------------------------------------ easing
def clamp(x):
    return max(0.0, min(1.0, x))


def ease_out(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x); return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def back_out(x, k=1.9):
    x = clamp(x); return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def prog(t, t0, dur):
    return clamp((t - t0) / dur)


def mix(a, b, s):
    return tuple(int(u + (v - u) * s) for u, v in zip(a, b))


def fade(col, a):
    return mix(BG, col, a)


# ------------------------------------------------------------------ canvas
def _background():
    yy, xx = np.mgrid[0:H * Z, 0:W * Z].astype(np.float32)
    r = np.hypot((xx - W * Z / 2) / (W * Z / 2), (yy - H * Z / 2) / (H * Z / 2))
    v = np.clip((r - 0.55) / 0.9, 0, 1)[..., None] ** 1.6
    edge = np.array((232, 229, 221), np.float32)
    return Image.fromarray((np.array(BG, np.float32) * (1 - v) + edge * v).astype(np.uint8))


BG2 = _background()
_SHADOW = {}


class V(Hi):
    def __init__(self):
        self.c = BG2.copy()
        self.d = ImageDraw.Draw(self.c)

    def bold(self, size):
        return inter(size * Z, b"Bold")

    def card(self, im, cx, cy, size, a=1.0, scale=1.0, r=12, nearest=False, outline=None, dashed=False, sx=1.0):
        """Square image centered at (cx, cy) with rounded corners and a soft shadow."""
        if a <= 0 or scale <= 0 or sx <= 0.01:
            return
        w = max(2, int(size * scale * sx * Z)); h = max(2, int(size * scale * Z))
        im = im.resize((w, h), Image.NEAREST if nearest else Image.LANCZOS)
        key = (w, h, r)
        if key not in _SHADOW:
            pad = 18 * Z
            sh = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
            ImageDraw.Draw(sh).rounded_rectangle([pad, pad + 5 * Z, pad + w, pad + h + 5 * Z], radius=r * Z, fill=70)
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=min(r * Z, w // 2), fill=255)
            _SHADOW[key] = (sh.filter(ImageFilter.GaussianBlur(9 * Z)), m, pad)
        sh, m, pad = _SHADOW[key]
        if a < 1:
            sh = sh.point(lambda v: int(v * a)); m = m.point(lambda v: int(v * a))
        X, Y = int(cx * Z - w / 2), int(cy * Z - h / 2)
        self.c.paste(Image.new("RGB", sh.size, (60, 52, 40)), (X - pad, Y - pad), sh)
        self.c.paste(im, (X, Y), m)
        if outline:
            box = (cx - size * scale * sx / 2 - 5, cy - size * scale / 2 - 5, cx + size * scale * sx / 2 + 5, cy + size * scale / 2 + 5)
            col = fade(outline, a)
            if dashed:
                self.dashed_rect(box, r + 4, col)
            else:
                self.rrect(box, r + 4, outline=col, width=3)

    def dashed_rect(self, box, r, col, on=7, off=6):
        x0, y0, x1, y1 = box
        pts = rounded_path([(x0 + (x1 - x0) / 2, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0), (x0 + (x1 - x0) / 2 + 1, y0)], r)
        self.line(pts, col, 2, dash=(on, off))

    def pline(self, pts, frac, col, width=2, dash=None):
        """Draw the first `frac` of a polyline (by length)."""
        if frac <= 0:
            return
        pts = np.array(pts, float)
        seg = np.hypot(*np.diff(pts, axis=0).T); cum = np.concatenate([[0], np.cumsum(seg)])
        L = cum[-1] * clamp(frac)
        k = int(np.searchsorted(cum, L, side="right"))
        out = [tuple(p) for p in pts[:k]]
        if k < len(pts):
            s = (L - cum[k - 1]) / max(seg[k - 1], 1e-6)
            out.append(tuple(pts[k - 1] + (pts[k] - pts[k - 1]) * s))
        if len(out) >= 2:
            self.line(out, col, width, dash=dash)

    def stamp(self, cx, cy, ok, p, r=20):
        """Check / cross in a filled circle that pops in."""
        if p <= 0:
            return
        s = back_out(p)
        self.mark(cx, cy, ok, GREEN if ok else RED, r=max(1, r * s))

    def ctext(self, xy, s, font, col, anchor="mm"):
        self.text(xy, s, font, col, anchor)

    def headline(self, t, parts, x=70, y=70, size=46, t0=0.15, sub=None):
        """Words rise and fade in one after another. parts: [(text, color)]."""
        f = self.bold(size)
        words = [(w, col) for txt, col in parts for w in txt.split(" ") if w]
        cx = x
        for i, (w, col) in enumerate(words):
            p = ease_out(prog(t, t0 + i * 0.07, 0.45))
            self.text((cx, y + 14 * (1 - p)), w, f, fade(col, p), "ls")
            cx += self.tlen(w + " ", f)
        if sub:
            p = ease_out(prog(t, t0 + len(words) * 0.07 + 0.1, 0.5))
            self.text((x, y + 40 + 8 * (1 - p)), sub, inter(21 * Z), fade(MUTED, p), "ls")

    def out(self):
        return self.c.resize((W, H), Image.LANCZOS)


def bezier(p0, p1, n=40, bend=0.55):
    (x0, y0), (x1, y1) = p0, p1
    xa, xb = x0 + (x1 - x0) * bend, x1 - (x1 - x0) * bend
    return [((1 - s) ** 3 * x0 + 3 * (1 - s) ** 2 * s * xa + 3 * (1 - s) * s * s * xb + s ** 3 * x1,
             (1 - s) ** 3 * y0 + 3 * (1 - s) ** 2 * s * y0 + 3 * (1 - s) * s * s * y1 + s ** 3 * y1) for s in np.linspace(0, 1, n)]


# ------------------------------------------------------------------ smooth Sokoban
TILE = 32


class Board:
    """A Sokoban level whose moves can be rendered mid-slide."""

    def __init__(self, S):
        self.S = S
        base = S.im.copy()
        for r, c in S.floor:
            base.paste(SPR["goal" if (r, c) in S.goals else "floor"], (c * TILE, r * TILE))
        self.base = base
        pa = np.asarray(SPR["player"]); self.pmask = Image.fromarray(((pa.sum(2) > 60) * 255).astype(np.uint8))

    def states(self, moves, start=None):
        s = start or (self.S.player, self.S.boxes)
        out = [s]
        for m in moves:
            s = self.S.step(*s, m); out.append(s)
        return out

    def draw(self, s0, s1=None, f=0.0):
        """State s0 moving toward s1 by fraction f (0..1)."""
        s1 = s1 or s0
        (p0, b0), (p1, b1) = s0, s1
        im = self.base.copy()
        lerp = lambda a, b: (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        moved = list(b0 - b1); new = list(b1 - b0)
        for b in b0 & b1:
            im.paste(SPR["box_goal" if b in self.S.goals else "box"], (b[1] * TILE, b[0] * TILE))
        if moved:
            r, c = lerp(moved[0], new[0])
            on = f > 0.5 and new[0] in self.S.goals
            im.paste(SPR["box_goal" if on else "box"], (int(c * TILE), int(r * TILE)))
        r, c = lerp(p0, p1)
        im.paste(SPR["player"], (int(c * TILE), int(r * TILE)), self.pmask)
        return im

    def at(self, states, t, step):
        """Board image at time t when each move takes `step` seconds."""
        if t <= 0:
            return self.draw(states[0])
        k = int(t / step)
        if k >= len(states) - 1:
            return self.draw(states[-1])
        return self.draw(states[k], states[k + 1], ease_io((t - k * step) / step))


BOARD = Board(DEAD)
TEMPT = BOARD.states(["down", "left"])
SOLVE_MOVES = DEAD.solve()
SOLVE = BOARD.states(SOLVE_MOVES)
STUCK_BOX = list(TEMPT[-1][1] - TEMPT[0][1])[0]   # the box that gets cornered


# ------------------------------------------------------------------ other data
def _fig5_after_rl():
    """The 3 x 4 board thumbnails of Fig. 5(b), after RL: two failed plans, then a successful one."""
    rows, xs, half = (638, 789, 955), (815, 1069, 1322, 1575), 63
    out = []
    for y in rows:
        out.append([FIG5.crop((x + 66 - half, y + 66 - half, x + 66 + half, y + 66 + half)) for x in xs])
    return out


RL_ROWS = _fig5_after_rl()
PAC_LIN = [img(p) for p in sorted(glob.glob(f"{EP34}/vanilla/ground_truth/*.png"))]
PAC_FWD = [img(p) for p in sorted(glob.glob(f"{EP34}/ground_truth/candidate_A_far_food_up_succeeds/*.png"))]
ENVS = [("Maze2D", "maze2d", "small mazes", "larger mazes"), ("Sokoban", "sokoban", "two boxes", "four boxes"),
        ("Pacman", "pacman", "9×9, 8 food", "12×12, 12 food"), ("PushT", "pusht", "near starts", "far starts")]
ENV = {k: [[f.crop((0, 0, 256, 256)) for f in gif_frames(f"{SITE}/gifs/env/{k}_{s}.gif")] for s in ("train", "test")]
       for _, k, _, _ in ENVS}


# ------------------------------------------------------------------ scenes
def sc_hook(v, t, dur):
    v.headline(t, [("One push can make a puzzle", INK), ("unsolvable.", RED)])
    cx, cy, size = W / 2, 400, 380
    shake = 7 * np.sin(t * 60) * np.exp(-6 * max(0, t - 2.05)) if t > 2.05 else 0
    board = BOARD.at(TEMPT, t - 1.0, 0.45)
    v.card(board, cx + shake, cy, size, ease_out(prog(t, 0.2, 0.6)), 0.94 + 0.06 * ease_out(prog(t, 0.2, 0.6)), nearest=True)
    p = prog(t, 2.0, 0.5)
    if p > 0:   # red ring on the cornered box, then the verdict
        bx = cx - size / 2 + (STUCK_BOX[1] + 0.5) * size / 8 + shake
        by = cy - size / 2 + (STUCK_BOX[0] + 0.5) * size / 8
        rr = 34 + 40 * (1 - ease_out(p))
        v.d.ellipse(v.z(bx - rr, by - rr, bx + rr, by + rr), outline=fade(RED, ease_out(p)), width=4 * Z)
        v.pill(cx + size / 2 + 26, cy, "Deadlock", RED, False, ease_out(prog(t, 2.2, 0.4)), size=20)
        q = ease_out(prog(t, 2.6, 0.5))
        v.ctext((W / 2, 648 + 8 * (1 - q)), "Real feedback arrives only after acting, when it is already too late.",
                inter(28 * Z, b"SemiBold"), fade(INK, q))


IM_A = [(TEMPT[1], "down"), (TEMPT[2], "left")]
_pick = [4, 8, 12, 15, len(SOLVE) - 1]
IM_B = [(SOLVE[k], SOLVE_MOVES[k - 1]) for k in _pick]
COMMIT_T, EXEC_STEP = 5.0, 0.11


def _imagined(state):
    im = BOARD.draw(state)
    return Image.blend(im, Image.new("RGB", im.size, (205, 220, 255)), 0.12)


def sc_imagine(v, t, dur):
    v.headline(t, [("Agima", INK), ("imagines", ACCENT), ("before it acts.", INK)],
               sub="It generates the boards its actions lead to, and plans against them.")
    bx, by, bs = 190, 418, 230
    commit = ease_io(prog(t, COMMIT_T, 0.6))
    # the real board: still while imagining, then executes the committed plan
    real = BOARD.at(SOLVE, t - COMMIT_T - 0.6, EXEC_STEP)
    done = t > COMMIT_T + 0.6 + EXEC_STEP * len(SOLVE_MOVES)
    v.card(real, bx, by, bs, ease_out(prog(t, 0.2, 0.5)), nearest=True,
           outline=GREEN if done else None)
    v.ctext((bx, by + bs / 2 + 26), "real board", inter(16 * Z, b"SemiBold"), MUTED)
    lanes = [(IM_A, 300, False, 0.9, "Deadlock"), (IM_B, 535, True, 1.9, "Solved")]
    x0, pitch, ts = 400, 150, 112
    for frames, ly, ok, t0, verdict in lanes:
        dim = 1 - 0.75 * commit if not ok else 1.0
        p_line = ease_io(prog(t, t0, 0.5))
        v.pline(bezier((bx + bs / 2 + 10, by), (x0 - ts / 2 - 14, ly)), p_line, fade(ACCENT, 0.55 * dim), 2, dash=(7, 6))
        for i, (state, mv) in enumerate(frames):
            ti = t0 + 0.35 + i * 0.42
            p = ease_out(prog(t, ti, 0.45))
            x = x0 + i * pitch
            if i:
                v.pline([(x - pitch + ts / 2 + 12, ly), (x - ts / 2 - 12, ly)], ease_io(prog(t, ti - 0.15, 0.3)),
                        fade(ACCENT, 0.55 * dim), 2, dash=(7, 6))
            v.card(_imagined(state), x, ly, ts, p * dim, 0.8 + 0.2 * back_out(prog(t, ti, 0.45)), nearest=True,
                   outline=ACCENT, dashed=True)
        xe = x0 + (len(frames) - 1) * pitch + ts / 2 + 36
        te = t0 + 0.35 + len(frames) * 0.42
        v.stamp(xe, ly, ok, prog(t, te, 0.35) * dim, r=19)
        q = ease_out(prog(t, te + 0.1, 0.4)) * dim
        v.text((xe + 30, ly), verdict, inter(20 * Z, b"SemiBold"), fade(GREEN if ok else RED, q), "lm")
        v.text((x0 - ts / 2, ly - ts / 2 - 18), "imagined", inter(15 * Z, b"SemiBold"), fade(ACCENT, ease_out(prog(t, t0 + 0.3, 0.4)) * dim), "lm")
    if commit > 0:
        v.pill(W / 2 + 40, 668, "Commit the plan that works, then act", GREEN, True, commit, size=19)
    if done:
        v.stamp(bx + bs / 2 - 4, by - bs / 2 + 4, True, prog(t, COMMIT_T + 0.6 + EXEC_STEP * len(SOLVE_MOVES), 0.35), r=17)


def _diagram(v, t, t0, kind, cx, cy):
    """Abstract node-link sketch of one imagination format, drawn on progressively."""
    n, gap, s = 4, 56, 30
    x0 = cx - (n - 1) * gap / 2 - 20
    col = ACCENT
    nodes = [(x0, cy)] + [(x0 + 40 + i * gap, cy) for i in range(n)]

    def node(x, y, p, c=col, real=False):
        if p <= 0:
            return
        r = s / 2 * back_out(p)
        v.rrect((x - r, y - r, x + r, y + r), 7, fill=WHITE if not real else c, outline=fade(c, p), width=3)

    lines = []
    if kind == "linear":
        for i in range(1, len(nodes)):
            lines.append(([nodes[i - 1], nodes[i]], i, col))
    elif kind == "branch":
        for i in range(1, len(nodes)):
            lines.append(([nodes[i - 1], nodes[i]], i, col))
        lines.append(([nodes[2], (nodes[2][0] + 30, cy - 62)], 2.6, RED))
    else:
        top = [(x0 + 40 + i * gap, cy - 44) for i in range(n)]
        bot = [(x0 + 40 + i * gap, cy + 44) for i in range(n - 1)]
        for row, c in ((top, GREEN), (bot, RED)):
            for i, p in enumerate(row):
                lines.append(([nodes[0] if i == 0 else row[i - 1], p], i + 1, col))
        nodes = [nodes[0]] + top + bot
    for pts, k, c in lines:
        v.pline(bezier(pts[0], pts[1], 20, 0.5) if pts[0][1] != pts[1][1] else pts, ease_io(prog(t, t0 + 0.12 * k, 0.25)), fade(c, 0.7), 2)
    for i, (x, y) in enumerate(nodes):
        k = 0 if i == 0 else (i if kind != "forward" else 1 + (i - 1) % n if i <= n else 1 + (i - 1 - n))
        node(x, y, prog(t, t0 + 0.12 * k + 0.1, 0.3), col, real=i == 0)
    if kind == "branch":
        bx, by = nodes[2][0] + 30, cy - 62
        node(bx, by, prog(t, t0 + 0.12 * 2.6 + 0.2, 0.3), RED)
        v.stamp(bx + 22, by - 14, False, prog(t, t0 + 0.6, 0.3), r=10)
    if kind == "forward":
        v.stamp(x0 + 40 + (n - 1) * gap + 30, cy - 44, True, prog(t, t0 + 0.75, 0.3), r=10)
        v.stamp(x0 + 40 + (n - 2) * gap + 30, cy + 44, False, prog(t, t0 + 0.75, 0.3), r=10)


WHITE = (255, 255, 255)


def sc_formats(v, t, dur):
    v.headline(t, [("Three ways to", INK), ("imagine.", ACCENT)],
               sub="One model, trained on all three: it chooses actions, then generates the boards they lead to.")
    cols = [("Linear rollout", "follow one plan", "linear"), ("Counterfactual branching", "test a risky move", "branch"),
            ("Forward thinking", "compare whole plans", "forward")]
    for i, (name, desc, kind) in enumerate(cols):
        cx = 245 + i * 395; t0 = 0.6 + i * 0.45
        p = ease_out(prog(t, t0, 0.5))
        v.rrect((cx - 175, 210 + 12 * (1 - p), cx + 175, 590 + 12 * (1 - p)), 22, fill=fade(WHITE, p))
        _diagram(v, t, t0 + 0.25, kind, cx, 400 + 12 * (1 - p))
        v.ctext((cx, 258 + 12 * (1 - p)), name, inter(21 * Z, b"Bold"), fade(INK, p))
        v.ctext((cx, 550 + 12 * (1 - p)), desc, inter(17 * Z), fade(MUTED, p))
    q = ease_out(prog(t, 2.6, 0.5))
    v.rrect((420, 640, 434, 654), 7, fill=fade(ACCENT, q)); v.text((444, 647), "real board", inter(15 * Z), fade(MUTED, q), "lm")
    v.rrect((560, 640, 574, 654), 5, fill=fade(WHITE, q), outline=fade(ACCENT, q), width=3)
    v.text((584, 647), "imagined board", inter(15 * Z), fade(MUTED, q), "lm")


def sc_general(v, t, dur):
    v.headline(t, [("Trained on", INK), ("easy", MUTED), ("puzzles. Tested on", INK), ("hard", ACCENT), ("ones.", INK)])
    size, gap = 236, 60
    x0 = (W - 4 * size - 3 * gap) / 2 + size / 2
    for i, (name, key, tr, te) in enumerate(ENVS):
        cx = x0 + i * (size + gap); cy = 380
        trf, tef = ENV[key]
        t_flip = 1.7 + i * 0.13
        f = prog(t, t_flip, 0.5)
        sx = abs(np.cos(np.pi * f)) if 0 < f < 1 else 1.0
        test = f >= 0.5
        frames = tef if test else trf
        k = int(max(0, t - (t_flip + 0.25 if test else 0)) / 0.2) % len(frames)
        p = ease_out(prog(t, 0.3 + i * 0.1, 0.5))
        v.card(frames[k], cx, cy + 16 * (1 - p), size, p, sx=max(sx, 0.02), outline=ACCENT if test else None)
        v.ctext((cx, cy - size / 2 - 30), name, inter(22 * Z, b"Bold"), fade(INK, p))
        lab, desc, col = ("TEST", te, ACCENT) if test else ("TRAIN", tr, MUTED)
        q = p if not test else ease_out(prog(t, t_flip + 0.25, 0.3))
        fb = inter(15 * Z, b"Bold"); fr = inter(17 * Z)
        w = v.tlen(lab, fb) + 10 + v.tlen(desc, fr)
        v.text((cx - w / 2, cy + size / 2 + 32), lab, fb, fade(col, q), "lm")
        v.text((cx - w / 2 + v.tlen(lab, fb) + 10, cy + size / 2 + 32), desc, fr, fade(INK, q), "lm")


RES_ENVS = ["Pacman", "Maze2D", "Sokoban", "PushT"]
AGIMA = next(v for n, k, v in K5 if k == "ours")
BEST_BASE = [max(r[2][i] for r in K5 if r[1] != "ours") for i in range(4)]


def sc_results(v, t, dur):
    v.headline(t, [("Agima", ACCENT), ("beats the best baselines.", INK)],
               sub="Success on the hard test sets at K = 5: a real board only every 5 actions.")
    base, top, maxv = 590, 240, 45.0
    gw, bw = 250, 70
    x0 = (W - 4 * gw) / 2
    v.line([(x0, base), (x0 + 4 * gw, base)], FAINT, 2)
    for i, env in enumerate(RES_ENVS):
        gx = x0 + i * gw + gw / 2
        for j, (val, col, lab) in enumerate([(BEST_BASE[i], FAINT, "best baseline"), (AGIMA[i], ACCENT, "Agima")]):
            p = ease_out(prog(t, 0.7 + i * 0.12 + j * 0.15, 0.9))
            x = gx - bw - 6 if j == 0 else gx + 6
            hgt = (base - top) * val / maxv * p
            if hgt > 2:
                v.rrect((x, base - hgt, x + bw, base), min(8, hgt / 2 - 0.5), fill=col, corners=(True, True, False, False))
            n = int(round(val * p))
            v.ctext((x + bw / 2, base - hgt - 20), f"{n}%", inter((24 if j else 18) * Z, b"Bold"), fade(ACCENT if j else MUTED, ease_out(prog(t, 0.7, 0.3))))
        v.ctext((gx, base + 30), env, inter(20 * Z, b"SemiBold"), fade(INK, ease_out(prog(t, 0.5, 0.4))))
    q = ease_out(prog(t, 1.4, 0.4))
    v.rrect((W / 2 - 230, 662, W / 2 - 214, 678), 4, fill=fade(ACCENT, q)); v.text((W / 2 - 206, 670), "Agima", inter(16 * Z), fade(INK, q), "lm")
    v.rrect((W / 2 - 110, 662, W / 2 - 94, 678), 4, fill=fade(FAINT, q))
    v.text((W / 2 - 86, 670), "best baseline (GPT-5.6, Gemini 3.1, fine-tuned Qwen)", inter(16 * Z), fade(INK, q), "lm")


def sc_rl(v, t, dur):
    v.headline(t, [("RL discovers an", INK), ("emergent", ACCENT), ("strategy.", INK)],
               sub="Draft a plan, imagine it, and try another only when it fails.")
    ts, pitch, x0 = 84, 104, 250
    rows = [(260, "Plan 1", False), (390, "Plan 2", False), (520, "Plan 3", True)]
    for r, (y, lab, ok) in enumerate(rows):
        t0 = 0.7 + r * 0.95
        p = ease_out(prog(t, t0, 0.4))
        v.text((90, y), lab, inter(20 * Z, b"Bold"), fade(INK, p), "lm")
        if r:
            v.pline([(120, y - 130 + 26), (120, y - 22)], ease_io(prog(t, t0 - 0.2, 0.3)), fade(LINE, p), 2, dash=(5, 5))
        for i, im in enumerate(RL_ROWS[r]):
            ti = t0 + 0.1 + i * 0.12
            q = ease_out(prog(t, ti, 0.35))
            if i:
                v.pline([(x0 + (i - 1) * pitch + ts / 2 + 6, y), (x0 + i * pitch - ts / 2 - 6, y)], q, fade(ACCENT, 0.5), 2, dash=(5, 5))
            v.card(im, x0 + i * pitch + 30 * (1 - q), y, ts, q, r=9)
        xe = x0 + 3 * pitch + ts / 2 + 34
        v.stamp(xe, y, ok, prog(t, t0 + 0.65, 0.35), r=18)
        v.text((xe + 28, y), "solves" if ok else "deadlock", inter(19 * Z, b"SemiBold"),
               fade(GREEN if ok else RED, ease_out(prog(t, t0 + 0.7, 0.3))), "lm")
    # stat card
    p = ease_out(prog(t, 1.2, 0.5))
    v.rrect((850, 230, 1210, 560), 22, fill=fade(WHITE, p))
    c = ease_out(prog(t, 1.5, 1.6))
    v.ctext((1030, 300), "second plans that follow", inter(18 * Z), fade(MUTED, p))
    v.ctext((1030, 326), "a plan that really failed", inter(18 * Z), fade(MUTED, p))
    v.ctext((1030, 420), f"{int(round(74 + 24 * c))}%", inter(84 * Z, b"Bold"), fade(ACCENT, p))
    v.ctext((1030, 500), "74% before RL", inter(18 * Z, b"SemiBold"), fade(MUTED, p))


def sc_traps(v, t, dur):
    v.headline(t, [("And it escapes", INK), ("planning traps.", ACCENT)],
               sub="Pacman: the nearer food first leads into the ghost.")
    size, step = 300, 0.21
    for i, (frames, title, ok, verdict) in enumerate([(PAC_LIN, "Linear rollout", False, "caught by the ghost"),
                                                       (PAC_FWD, "Agima, forward thinking", True, "all food eaten")]):
        cx = 300 + i * 380; cy = 400
        p = ease_out(prog(t, 0.4 + i * 0.15, 0.5))
        k = min(len(frames) - 1, int(max(0, t - 0.9) / step))
        v.card(frames[k], cx, cy + 14 * (1 - p), size, p, outline=(GREEN if ok else RED) if k == len(frames) - 1 else None)
        v.ctext((cx, cy - size / 2 - 28), title, inter(21 * Z, b"Bold"), fade(INK, p))
        if k == len(frames) - 1:
            te = 0.9 + (len(frames) - 1) * step
            v.stamp(cx + size / 2 - 6, cy - size / 2 + 6, ok, prog(t, te, 0.35), r=20)
            v.ctext((cx, cy + size / 2 + 30), verdict, inter(19 * Z, b"SemiBold"), fade(GREEN if ok else RED, ease_out(prog(t, te, 0.3))))
    p = ease_out(prog(t, 1.0, 0.5)); c = ease_out(prog(t, 1.2, 1.5))
    v.rrect((900, 250, 1210, 550), 22, fill=fade(WHITE, p))
    v.ctext((1055, 300), "route-order traps (G8 / F4)", inter(17 * Z), fade(MUTED, p))
    v.ctext((1055, 390), f"{int(round(54 * c))}%", inter(80 * Z, b"Bold"), fade(ACCENT, p))
    v.ctext((1055, 450), "Agima, forward thinking", inter(17 * Z, b"SemiBold"), fade(INK, p))
    v.ctext((1055, 500), f"vs {int(round(19 * c))}% GPT-5.6 Sol", inter(17 * Z), fade(MUTED, p))


AUTHORS = [[("Jiaxin Ge", "1,*"), ("Junyi Zhang", "1,*"), ("Tianning Chai", "1,*"), ("Hanjun Yoo", "1")],
           [("Chenyang Wang", "2"), ("Xiuyu Li", "2"), ("Weijia Shi", "3"), ("Angjoo Kanazawa", "1"), ("XuDong Wang", "1,4")],
           [("Sewon Min", "1"), ("Luke Zettlemoyer", "3"), ("Trevor Darrell", "1,†"), ("Haiwen Feng", "1,2,†")]]
AFFIL = [("1", "UC Berkeley"), ("2", "Impossible Research"), ("3", "University of Washington"), ("4", "Duke University")]


def _marked_row(v, items, y, f_name, f_mark, col, mcol, gap, mark_first=False):
    """One centered line of 'name' + raised small 'mark' pairs."""
    widths = [v.tlen(n, f_name) + 2 + v.tlen(m, f_mark) for n, m in items]
    x = W / 2 - (sum(widths) + gap * (len(items) - 1)) / 2
    up = f_name.size / Z * 0.42
    for (n, m), w in zip(items, widths):
        if mark_first:
            v.text((x, y - up), m, f_mark, mcol, "ls")
            v.text((x + v.tlen(m, f_mark) + 2, y), n, f_name, col, "ls")
        else:
            v.text((x, y), n, f_name, col, "ls")
            v.text((x + v.tlen(n, f_name) + 2, y - up), m, f_mark, mcol, "ls")
        x += w + gap


def sc_end(v, t, dur):
    p = ease_out(prog(t, 0.2, 0.7))
    v.ctext((W / 2, 228 + 16 * (1 - p)), "Agima", inter(104 * Z, b"Bold"), fade(INK, p))
    q = ease_out(prog(t, 0.5, 0.6))
    v.ctext((W / 2, 312 + 10 * (1 - q)), "Planning with Agentic Imagination", inter(26 * Z), fade(MUTED, q))
    w = 120 * ease_io(prog(t, 0.8, 0.6))
    v.line([(W / 2 - w, 352), (W / 2 + w, 352)], ACCENT, 3)
    for i, row in enumerate(AUTHORS):
        r = ease_out(prog(t, 1.0 + i * 0.12, 0.5))
        _marked_row(v, row, 418 + i * 36 + 8 * (1 - r), inter(19 * Z, b"SemiBold"), inter(12 * Z, b"SemiBold"),
                    fade(INK, r), fade(MUTED, r), 26)
    r = ease_out(prog(t, 1.45, 0.5))
    _marked_row(v, [(n, m) for m, n in AFFIL], 546 + 8 * (1 - r), inter(16 * Z), inter(11 * Z), fade(MUTED, r), fade(MUTED, r), 26,
                mark_first=True)
    v.ctext((W / 2, 578 + 8 * (1 - r)), "* equal contribution     † equal advising", inter(14 * Z), fade(MUTED, r))


SCENES = [(sc_hook, 4.4), (sc_imagine, COMMIT_T + 0.6 + EXEC_STEP * len(SOLVE_MOVES) + 1.3), (sc_formats, 4.6),
          (sc_general, 4.6), (sc_results, 4.2), (sc_rl, 5.2), (sc_traps, 4.8), (sc_end, 4.2)]
XF = 0.35   # transition: outgoing scene shrinks and fades; incoming grows in


def frame(fn, t, dur):
    v = V(); fn(v, t, dur); im = v.out()
    a_in, a_out = ease_out(t / XF), ease_out((dur - t) / XF)
    if a_in >= 1 and a_out >= 1:
        return im
    bg = BG2.resize((W, H))
    s = 1.0 + 0.03 * (1 - a_in) - 0.03 * (1 - a_out)
    if abs(s - 1) > 1e-3:
        im = im.resize((int(W * s), int(H * s)), Image.BILINEAR)
        im = im.crop(((im.width - W) // 2, (im.height - H) // 2, (im.width - W) // 2 + W, (im.height - H) // 2 + H)) if s > 1 else \
            _pad(im, bg)
    return Image.blend(bg, im, min(a_in, a_out))


def _pad(im, bg):
    out = bg.copy(); out.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); return out


def main():
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for fn, dur in SCENES:
        for i in range(int(round(dur * FPS))):
            ff.stdin.write(frame(fn, i / FPS, dur).tobytes()); n += 1
    ff.stdin.close(); ff.wait()
    print("frames", n, "seconds", round(n / FPS, 1), "->", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "stills":
        outdir = sys.argv[3]
        for k, (fn, dur) in enumerate(SCENES):
            for frac in (0.35, 0.65, 0.95):
                frame(fn, dur * frac, dur).save(f"{outdir}/v2_{k:02d}_{int(frac * 100)}.png")
    else:
        main()
