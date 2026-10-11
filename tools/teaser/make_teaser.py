"""Render the Agima teaser video (1280x720, 24 fps) straight into ffmpeg.

usage: make_teaser.py [out.mp4] [stills]

Inputs are the website's own assets (figures, trap images, GIFs), the exported training/eval
examples under unified_world_model/artifacts, and fonts/Gelasio.ttf (Google Fonts, OFL).
The Sokoban trap trajectories are solved and re-rendered from the trap images' own tiles.
"""
import glob
import os
import subprocess
import sys
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.normpath(os.path.join(HERE, "..", "..", "assets"))   # the website's assets/
ART = os.path.join(HERE, "inputs")   # copies of the exported examples (originally unified_world_model/artifacts)
EP34 = f"{ART}/pacman_near_far_food_vanilla_vs_forward_ep034_branch_start_20260924"
TRAIN_EX = f"{ART}/sokoban_train_data_forward_branch_examples_20260925"
IDX19 = f"{ART}/sokoban_branching_checkpoint_examples_20260926/candidate_02_idx19/ground_truth/all_frames"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{SITE}/video/agima_teaser.mp4"

W, H, FPS = 1280, 720, 24
WHITE, INK, MUTED = (255, 255, 255), (23, 23, 23), (105, 105, 105)
NAVY, TEAL, BRICK, LINE = (36, 59, 121), (8, 124, 119), (191, 81, 59), (214, 218, 226)
GRAY_MID, GRAY_LIGHT = (139, 148, 168), (201, 206, 217)


def serif(size, weight=b"Regular"):
    f = ImageFont.truetype(f"{HERE}/fonts/Gelasio.ttf", size)
    f.set_variation_by_name(weight)
    return f


def inter(size, weight=b"Regular"):
    f = ImageFont.truetype(f"{HERE}/fonts/Inter.ttf", size)
    f.set_variation_by_name(weight)
    return f


F_EYE, F_H1, F_BODY, F_SMALL, F_LAB = inter(14, b"SemiBold"), serif(44), serif(25), inter(17), inter(14, b"SemiBold")
F_TAB, F_TABB = inter(17), inter(17, b"SemiBold")


# ------------------------------------------------------------------ helpers
def img(src, size=None):
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, WHITE + (255,)); bg.alpha_composite(im)
    im = bg.convert("RGB")
    return im.resize(size, Image.LANCZOS) if size else im


def ease(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)


def header(d, eyebrow, title, sub=None):
    d.text((70, 40), eyebrow.upper(), font=F_EYE, fill=MUTED)
    d.text((70, 62), title, font=F_H1, fill=INK)
    if sub:
        d.text((70, 118), sub, font=F_BODY, fill=MUTED)


def fade_alpha(t, dur, f=0.35):
    return min(1.0, t / f, (dur - t) / f)


def appear(t, start, step):
    """Number of items visible at time t when item i appears at start + i*step."""
    return 0 if t < start else int((t - start) / step) + 1


def loop_index(t, start, step, n, hold=3):
    """Frame of a looping GIF-like panel: plays n frames, holds the last for `hold` ticks."""
    return 0 if t < start else min(n - 1, int((t - start) / step) % (n + hold))


def gif_frames(path, box=None):
    im = Image.open(path); out = []
    for i in range(im.n_frames):
        im.seek(i)
        f = im.convert("RGB")
        out.append(f.crop(box) if box else f)
    return out


# ------------------------------------------------------------------ Sokoban trap trajectories
T = 32   # tile size of the 256 px trap renders (8x8 grid)
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def sok_parse(im):
    a = np.asarray(im.convert("RGB")).astype(int)
    grid = []
    for r in range(a.shape[0] // T):
        row = ""
        for c in range(a.shape[1] // T):
            t = a[r * T:(r + 1) * T, c * T:(c + 1) * T]
            blk = (t.sum(2) < 60).mean()
            yel = ((t[..., 0] > 200) & (t[..., 1] > 150) & (t[..., 2] < 100)).mean()
            grn = ((t[..., 1] > 150) & (t[..., 0] < 120)).mean()
            red = ((t[..., 0] > 180) & (t[..., 1] < 60)).mean()
            if grn > .05:
                row += "@"
            elif yel > .2:
                row += "*" if red > .03 else "$"
            elif blk < .2:
                row += "#"
            else:
                row += "." if red > .03 else " "
        grid.append(row)
    return grid


def _tile(im, grid, ch):
    for r, row in enumerate(grid):
        c = row.find(ch)
        if c >= 0:
            return im.crop((c * T, r * T, (c + 1) * T, (r + 1) * T))
    raise ValueError(ch)


class Sokoban:
    def __init__(self, path):
        self.im = img(path)
        g = sok_parse(self.im)
        self.walls = {(r, c) for r, row in enumerate(g) for c, ch in enumerate(row) if ch == "#"}
        self.goals = {(r, c) for r, row in enumerate(g) for c, ch in enumerate(row) if ch in ".*"}
        self.boxes = frozenset((r, c) for r, row in enumerate(g) for c, ch in enumerate(row) if ch in "$*")
        self.player = next((r, c) for r, row in enumerate(g) for c, ch in enumerate(row) if ch == "@")
        self.floor = [(r, c) for r, row in enumerate(g) for c, ch in enumerate(row) if ch != "#"]

    def step(self, player, boxes, move):
        dr, dc = MOVES[move]
        p = (player[0] + dr, player[1] + dc)
        if p in self.walls:
            return None
        if p in boxes:
            b = (p[0] + dr, p[1] + dc)
            if b in self.walls or b in boxes:
                return None
            boxes = (boxes - {p}) | {b}
        return p, frozenset(boxes)

    def solve(self):
        start = (self.player, self.boxes)
        prev = {start: None}
        q = deque([start])
        while q:
            s = q.popleft()
            if s[1] <= self.goals:
                path = []
                while prev[s]:
                    s, m = prev[s]; path.append(m)
                return path[::-1]
            for m in MOVES:
                n = self.step(*s, m)
                if n and n not in prev:
                    prev[n] = (s, m); q.append(n)
        raise RuntimeError("unsolvable")

    def render(self, player, boxes):
        im = self.im.copy()
        for r, c in self.floor:
            if (r, c) in boxes:
                spr = SPR["box_goal" if (r, c) in self.goals else "box"]
            elif (r, c) == player:
                spr = SPR["player_goal" if (r, c) in self.goals else "player"]
            else:
                spr = SPR["goal" if (r, c) in self.goals else "floor"]
            im.paste(spr, (c * T, r * T))
        return im

    def rollout(self, moves, size):
        s = (self.player, self.boxes)
        out = [(img(self.render(*s), size), "start")]
        for m in moves:
            n = self.step(*s, m)
            pushed = n[1] != s[1]
            s = n
            out.append((img(self.render(*s), size), f"{m} (push)" if pushed else m))
        return out


def _sprites():
    dead = img(f"{SITE}/images/traps/deadlock_start.webp"); g = sok_parse(dead)
    sp = {k: _tile(dead, g, ch) for k, ch in [("floor", " "), ("goal", "."), ("box", "$"), ("player", "@")]}
    done = img(sorted(glob.glob(f"{IDX19}/*.png"))[-1])
    sp["box_goal"] = _tile(done, sok_parse(done), "*")
    pg = sp["goal"].copy()
    pa = np.asarray(sp["player"]); mask = Image.fromarray(((pa.sum(2) > 60) * 255).astype(np.uint8))
    pg.paste(sp["player"], (0, 0), mask)
    sp["player_goal"] = pg
    return sp


SPR = _sprites()
DEAD = Sokoban(f"{SITE}/images/traps/deadlock_start.webp")
FORK = Sokoban(f"{SITE}/images/traps/fork_critical_step.webp")

# ------------------------------------------------------------------ data


FIG1 = img(f"{SITE}/images/fig1_overview.webp")
FIG4 = img(f"{SITE}/images/fig4_rl_curves.webp")
FIG5 = img(f"{SITE}/images/fig5_rl_qualitative.webp")

PS = 250   # problem scene panels
PROB = DEAD.rollout(["down", "left"], (2 * PS, 2 * PS))

FS = 104   # logical frame size in the format scenes (drawn at 2x, see Hi)
_bn = lambda p: os.path.basename(p)[3:-4]
BR_MAIN = [(img(p, (2 * FS, 2 * FS)), _bn(p))
           for p in sorted(glob.glob(f"{TRAIN_EX}/branching_example/main_line/*.png")) if "stop" not in p]
BR1 = img(glob.glob(f"{TRAIN_EX}/branching_example/branch_1_at_step_03/01_*.png")[0], (2 * FS, 2 * FS))
BR2 = img(glob.glob(f"{TRAIN_EX}/branching_example/branch_2_at_step_05/01_*.png")[0], (2 * FS, 2 * FS))
FF = 92    # candidate frames in forward thinking
FW_INIT = img(f"{TRAIN_EX}/forward_thinking_example/00_initial_state.png", (2 * FS, 2 * FS))
FW_C1 = [(img(p, (2 * FF, 2 * FF)), _bn(p)) for p in sorted(glob.glob(f"{TRAIN_EX}/forward_thinking_example/candidate_1_succeeds/0[1-9]_*.png")) if "stop" not in p]
FW_C2 = [(img(p, (2 * FF, 2 * FF)), _bn(p)) for p in sorted(glob.glob(f"{TRAIN_EX}/forward_thinking_example/candidate_2_certified_deadlock/0[1-9]_*.png"))]

GS = 196
ENVS = [("Maze2D", "maze2d", "small mazes", "larger mazes"),
        ("Sokoban", "sokoban", "two boxes", "four boxes"),
        ("Pacman", "pacman", "smaller grids, less food", "12×12 grid, 12 food"),
        ("PushT", "pusht", "near starts", "far starts")]
ENV_FRAMES = {k: tuple([img(f, (2 * GS, 2 * GS)) for f in gif_frames(f"{SITE}/gifs/env/{k}_{split}.gif", (0, 0, 256, 256))]
                       for split in ("train", "test")) for _, k, _, _ in ENVS}

# planning traps: each is (start image, start label, [candidate 1, candidate 2]); a candidate is
# (title, frames [(image, caption)], outcome, succeeded). Frames are real environment states.
TS, TG = 200, 160   # start image, candidate panels (stored at 2x)


def _pac_candidate(folder, size):
    out = []
    for p in sorted(glob.glob(f"{EP34}/ground_truth/{folder}/*.png")):
        name = os.path.basename(p)[3:-4]
        cap = "start" if name == "initial_state" else name.split("_")[0]
        out.append((img(p, size), cap))
    return out


def _pac_start(size):
    """Branching state with the two foods ringed, like the website's decision-point image."""
    im = img(f"{EP34}/ground_truth/00_initial_state.png")
    a = np.asarray(im)
    n, _, st, cen = cv2.connectedComponentsWithStats((a.min(2) > 200).astype(np.uint8))
    pellets = [cen[i] for i in range(1, n) if 9 <= st[i][2] <= 13 and 9 <= st[i][3] <= 13 and st[i][4] > 60]
    player = np.argwhere((a[..., 0] > 200) & (a[..., 1] > 200) & (a[..., 2] < 80)).mean(0)[::-1]
    pellets.sort(key=lambda p: np.abs(p - player).sum())   # nearer food first
    big = im.resize((size * 4, size * 4), Image.NEAREST)
    d = ImageDraw.Draw(big); s = size * 4 / 256
    for (x, y), col in zip(pellets, [BRICK, TEAL]):
        r = 15 * s
        d.ellipse([x * s - r, y * s - r, x * s + r, y * s + r], outline=col, width=int(3.5 * s))
    return big.resize((size, size), Image.LANCZOS)


TRAPS = [
    ("Deadlock", "One tempting push can never be undone.",
     "Sokoban: the natural push corners a box; another route solves the puzzle.",
     img(DEAD.render(DEAD.player, DEAD.boxes), (2 * TS, 2 * TS)), "start",
     [("Candidate 1", "the tempting push", DEAD.rollout(["down", "left"], (2 * TG, 2 * TG)), "Deadlock", False),
      ("Candidate 2", "another route", DEAD.rollout(DEAD.solve(), (2 * TG, 2 * TG)), "Solved", True)]),
    ("Structural Fork", "Two pushes look alike; only one keeps it solvable.",
     "Sokoban: at a critical step, the wrong push pins a box against the wall.",
     img(FORK.render(FORK.player, FORK.boxes), (2 * TS, 2 * TS)), "critical step",
     [("Candidate 1", "the wrong push", FORK.rollout(["up"], (2 * TG, 2 * TG)), "Deadlock", False),
      ("Candidate 2", "the right push", FORK.rollout(FORK.solve(), (2 * TG, 2 * TG)), "Solved", True)]),
    ("Pacman Route-order", "The nearer food is the wrong one to eat first.",
     "Pacman: which food to go for first decides whether the ghost catches you.",
     _pac_start(2 * TS), "red: nearer food · teal: farther food",
     [("Candidate 1", "nearer food first", _pac_candidate("candidate_B_near_food_left_caught", (2 * TG, 2 * TG)), "Caught by the ghost", False),
      ("Candidate 2", "farther food first", _pac_candidate("candidate_A_far_food_up_succeeds", (2 * TG, 2 * TG)), "All food eaten", True)]),
]

# Table 1 at K = 5 (hard test sets)
K5 = [("Gemini 3.1 Pro", "closed", [0, 20, 6, 0]), ("GPT-5.6 Sol", "closed", [1, 34, 20, 1]),
      ("Qwen2.5-VL-7B", "tuned", [0, 28, 0, 2]), ("Qwen3-VL-8B", "tuned", [0, 30, 3, 1]),
      ("Agima (No Imagine)", "noimg", [1, 23, 2, 1]), ("Agima", "ours", [37, 40, 40, 18])]
# planning-trap tables
SOK_T = (["Deadlock 2", "Deadlock 3", "Deadlock 4", "Fork 2", "Fork 3", "Fork 4"],
         [("Gemini 3.1 Pro", "closed", [74, 42, 19, 84, 34, 7]), ("GPT-5.6 Sol", "closed", [44, 23, 17, 60, 20, 12]),
          ("Qwen2.5-VL-7B", "tuned", [1, 2, 0, 12, 2, 0]), ("Qwen3-VL-8B", "tuned", [6, 1, 1, 19, 2, 0]),
          ("Agima (No Imagine)", "tuned", [12, 5, 1, 15, 2, 1]), ("Agima (Linear)", "lin", [22, 39, 27, 87, 34, 22]),
          ("Agima (Branching)", "ours", [30, 39, 32, 84, 39, 32]), ("Agima (Forward)", "ours", [60, 48, 24, 93, 46, 22])])
PAC_T = (["G8 / F2", "G8 / F4", "G8 / F6", "G8 / F8", "G10 / F8"],
         [("Gemini 3.1 Pro", "closed", [1, 1, 0, 1, 0]), ("GPT-5.6 Sol", "closed", [16, 19, 6, 8, 0]),
          ("Qwen2.5-VL-7B", "tuned", [0, 0, 0, 0, 0]), ("Qwen3-VL-8B", "tuned", [0, 0, 0, 1, 0]),
          ("Agima (No Imagine)", "tuned", [3, 0, 2, 0, 0]), ("Agima (Linear)", "lin", [7, 22, 16, 16, 5]),
          ("Agima (Forward)", "ours", [40, 54, 41, 37, 12])])


# ------------------------------------------------------------------ scenes
def s_title(c, d, t, dur):
    d.text((W // 2, 300), "Learning to Plan with", font=serif(56), fill=INK, anchor="mm")
    d.text((W // 2, 372), "Agentic Imagination", font=serif(56, b"SemiBold"), fill=NAVY, anchor="mm")
    if t > 1.0:
        d.text((W // 2, 450), "Agima", font=serif(30), fill=MUTED, anchor="mm")


def section(text):
    def s_section(c, d, t, dur):
        d.text((W // 2, 345), text, font=serif(56, b"SemiBold"), fill=INK, anchor="mm")
        w = 140 * ease((t - 0.2) / 0.5)
        d.line([(W // 2 - w, 410), (W // 2 + w, 410)], fill=NAVY, width=3)
    return s_section


PROB_GIF, PROB_END = 0.9, 3.0


def s_problem(c, d, t, dur):
    h = Hi(); gap = 115; x0 = (W - 3 * PS - 2 * gap) // 2; y = 180; cy = y + PS / 2
    h.card(PROB[0][0], x0, y)
    h.text((x0 + PS / 2, y + PS + 26), "start", h.font(17), MUTED, "mm")
    a = _alpha(t, PROB_GIF)
    if a > 0:
        xg = x0 + PS + gap
        h.arrow(x0 + PS + 16, xg - 14, cy, mix(LINE2, a))
        k = loop_index(t, PROB_GIF, 0.6, len(PROB))
        h.card(PROB[k][0], xg, y, a)
        cap = ["start", "move down", "push left"][k]
        h.text((xg + PS / 2, y + PS + 26), cap, h.font(17), mix(MUTED, a), "mm")
    a = _alpha(t, PROB_END)
    if a > 0:
        xe = x0 + 2 * (PS + gap)
        h.arrow(xe - gap + 16, xe - 14, cy, mix(LINE2, a))
        h.card(PROB[-1][0], xe, y, a)
        h.mark(xe + PS - 6, y + 6, False, mix(BRICK, a), r=15)
        h.text((xe + PS / 2, y + PS + 26), "Deadlock: this box can never reach a goal", h.font(17, b"SemiBold"), mix(BRICK, a), "mm")
    h.paste_to(c)
    header(d, "The problem", "One push can make a puzzle unsolvable.")
    if t > PROB_END + 0.8:
        d.text((W // 2, 590), "Real feedback arrives only after acting, when it is already too late.", font=F_BODY, fill=INK, anchor="mm")


# Figure 1 bands (pixel rows of the full-size figure)
FIG1_BANDS = [("IMAGINE", 312, 944, NAVY, "Imagine: generate the frames an action leads to, and plan against them."),
              ("ACT", 944, 1143, TEAL, "Act: only the committed actions reach the real environment.")]
IDEA_HL = [(0.9, 3.2), (3.4, 5.7)]   # time windows for the two highlights


def _fig1_frame(t):
    """Figure 1 with the IMAGINE / ACT band highlighted according to t."""
    for (a, b), (name, y0, y1, col, cap) in zip(IDEA_HL, FIG1_BANDS):
        if a <= t < b:
            k = ease(min((t - a) / 0.35, (b - t) / 0.35))
            dim = Image.blend(FIG1, Image.new("RGB", FIG1.size, WHITE), 0.72 * k)
            dim.paste(FIG1.crop((0, y0, FIG1.width, y1)), (0, y0))
            return dim, (name, y0, y1, col, cap, k)
    return FIG1, None


def s_idea(c, d, t, dur):
    h = Hi()
    fw = 1040; fh = int(FIG1.height * fw / FIG1.width)
    fx, fy = (W - fw) // 2, 150
    fig, hl = _fig1_frame(t)
    h.image(fig.resize((2 * fw, 2 * fh), Image.LANCZOS), fx, fy)
    cap = None
    if hl:
        name, y0, y1, col, cap, k = hl
        sc = fw / FIG1.width
        ya, yb = fy + y0 * sc, fy + y1 * sc
        if k > 0.3:
            h.rrect((fx - 6, ya - 3, fx + fw + 6, yb + 3), 12, outline=col, width=3)
            pw = h.tlen(name, h.font(14, b"SemiBold")) + 30
            h.pill(fx + fw - 24 - pw, ya - 3, name, col, size=14)
    h.paste_to(c)
    header(d, "The idea", "Agima imagines before it acts.")
    if cap and k > 0.3:
        d.text((W // 2, fy + fh + 34), cap, font=F_BODY, fill=col, anchor="mm")


# ------------------------------------------------------------------ the three formats
# Drawn on a 2x canvas and downsampled, so lines, circles and text are antialiased.
LINE2 = (178, 186, 202)
MOVE_VEC = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def mix(col, a):
    """col faded toward white; a = 1 is full color."""
    return tuple(int(255 + (v - 255) * a) for v in col)


def rounded_path(pts, r):
    """Polyline through pts with its corners rounded by radius r."""
    out = [pts[0]]
    for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
        v0 = np.subtract(p1, p0); v1 = np.subtract(p2, p1)
        a = np.subtract(p1, v0 / np.linalg.norm(v0) * r); b = np.add(p1, v1 / np.linalg.norm(v1) * r)
        out.append(tuple(a))
        out += [tuple((1 - s) ** 2 * a + 2 * (1 - s) * s * np.array(p1) + s * s * b) for s in np.linspace(0, 1, 10)]
    out.append(pts[-1])
    return out


class Hi:
    """A 2x canvas addressed in 1280x720 coordinates."""
    Z = 2
    _shadow = {}

    def __init__(self):
        self.c = Image.new("RGB", (W * self.Z, H * self.Z), WHITE)
        self.d = ImageDraw.Draw(self.c)

    def z(self, *v):
        return [x * self.Z for x in v]

    def card(self, im, x, y, a=1.0, r=9):
        """Image (already at 2x) with rounded corners and a soft shadow; fades and rises in with a."""
        if a <= 0:
            return
        Z = self.Z; w, h = im.size; pad = 14 * Z
        key = (w, h)
        if key not in self._shadow:
            sh = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
            ImageDraw.Draw(sh).rounded_rectangle([pad, pad + 3 * Z, pad + w, pad + h + 3 * Z], radius=r * Z, fill=64)
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r * Z, fill=255)
            self._shadow[key] = (sh.filter(ImageFilter.GaussianBlur(6 * Z)), m)
        sh, m = self._shadow[key]
        if a < 1:
            sh = sh.point(lambda v: int(v * a)); m = m.point(lambda v: int(v * a))
        X, Y = int(x * Z), int((y + 10 * (1 - ease(a))) * Z)
        self.c.paste(Image.new("RGB", sh.size, (38, 46, 72)), (X - pad, Y - pad), sh)
        self.c.paste(im, (X, Y), m)

    def line(self, pts, col, width=2, dash=None):
        pts = [tuple(np.multiply(p, self.Z)) for p in pts]
        if not dash:
            self.d.line(pts, fill=col, width=width * self.Z, joint="curve")
            return
        on, off = dash[0] * self.Z, dash[1] * self.Z
        for p, q in zip(pts, pts[1:]):
            n = np.hypot(*np.subtract(q, p)); u = np.subtract(q, p) / max(n, 1e-6); s = 0
            while s < n:
                e = min(n, s + on)
                self.d.line([tuple(p + u * s), tuple(p + u * e)], fill=col, width=width * self.Z)
                s += on + off

    def head(self, x, y, col, direction=(1, 0), size=7):
        dx, dy = direction; px, py = -dy, dx
        pts = [(x, y), (x - dx * size + px * size * 0.6, y - dy * size + py * size * 0.6),
               (x - dx * size - px * size * 0.6, y - dy * size - py * size * 0.6)]
        self.d.polygon([tuple(np.multiply(p, self.Z)) for p in pts], fill=col)

    def badge(self, cx, cy, move, col, a=1.0, r=15):
        """Circled action arrow, as in the paper's Figure 1."""
        if a <= 0:
            return
        col = mix(col, a)
        self.d.ellipse(self.z(cx - r, cy - r, cx + r, cy + r), fill=WHITE, outline=col, width=2 * self.Z)
        dx, dy = MOVE_VEC[move]; L = 7
        self.line([(cx - dx * L, cy - dy * L), (cx + dx * (L - 4), cy + dy * (L - 4))], col, 2)
        self.head(cx + dx * L, cy + dy * L, col, (dx, dy), 6)

    def text(self, xy, s, font, col, anchor="la"):
        self.d.text(tuple(np.multiply(xy, self.Z)), s, font=font, fill=col, anchor=anchor)

    def font(self, size, weight=b"Regular"):
        return inter(size * self.Z, weight)

    def tlen(self, s, font):
        return self.d.textlength(s, font=font) / self.Z

    def mark(self, cx, cy, ok, col, r=11, fill=True):
        """Check or cross icon in a filled circle."""
        if fill:
            self.d.ellipse(self.z(cx - r, cy - r, cx + r, cy + r), fill=col)
        ink = WHITE if fill else col
        k = r * 0.42
        if ok:
            self.line([(cx - k, cy + 0.1 * r), (cx - 0.12 * r, cy + k), (cx + k * 1.1, cy - k * 0.9)], ink, 2)
        else:
            self.line([(cx - k, cy - k), (cx + k, cy + k)], ink, 2)
            self.line([(cx - k, cy + k), (cx + k, cy - k)], ink, 2)

    def pill(self, x, cy, label, col, ok=None, a=1.0, size=17):
        """Filled pill with an optional check/cross icon; x is the left edge."""
        if a <= 0:
            return 0
        col = mix(col, a); f = inter(size * self.Z, b"SemiBold")
        icon = 0 if ok is None else 22
        w = self.tlen(label, f) + 30 + icon; h = size + 17
        self.d.rounded_rectangle(self.z(x, cy - h / 2, x + w, cy + h / 2), radius=h / 2 * self.Z, fill=col)
        if ok is not None:
            self.mark(x + 15 + 5, cy, ok, WHITE, r=9, fill=False)
        self.text((x + 15 + icon, cy), label, f, WHITE, "lm")
        return w

    def curve(self, p0, p1, col, bend=0.5):
        """Horizontal-tangent S-curve from p0 to p1 ending in an arrowhead."""
        (x0, y0), (x1, y1) = p0, p1
        xa, xb = x0 + (x1 - x0) * bend, x1 - (x1 - x0) * bend
        pts = [((1 - s) ** 3 * x0 + 3 * (1 - s) ** 2 * s * xa + 3 * (1 - s) * s * s * xb + s ** 3 * (x1 - 5),
                (1 - s) ** 3 * y0 + 3 * (1 - s) ** 2 * s * y0 + 3 * (1 - s) * s * s * y1 + s ** 3 * y1) for s in np.linspace(0, 1, 40)]
        self.line(pts, col); self.head(x1, y1, col)

    def arrow(self, x0, x1, y, col=LINE2):
        self.line([(x0, y), (x1 - 5, y)], col); self.head(x1, y, col)

    def image(self, im, x, y):
        self.c.paste(im, (int(x * self.Z), int(y * self.Z)))

    def rrect(self, box, r, fill=None, outline=None, width=2, corners=None):
        self.d.rounded_rectangle(self.z(*box), radius=r * self.Z, fill=fill, outline=outline,
                                 width=width * self.Z, corners=corners)

    def paste_to(self, c):
        c.paste(self.c.resize((W, H), Image.LANCZOS), (0, 0))


FMT_TABS = ["Linear rollout", "Counterfactual branching", "Forward thinking"]


def fmt_header(c, d, k, title, sub):
    header(d, "Three ways Agima imagines", title, sub)
    f, fb = inter(14), inter(14, b"SemiBold")
    x = W - 70
    for i in reversed(range(3)):
        lab = FMT_TABS[i]; font = fb if i == k else f
        w = d.textlength(lab, font=font); x -= w
        d.text((x, 42), lab, font=font, fill=NAVY if i == k else (150, 156, 170))
        if i == k:
            d.line([(x, 62), (x + w, 62)], fill=NAVY, width=2)
        x -= 26


def _alpha(t, t0, f=0.28):
    return max(0.0, min(1.0, (t - t0) / f))


# two rows of six for the 11-move solution (12 states)
ROW, GAP = 6, 52
PITCH = FS + GAP
GX0 = (W - (ROW - 1) * PITCH - FS) // 2


def _pos(i, y1, y2):
    return GX0 + (i % ROW) * PITCH, (y1, y2)[i // ROW]


def _link(h, i, move, y1, y2, a, col=LINE2, badge_col=NAVY):
    """Connector into state i with its action badge: straight within a row, a rounded return between rows."""
    if a <= 0:
        return
    x, y = _pos(i, y1, y2); cy = y + FS / 2; lc = mix(col, a)
    if i % ROW:
        x0 = x - GAP + 5
        h.line([(x0, cy), (x - 9, cy)], lc); h.head(x - 4, cy, lc)
        h.badge(x - GAP / 2, cy, move, badge_col, a)
        return
    xr = GX0 + (ROW - 1) * PITCH + FS
    cy1 = y1 + FS / 2; ym = (y1 + FS + y2) / 2
    h.line(rounded_path([(xr + 5, cy1), (xr + 30, cy1), (xr + 30, ym), (GX0 - 30, ym), (GX0 - 30, cy), (GX0 - 9, cy)], 16), lc)
    h.head(GX0 - 4, cy, lc)
    h.badge(W / 2, ym, move, badge_col, a)


def _caption(h, text, a=1.0, y=660):
    h.text((W / 2, y), text, h.font(19), mix(MUTED, a), "mm")


STEP = 0.42


def s_linear(c, d, t, dur):
    h = Hi(); y1, y2 = 222, 434
    for i, (im, move) in enumerate(BR_MAIN):
        a = _alpha(t, 0.3 + i * STEP)
        if i:
            _link(h, i, move, y1, y2, a)
        h.card(im, *_pos(i, y1, y2), a)
    t_end = 0.3 + len(BR_MAIN) * STEP
    x, y = _pos(len(BR_MAIN) - 1, y1, y2)
    h.pill(x + FS + 22, y + FS / 2, "Solved", TEAL, True, _alpha(t, t_end))
    _caption(h, "The first frame is the real board; every frame after it is imagined by Agima.", _alpha(t, t_end + 0.3))
    h.paste_to(c)
    fmt_header(c, d, 0, "Linear rollout: follow one plan.",
               "Imagine the next frame after every action and keep planning from it.")


def s_branch(c, d, t, dur):
    h = Hi(); y1, y2, yb = 340, 530, 174
    seq = ([("m", i) for i in range(4)] + [("b", 1)] + [("m", i) for i in range(4, 6)] + [("b", 2)]
           + [("m", i) for i in range(6, len(BR_MAIN))])
    for k, (kind, i) in enumerate(seq):
        a = _alpha(t, 0.3 + k * STEP)
        if kind == "m":
            im, move = BR_MAIN[i]
            if i:
                _link(h, i, move, y1, y2, a)
            h.card(im, *_pos(i, y1, y2), a)
        elif a > 0:
            src, im, alt = (3, BR1, "up") if i == 1 else (5, BR2, "right")
            bx = _pos(src, y1, y2)[0]; sx = bx + FS / 2
            col = mix(BRICK, a)
            h.line([(sx, y1 - 6), (sx, yb + FS + 10)], col, 2, dash=(6, 5)); h.head(sx, yb + FS + 5, col, (0, -1))
            h.badge(sx, (y1 + yb + FS) / 2, alt, BRICK, a)
            h.card(im, bx, yb, a)
            h.mark(bx + FS - 4, yb + 4, False, col, r=12)
            h.text((bx + FS + 16, yb + FS / 2 - 12), "Deadlock", h.font(18, b"SemiBold"), col, "lm")
            h.text((bx + FS + 16, yb + FS / 2 + 13), "abandoned", h.font(15), mix(MUTED, a), "lm")
    t_end = 0.3 + len(seq) * STEP
    x, y = _pos(len(BR_MAIN) - 1, y1, y2)
    h.pill(x + FS + 22, y + FS / 2, "Solved", TEAL, True, _alpha(t, t_end))
    h.paste_to(c)
    fmt_header(c, d, 1, "Counterfactual branching: test one move.",
               "At a risky step, imagine another action, see it fail, and continue the main plan.")


FW_LANES = [(FW_C1, 226, "Candidate 1", "push the lower-right box first", "Solves", True),
            (FW_C2, 458, "Candidate 2", "push the upper-left box first", "Deadlock", False)]
FW_LX, FW_PITCH = 272, FF + 48
FW_T2 = 0.5 + STEP * (len(FW_C1) + 1.5)                   # candidate 2 starts
FW_COMMIT = FW_T2 + STEP * (len(FW_C2) + 1.5)


def s_forward(c, d, t, dur):
    h = Hi()
    ix = 62; icy = (FW_LANES[0][1] + FW_LANES[1][1] + FF) / 2; iy = icy - FS / 2
    commit = ease((t - FW_COMMIT) / 0.4)
    for li, (frames, y, name, desc, outcome, ok) in enumerate(FW_LANES):
        t0 = 0.5 if li == 0 else FW_T2
        la = _alpha(t, t0 - 0.2)
        if la <= 0:
            continue
        fade = 1 - 0.55 * commit if li == 1 else 1.0
        x_end = FW_LX + len(frames) * FW_PITCH + 150
        band = (FW_LX - 22, y - 50, 1212, y + FF + 18)
        fill = tuple(int(u + (v - u) * commit) for u, v in zip((245, 247, 251), (232, 245, 243))) if li == 0 else (245, 247, 251)
        h.d.rounded_rectangle(h.z(*band), radius=14 * h.Z, fill=mix(fill, la) if la < 1 else fill,
                              outline=mix(TEAL, commit) if li == 0 and commit > 0 else None, width=2 * h.Z)
        h.text((FW_LX, y - 36), name, h.font(17, b"SemiBold"), mix(NAVY, la * fade), "lm")
        h.text((FW_LX + h.tlen(name, h.font(17, b"SemiBold")) + 12, y - 36), desc, h.font(16), mix(MUTED, la * fade), "lm")
        cy = y + FF / 2
        for i, (im, move) in enumerate(frames):
            a = _alpha(t, t0 + i * STEP) * fade
            if a <= 0:
                break
            x = FW_LX + i * FW_PITCH
            if i == 0:   # branch out of the shared start state
                pts = [(ix + FS + 6, icy)] + [((1 - s) ** 3 * (ix + FS + 6) + 3 * (1 - s) ** 2 * s * (FW_LX - 70) + 3 * (1 - s) * s * s * (FW_LX - 110) + s ** 3 * (x - 9),
                                               (1 - s) ** 3 * icy + 3 * (1 - s) ** 2 * s * icy + 3 * (1 - s) * s * s * cy + s ** 3 * cy) for s in np.linspace(0, 1, 30)]
                h.line(pts, mix(LINE2, a)); h.head(x - 4, cy, mix(LINE2, a))
                h.badge(x - 30, cy, move, NAVY, a)
            else:
                h.line([(x - 48 + 5, cy), (x - 9, cy)], mix(LINE2, a)); h.head(x - 4, cy, mix(LINE2, a))
                h.badge(x - 24, cy, move, NAVY, a)
            h.card(im, x, y, a)
        t_res = t0 + len(frames) * STEP
        xr = FW_LX + (len(frames) - 1) * FW_PITCH + FF + 22
        h.pill(xr, cy, outcome, TEAL if ok else BRICK, ok, _alpha(t, t_res) * fade)
    h.card(FW_INIT, ix, iy)
    h.text((ix + FS / 2, iy + FS + 18), "current state", h.font(15), MUTED, "mm")
    if commit > 0:
        f = h.font(19, b"SemiBold"); s = "Commit Candidate 1 and execute it."
        w = h.tlen(s, f) + 60
        h.pill(W / 2 - w / 2, 650, s, TEAL, True, commit, size=19)
    h.paste_to(c)
    fmt_header(c, d, 2, "Forward thinking: compare whole plans.",
               "Imagine complete candidate plans from the same state, then commit to one that succeeds.")


def s_general(c, d, t, dur):
    h = Hi()
    fb, fr = h.font(16, b"SemiBold"), h.font(16)
    for j, (name, key, tr, te) in enumerate(ENVS):
        gw = 2 * GS + 70; gx0 = (W - 2 * gw - 130) / 2
        cx = gx0 + (j % 2) * (gw + 130); cy = 162 + (j // 2) * 280
        trf, tef = ENV_FRAMES[key]
        ktr, kte = int(t / 0.32) % len(trf), int(t / 0.32) % len(tef)
        h.text((cx, cy), name, serif(2 * 26, b"SemiBold"), INK, "ls")
        ax, bx, fy = cx, cx + GS + 70, cy + 18
        h.card(trf[ktr], ax, fy); h.card(tef[kte], bx, fy)
        h.arrow(ax + GS + 14, bx - 12, fy + GS / 2)
        for x, lab, desc, col in [(ax, "Train", tr, GRAY_MID), (bx, "Test", te, NAVY)]:
            w1 = h.tlen(lab, fb); w = w1 + 8 + h.tlen(desc, fr)
            x0 = x + GS / 2 - w / 2
            h.text((x0, fy + GS + 24), lab, fb, col, "lm")
            h.text((x0 + w1 + 8, fy + GS + 24), desc, fr, MUTED if col == GRAY_MID else NAVY, "lm")
    h.paste_to(c)
    header(d, "Easy-to-hard generalization", "Trained on easy tasks, tested on harder ones.")


def _bar_fill(kind):
    return {"tuned": GRAY_LIGHT, "noimg": GRAY_MID, "ours": NAVY}.get(kind)


def s_results(c, d, t, dur):
    h = Hi()
    envs = ["Pacman", "Maze2D", "Sokoban", "PushT"]
    base_y, top_y = 590, 200
    scale = (base_y - top_y) / 50.0
    gx0, gw, bw, bg = 150, 272, 30, 8
    g = ease((t - 0.25) / 0.5)
    for v in (10, 20, 30, 40, 50):
        y = base_y - v * scale
        h.line([(gx0 - 20, y), (gx0 + 4 * gw - 30, y)], (236, 238, 242), 1)
        h.text((gx0 - 30, y), f"{v}%", h.font(15), MUTED, "rm")
    for ei, env in enumerate(envs):
        gx = gx0 + ei * gw
        for bi, (name, kind, vals) in enumerate(K5):
            x = gx + bi * (bw + bg)
            y = base_y - vals[ei] * g * scale
            r = min(4, (base_y - y) / 2 - 1)   # Pillow rejects a radius larger than half the bar
            if r >= 1:
                if kind == "closed":
                    h.rrect((x + 1, y, x + bw - 1, base_y), r, outline=GRAY_MID, width=2, corners=(True, True, False, False))
                else:
                    h.rrect((x, y, x + bw, base_y), r, fill=_bar_fill(kind), corners=(True, True, False, False))
            elif base_y - y >= 0.5:
                h.d.rectangle(h.z(x, y, x + bw, base_y), fill=_bar_fill(kind) if kind != "closed" else None,
                              outline=GRAY_MID if kind == "closed" else None)
            if g > 0.97:
                h.text((x + bw / 2, y - 6), str(vals[ei]), h.font(15, b"Bold") if kind == "ours" else h.font(13),
                       NAVY if kind == "ours" else MUTED, "mb")
        h.text((gx + 3 * (bw + bg) - bg / 2, base_y + 14), env, serif(2 * 24), INK, "mt")
    h.line([(gx0 - 20, base_y), (gx0 + 4 * gw - 30, base_y)], INK, 1)
    items = [("Agima", "ours"), ("Agima (No Imagine)", "noimg"), ("Qwen2.5-VL / Qwen3-VL, fine-tuned", "tuned"),
             ("Gemini / GPT, closed-source", "closed")]
    f = h.font(16)
    widths = [26 + h.tlen(n, f) for n, _ in items]
    lx = (W - sum(widths) - 36 * (len(items) - 1)) / 2
    for (name, kind), w in zip(items, widths):
        if kind == "closed":
            h.rrect((lx, 660, lx + 16, 676), 3, outline=GRAY_MID, width=2)
        else:
            h.rrect((lx, 660, lx + 16, 676), 3, fill=_bar_fill(kind))
        h.text((lx + 26, 668), name, f, INK, "lm")
        lx += w + 36
    h.paste_to(c)
    header(d, "Results", "At K = 5, Agima beats every baseline on all four tasks.",
           "Success rate on the hard test sets; a real frame only every 5 actions.")


# Fig. 5 regions in percent of the figure (left, top, right, bottom), in reading order.
# The figure's panels and white boxes stay put; only these regions' text and images appear.
_SEGS = [(29.5, 40.0), (40.0, 50.0), (50.0, 61.0), (61.0, 71.5)]


def _cand_row(y0, y1):
    return [(6.2, y0, 29.5, y1)] + [(a, y0, b, y1) for a, b in _SEGS] + [(71.5, y0, 100, y1)]


RL_A = ([(0, 1, 3.3, 52), (3.3, 0, 29.5, 16.5), (29.5, 0, 50.5, 16.5), (50.5, 0, 60.5, 16.5), (60.5, 0, 70.5, 16.5),
         (70.5, 0, 80.5, 16.5), (80.5, 0, 91.0, 16.5), (91.0, 0, 100, 16.5), (3.3, 16.5, 6.2, 51.5), (6.2, 16.5, 100, 23.1)]
        + _cand_row(23.1, 37.0) + _cand_row(37.0, 51.5))
RL_B = [(0, 54, 3.3, 100), (3.3, 54, 6.2, 100)] + _cand_row(55.8, 69.7) + _cand_row(69.7, 83.7) + _cand_row(83.7, 100)
RL_STEP, RL_START, RL_GAP = 0.22, 0.3, 0.9
RL_B_TOP = 54.2   # top edge of panel (b), percent


def _filled(mask, keep):
    """Solid (hole-free) versions of the connected components of mask that pass keep(x, y, w, h)."""
    out = np.zeros_like(mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if keep(*st[i][:4]):
            cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, 1, -1)
    return out


def _rl_plate(fig):
    """Fig. 5 with every text and image removed: panels, white boxes and their outlines only."""
    a = np.asarray(fig).copy()
    H_, W_ = a.shape[:2]
    inner = lambda x, y, w, h: x > 0 and y > 0 and x + w < W_ and y + h < H_
    boxes = _filled((a.min(2) >= 250).astype(np.uint8), lambda x, y, w, h: inner(x, y, w, h) and w > W_ / 3)
    panel_col = np.median(a[(a.min(2) < 250) & (a.min(2) > 235)], axis=0)
    near_panel = (np.abs(a.astype(int) - panel_col).max(2) <= 3).astype(np.uint8)
    panels = _filled(near_panel, lambda x, y, w, h: w > W_ / 2)
    k = lambda r: np.ones((r, r), np.uint8)
    panel_in = cv2.erode(panels, k(17)) & (1 - cv2.dilate(boxes, k(9)))
    a[panel_in > 0] = panel_col
    a[cv2.erode(boxes, k(7)) > 0] = 255
    return Image.fromarray(a)


RL_FW = 1110
RL_FH = int(FIG5.height * RL_FW / FIG5.width)
RL_FIG = FIG5.resize((2 * RL_FW, 2 * RL_FH), Image.LANCZOS)
RL_PLATE = _rl_plate(FIG5).resize((2 * RL_FW, 2 * RL_FH), Image.LANCZOS)


def s_rl_fig(c, d, t, dur):
    h = Hi(); Z = h.Z
    fw, fh = RL_FW, RL_FH
    fx, fy = (W - fw) // 2, 150
    frame = RL_PLATE.copy()
    tb = RL_START + RL_STEP * len(RL_A) + RL_GAP
    regions = RL_A[:appear(t, RL_START, RL_STEP)] + RL_B[:appear(t, tb, RL_STEP)]
    for l, tp, r, b in regions:
        box = (int(l / 100 * fw * Z), int(tp / 100 * fh * Z), int(r / 100 * fw * Z), int(b / 100 * fh * Z))
        frame.paste(RL_FIG.crop(box), box[:2])
    yb = int(RL_B_TOP / 100 * fh)
    if t > tb:   # fade panel (a) back so the emergent format stands out
        k = 0.55 * ease((t - tb) / 0.5)
        top = frame.crop((0, 0, fw * Z, yb * Z))
        frame.paste(Image.blend(top, Image.new("RGB", top.size, WHITE), k), (0, 0))
    h.image(frame, fx, fy)
    if t > tb + RL_STEP * len(RL_B):
        k = ease((t - tb - RL_STEP * len(RL_B)) / 0.4)
        y0 = fy + yb - 6
        h.rrect((fx - 10, y0 - 4, fx + fw + 10, fy + fh + 8), 18, outline=mix(NAVY, 0.18 * k), width=8)
        h.rrect((fx - 6, y0, fx + fw + 6, fy + fh + 4), 16, outline=mix(NAVY, k), width=3)
        if k > 0.5:
            f_b, f_r = h.font(16, b"Bold"), h.font(16)
            head, rest = "EMERGENT", "never shown in any demonstration"
            w = h.tlen(head, f_b) + h.tlen(rest, f_r) + 58
            px = W / 2 - w / 2
            h.rrect((px, y0 - 17, px + w, y0 + 17), 17, fill=NAVY)
            h.text((px + 20, y0), head, f_b, WHITE, "lm")
            xd = px + 20 + h.tlen(head, f_b) + 9
            h.d.ellipse(h.z(xd - 2.5, y0 - 2.5, xd + 2.5, y0 + 2.5), fill=(160, 175, 215))
            h.text((xd + 9, y0), rest, f_r, (214, 222, 242), "lm")
    h.paste_to(c)
    d.text((70, 40), "REINFORCEMENT LEARNING", font=F_EYE, fill=MUTED)
    if t < tb:
        d.text((70, 62), "Before RL: the formats it was taught.", font=F_H1, fill=INK)
        d.text((70, 112), "One linear rollout, or exactly two candidate plans every time.", font=F_BODY, fill=MUTED)
    else:
        d.text((70, 62), "After RL: an emergent, adaptive format.", font=F_H1, fill=NAVY)
        d.text((70, 112), "It checks each imagined plan and drafts another only when one fails.", font=F_BODY, fill=MUTED)


FIG4_2X = FIG4.resize((1400, int(FIG4.height * 1400 / FIG4.width)), Image.LANCZOS)


def s_rl(c, d, t, dur):
    h = Hi()
    h.image(FIG4_2X, 40, 196 + (372 - FIG4_2X.height / 2) / 2)
    lines = [("79% → 26–31%", "of episodes draft a second plan"),
             ("74% → 98%", "of second plans follow a first plan that really fails"),
             ("1.70 → 1.95", "branches per plan, focused on critical steps")]
    for i, (big, small) in enumerate(lines):
        a = _alpha(t, 0.6 + i * 1.0, 0.35)
        if a <= 0:
            continue
        y = 196 + i * 124
        h.rrect((770, y, 1225, y + 106), 14, fill=mix((243, 246, 251), a))
        h.rrect((770, y + 18, 774, y + 88), 2, fill=mix(NAVY, a))
        h.text((796, y + 38), big, serif(2 * 34, b"SemiBold"), mix(NAVY, a), "lm")
        h.text((796, y + 78), small, h.font(16), mix(INK, a), "lm")
    h.paste_to(c)
    header(d, "Reinforcement learning", "Learned from task success alone.",
           "No demonstration says when to draft another plan or where to branch.")


TRAP_GO = 1.0   # candidates start playing


def _trap_step(frames):
    return min(0.42, 4.2 / len(frames))


def trap_scene(trap):
    name, title, sub, start, start_label, cands = trap

    def scene(c, d, t, dur):
        h = Hi()
        sx = 62; rows = (256, 498); lx = 330; gx = 356; fx = gx + TG + 100
        sy = (rows[0] + rows[1] + TG) / 2 - TS / 2
        h.card(start, sx, sy)
        h.text((sx + TS / 2, sy + TS + 22), start_label, h.font(15), MUTED, "mm")
        for li, ((cname, desc, frames, outcome, ok), y) in enumerate(zip(cands, rows)):
            la = _alpha(t, TRAP_GO - 0.3)
            if la <= 0:
                break
            col = TEAL if ok else BRICK
            cy = y + TG / 2
            h.rrect((lx, y - 50, 1212, y + TG + 16), 14, fill=mix((245, 247, 251), la))
            f_n = h.font(17, b"SemiBold")
            h.text((gx, y - 27), cname, f_n, mix(NAVY, la), "lm")
            h.text((gx + h.tlen(cname, f_n) + 12, y - 27), desc, h.font(16), mix(MUTED, la), "lm")
            h.curve((sx + TS + 8, sy + TS / 2), (gx - 8, cy), mix(LINE2, la))
            step = _trap_step(frames)
            k = loop_index(t, TRAP_GO, step, len(frames))
            im, cap = frames[k]
            h.card(im, gx, y, la)
            # connector to the final state carries the move being played
            h.arrow(gx + TG + 12, fx - 12, cy, mix(LINE2, la))
            move = cap.split()[0]
            if k and move in MOVE_VEC:
                h.badge((gx + TG + fx) / 2, cy, move, NAVY, la)
            h.text(((gx + TG + fx) / 2, cy + 30), f"step {k}" if k else "start", h.font(13), mix(MUTED, la), "mm")
            a = _alpha(t, TRAP_GO + step * len(frames))
            if a > 0:
                h.card(frames[-1][0], fx, y, a)
                h.mark(fx + TG - 6, y + 6, ok, mix(col, a), r=14)
                h.pill(fx + TG + 26, cy, outcome, col, ok, a)
        h.paste_to(c)
        d.text((70, 40), "PLANNING TRAPS", font=F_EYE, fill=MUTED)
        d.text((70, 58), name, font=serif(50, b"SemiBold"), fill=INK)
        d.text((70, 124), title, font=serif(28), fill=INK)
        d.text((70, 162), sub, font=serif(22), fill=INK)
    return scene


def trap_dur(trap):
    return TRAP_GO + max(_trap_step(f) * len(f) for _, _, f, _, _ in trap[5]) + 2.6


def draw_table(c, d, t, title, sub, table, name_w=260):
    d.text((70, 40), "PLANNING TRAPS: RESULTS", font=F_EYE, fill=MUTED)
    d.text((70, 62), "Agima succeeds on planning traps.", font=F_H1, fill=INK)
    d.text((70, 118), title, font=serif(28, b"SemiBold"), fill=INK)
    d.text((70 + d.textlength(title, font=serif(28, b"SemiBold")) + 16, 122), sub, font=serif(22), fill=MUTED)
    cols, rows = table
    x0 = 90
    cw = (W - 2 * x0 - name_w) / len(cols)
    best = [max(r[2][i] for r in rows) for i in range(len(cols))]
    y = 176
    for i, col in enumerate(cols):
        d.text((x0 + name_w + (i + 0.5) * cw, y + 12), col, font=F_TABB, fill=INK, anchor="mm")
    y += 32
    d.line([(x0 - 12, y), (W - x0 + 12, y)], fill=INK, width=2)
    y += 6
    group = None
    for name, kind, vals in rows:
        g = "Closed-source" if kind == "closed" else "Fine-tuned on our training data"
        if g != group:
            if group:
                y += 4
            d.text((x0, y + 6), g.upper(), font=F_LAB, fill=MUTED)
            y += 28; group = g
        rh = 40
        if kind in ("ours", "lin"):
            d.rounded_rectangle([x0 - 12, y, W - x0 + 12, y + rh - 4], radius=8, fill=(236, 241, 250) if kind == "ours" else (245, 246, 248))
        ink = MUTED if kind == "closed" else (NAVY if kind == "ours" else INK)
        d.text((x0, y + rh // 2 - 2), name, font=F_TABB if kind == "ours" else F_TAB, fill=ink, anchor="lm")
        for i, v in enumerate(vals):
            d.text((x0 + name_w + (i + 0.5) * cw, y + rh // 2 - 2), f"{v}%",
                   font=F_TABB if v == best[i] else F_TAB, fill=ink, anchor="mm")
        y += rh
    d.text((W - x0, y + 14), "bold = best in column", font=F_SMALL, fill=MUTED, anchor="rt")


def s_table_sok(c, d, t, dur):
    draw_table(c, d, t, "Sokoban: Deadlock and Structural Fork",
               "trained on two boxes; tested with 2, 3 and 4", SOK_T)


def s_table_pac(c, d, t, dur):
    draw_table(c, d, t, "Pacman: Route-order trap",
               "G = grid size, F = food count", PAC_T)


def s_end(c, d, t, dur):
    d.text((W // 2, 320), "Agima", font=serif(72, b"SemiBold"), fill=NAVY, anchor="mm")
    d.text((W // 2, 400), "Learning to Plan with Agentic Imagination", font=serif(34), fill=INK, anchor="mm")


SCENES = ([(s_title, 3.0), (s_problem, 6.5), (s_idea, 6.4), (section("Three ways Agima imagines"), 2.4),
           (s_linear, 0.3 + STEP * len(BR_MAIN) + 2.6), (s_branch, 0.3 + STEP * (len(BR_MAIN) + 2) + 2.4),
           (s_forward, FW_COMMIT + 2.6), (s_general, 8.0), (s_results, 5.0), (section("Reinforcement Learning on Agima"), 2.4),
           (s_rl_fig, 2 * RL_START + RL_STEP * (len(RL_A) + len(RL_B)) + RL_GAP + 2.8), (s_rl, 6.0)]
          + [(section("Evaluating on Planning Traps"), 2.4)]
          + [(trap_scene(tr), trap_dur(tr)) for tr in TRAPS]
          + [(s_table_sok, 5.5), (s_table_pac, 5.0), (s_end, 3.5)])


def render(fn, t, dur):
    c = Image.new("RGB", (W, H), WHITE)
    fn(c, ImageDraw.Draw(c), t, dur)
    return c


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    white = np.full((H, W, 3), 255, np.float32)
    n = 0
    for fn, dur in SCENES:
        for i in range(int(dur * FPS)):
            t = i / FPS
            arr = np.asarray(render(fn, t, dur), dtype=np.float32)
            a = fade_alpha(t, dur)
            if a < 1:
                arr = arr * a + white * (1 - a)
            ff.stdin.write(arr.astype(np.uint8).tobytes()); n += 1
    ff.stdin.close(); ff.wait()
    print("frames", n, "seconds", round(n / FPS, 1), "->", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "stills":
        outdir = sys.argv[3] if len(sys.argv) > 3 else HERE
        for f in glob.glob(f"{outdir}/still_*.png"):
            os.remove(f)
        for k, (fn, dur) in enumerate(SCENES):
            for frac in (0.3, 0.6, 0.93):
                render(fn, dur * frac, dur).save(f"{outdir}/still_{k:02d}_{int(frac * 100)}.png")
    else:
        main()
