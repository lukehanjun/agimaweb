"""Render the Agima teaser (1280x720, 30 fps). Every frame is drawn at 2x and downsampled.

usage:
  python make_teaser_v3.py                    # writes ../../assets/video/agima_teaser_v3.mp4
  python make_teaser_v3.py -o out.mp4         # writes somewhere else
  python make_teaser_v3.py --stills DIR       # saves preview stills of every scene to DIR instead

Scenes
  wall      a wall of the frames Agima imagined (from the site's rollout GIFs); the camera pulls back,
            then the tiles swirl into one Sokoban board
  omni      one omni model: board in, action out, then the imagined frame, which becomes the next input
  search    camera-follow Sokoban rollout: forward thinking, counterfactual branching, forward thinking;
            losing plans die, the kept one is acted on, then the camera pulls back over the whole path
  rlpac     after RL, Pacman plans are drafted one at a time until one works (left), next to the
            route-order trap clips and 54% vs 19% (right)
  envs      each task's train episode flips to its test episode, with its result bars beneath
  wipe      imagined vs real frames, compared with a sliding wipe
  end       title and authors
"""
import glob
import os
import random
import subprocess
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_argv, sys.argv = sys.argv, sys.argv[:1]
import make_teaser_v2 as V2   # noqa: E402
sys.argv = _argv

from make_teaser import MOVE_VEC  # noqa: E402
from make_teaser_v2 import (ACCENT, BG, BG2, BOARD, DEAD, ENV, EP34, FAINT, FIG5, GREEN, H, INK, LINE, MUTED, PAC_FWD, PAC_LIN,  # noqa: E402
                            RED, RL_ROWS, img, SITE, WHITE, W, Z, V, back_out, clamp, ease_io, ease_out, fade, gif_frames,
                            inter, mix, prog, sc_end)

OUT = f"{SITE}/video/agima_teaser_v3.mp4"
FPS = 30


def ease_in(x):
    x = clamp(x); return x ** 3


# ------------------------------------------------------------------ data: frames Agima imagined
def _is_imagined(f):
    r, g, b = np.asarray(f)[150, 1]
    return abs(int(r) - 36) < 20 and abs(int(b) - 121) < 25


def _episodes():
    """[(task, k, [(imagined, real), ...])] from the site's rollout GIFs (left panel = what Agima saw)."""
    out = []
    for g in sorted(glob.glob(f"{SITE}/gifs/rollout/*.gif")):
        task, k = os.path.basename(g)[:-4].split("_")
        seq = [(f.crop((4, 34, 252, 282)), f.crop((278, 34, 526, 282))) for f in gif_frames(g) if _is_imagined(f)]
        if seq:
            out.append((task, k, seq))
    return out


EPIS = _episodes()

# wall layout: every imagined frame once (plus repeats to fill the grid), one Sokoban frame in the middle
COLS, ROWS, PITCH, TS = 27, 15, 49, 45
_tiles = [(e, j) for e, (_, _, seq) in enumerate(EPIS) for j in range(len(seq))]
random.Random(7).shuffle(_tiles)
_tiles = (_tiles * 2)[:COLS * ROWS]
FOCUS = (COLS // 2, ROWS // 2)
_sok = next(i for i, (e, j) in enumerate(_tiles) if EPIS[e][0] == "sokoban")
_fi = FOCUS[1] * COLS + FOCUS[0]
_tiles[_fi], _tiles[_sok] = _tiles[_sok], _tiles[_fi]
MIP = {}


def _mip(e, j, s):
    """Tile image near size s (2x pixels), cached per octave."""
    lvl = 64 if s <= 64 else 128 if s <= 128 else 248
    key = (e, j, lvl)
    if key not in MIP:
        MIP[key] = EPIS[e][2][j][0].resize((lvl, lvl), Image.LANCZOS) if lvl < 248 else EPIS[e][2][j][0]
    return MIP[key]


# ------------------------------------------------------------------ data: the tree of pushes
S = DEAD
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reach(p, boxes):
    seen = {p}; q = deque([p])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n not in seen and n not in S.walls and n not in boxes:
                seen.add(n); q.append(n)
    return seen


def _pushes(p, boxes):
    R = _reach(p, boxes); out = []
    for b in sorted(boxes):
        for dr, dc in DIRS:
            stand, to = (b[0] - dr, b[1] - dc), (b[0] + dr, b[1] + dc)
            if stand in R and to not in S.walls and to not in boxes:
                out.append((b, frozenset((boxes - {b}) | {to})))
    return out


_solv = {}


def _solvable(p, boxes):
    key = (min(_reach(p, boxes)), boxes)
    if key not in _solv:
        seen = {key}; q = deque([(p, boxes)]); ok = False
        while q:
            pp, bb = q.popleft()
            if bb <= S.goals:
                ok = True; break
            for np_, nb in _pushes(pp, bb):
                k2 = (min(_reach(np_, nb)), nb)
                if k2 not in seen:
                    seen.add(k2); q.append((np_, nb))
        _solv[key] = ok
    return _solv[key]


def _img(state, size):
    return BOARD.draw(state).resize((size * Z, size * Z), Image.NEAREST)


ROOT_IMG = _img((S.player, S.boxes), 150)
CY = H / 2 - 20     # where the search scene starts its first board (camera sits 20 px below it)


# ------------------------------------------------------------------ drawing helpers
def glow_line(v, pts, col, width=3, glow=10, a=1.0):
    """Polyline with a soft halo (drawn on a cropped layer so the blur stays cheap)."""
    if len(pts) < 2 or a <= 0:
        return
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    pad = glow * 3
    x0, y0 = int((min(xs) - pad) * Z), int((min(ys) - pad) * Z)
    x1, y1 = int((max(xs) + pad) * Z), int((max(ys) + pad) * Z)
    m = Image.new("L", (max(1, x1 - x0), max(1, y1 - y0)), 0)
    ImageDraw.Draw(m).line([((x - min(xs) + pad) * Z, (y - min(ys) + pad) * Z) for x, y in pts], fill=int(150 * a),
                           width=int(glow * Z), joint="curve")
    m = m.filter(ImageFilter.GaussianBlur(glow * Z / 2.2))
    v.c.paste(Image.new("RGB", m.size, col), (x0, y0), m)
    v.line(pts, fade(col, a), width)


def glow_dot(v, x, y, r, col, a=1.0):
    pad = int(r * 3)
    m = Image.new("L", (2 * pad * Z, 2 * pad * Z), 0)
    ImageDraw.Draw(m).ellipse([(pad - r) * Z, (pad - r) * Z, (pad + r) * Z, (pad + r) * Z], fill=int(170 * a))
    m = m.filter(ImageFilter.GaussianBlur(r * Z * 0.9))
    v.c.paste(Image.new("RGB", m.size, col), (int((x - pad) * Z), int((y - pad) * Z)), m)


def paste_sq(v, im, cx, cy, size, a=1.0, ring=None, ring_w=2):
    """Square image (already at 2x) scaled to `size`, with rounded corners."""
    if a <= 0 or size < 1:
        return
    s = max(2, int(size * Z))
    im = im.resize((s, s), Image.NEAREST if im.width % s == 0 or s % im.width == 0 else Image.BILINEAR)
    m = Image.new("L", (s, s), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, s - 1, s - 1], radius=max(1, s // 8), fill=int(255 * a))
    v.c.paste(im, (int(cx * Z - s / 2), int(cy * Z - s / 2)), m)
    if ring:
        rr = size / 2 + 3
        v.rrect((cx - rr, cy - rr, cx + rr, cy + rr), max(2, size / 7), outline=fade(ring, a), width=ring_w)


def scrim_text(v, xy, text, font, col, a=1.0, pad=(26, 12)):
    """Text on a soft rounded plate, for reading over busy backgrounds."""
    if a <= 0:
        return
    w = v.tlen(text, font); h = font.size / Z
    x, y = xy
    box = (x - w / 2 - pad[0], y - h / 2 - pad[1], x + w / 2 + pad[0], y + h / 2 + pad[1] + 4)
    m = Image.new("L", (int((box[2] - box[0]) * Z), int((box[3] - box[1]) * Z)), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius=int(h * Z * 0.6), fill=int(225 * a))
    v.c.paste(Image.new("RGB", m.size, BG), (int(box[0] * Z), int(box[1] * Z)), m)
    v.ctext((x, y), text, font, fade(col, a))


# ------------------------------------------------------------------ scenes
WALL_ZOOM, WALL_TITLE, WALL_FLY = 3.0, 2.9, 4.4


def sc_wall(v, t, dur):
    # camera: exponential pull-back from one tile filling the frame to the whole wall
    z0, z1 = 600 / TS, 1.0
    zf = np.exp(np.log(z0) + (np.log(z1) - np.log(z0)) * ease_io(prog(t, 0.25, WALL_ZOOM)))
    fx = (FOCUS[0] - (COLS - 1) / 2) * PITCH; fy = (FOCUS[1] - (ROWS - 1) / 2) * PITCH
    cam = (fx * (zf - 1) / (z0 - 1) if z0 != 1 else 0, fy * (zf - 1) / (z0 - 1))
    fly = prog(t, WALL_FLY, 1.0)
    dim = 0.72 * ease_out(prog(t, WALL_TITLE, 0.6)) * (1 - ease_in(fly))
    for idx, (e, j) in enumerate(_tiles):
        c, r = idx % COLS, idx // COLS
        x = W / 2 + ((c - (COLS - 1) / 2) * PITCH - cam[0]) * zf
        y = H / 2 + ((r - (ROWS - 1) / 2) * PITCH - cam[1]) * zf
        s = TS * zf
        if fly > 0:   # every tile is drawn into the board at the centre, far ones last
            dist = np.hypot(x - W / 2, y - H / 2) / 760
            f = ease_in(prog(t, WALL_FLY + 0.35 * dist, 0.6))
            x, y, s = x + (W / 2 - x) * f, y + (CY - y) * f, s * (1 - f)
        if s < 1.5 or x + s < 0 or x - s > W or y + s < 0 or y - s > H:
            continue
        seq = EPIS[e][2]
        jj = (j + int((t + idx * 0.037) / 0.45)) % len(seq)
        paste_sq(v, _mip(e, jj, s * Z), x, y, s)
    if dim > 0:
        v.c.paste(Image.blend(v.c, BG2, dim))
    a = ease_out(prog(t, 1.1, 0.5)) * (1 - ease_out(prog(t, WALL_TITLE - 0.3, 0.4)))
    scrim_text(v, (W / 2, 640), "Every frame on this wall was imagined by Agima.", inter(24 * Z, b"SemiBold"), INK, a)
    p = ease_out(prog(t, WALL_TITLE + 0.1, 0.6)) * (1 - ease_out(prog(t, WALL_FLY - 0.1, 0.4)))
    if p > 0:
        v.ctext((W / 2, 330 + 16 * (1 - p)), "Agima", inter(120 * Z, b"Bold"), fade(INK, p))
        v.ctext((W / 2, 418 + 10 * (1 - p)), "Planning with Agentic Imagination", inter(28 * Z), fade(MUTED, p))
    q = ease_out(prog(t, WALL_FLY + 0.4, 0.6))
    if q > 0:
        paste_sq(v, ROOT_IMG, W / 2, CY, 150 * q)


# ------------------------------------------------------------------ one omni model: image in, action + imagined image out
OM_IN, OM_MID, OM_ACT, OM_OUT = (215, CY), (560, CY), (812, CY), (1040, CY)
OM_BS, OM_W, OM_H = 150, 210, 170
OM_IT = [(0.95, "down", 1), (2.55, "down", 2)]      # (start, action, resulting solution state)
OM_END = 4.25


def _model_box(v, t, a):
    """The single omni model, shown as its name; it glows while it generates."""
    if a <= 0:
        return
    s_ = 0.9 + 0.1 * back_out(a)
    busy = max([clamp(1 - abs(t - (t0 + 0.3)) / 0.45) for t0, _, _ in OM_IT] + [0])
    if busy > 0:
        glow_dot(v, OM_MID[0], OM_MID[1], 64, ACCENT, a * 0.5 * busy)
    v.ctext((OM_MID[0], OM_MID[1] - 4), "Agima", inter(int(58 * Z * s_), b"Bold"), fade(mix(INK, ACCENT, busy), a))


def _stream(v, a, b, t, t0, col):
    """Tokens flowing from a to b during [t0, t0 + 0.5]."""
    for k in range(6):
        f = (t - t0) * 2.4 - k * 0.16
        if 0 < f < 1:
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            glow_dot(v, x, y, 5, col, 0.9)
            v.d.ellipse(v.z(x - 3, y - 3, x + 3, y + 3), fill=col)


def sc_omni(v, t, dur):
    v.headline(t, [("One", INK), ("omni model", ACCENT), ("does both.", INK)], size=44, t0=0.25,
               sub="It reads the board, writes the action, then draws the frame that action leads to.")
    hide = ease_io(prog(t, OM_END, 0.6))
    # input slot: the real board, later replaced by the frame it just imagined
    slide = ease_io(prog(t, 0.05, 0.6)) * (1 - hide)
    ix = W / 2 + (OM_IN[0] - W / 2) * slide
    cur = 0
    for t0, _, k in OM_IT:
        if t > t0 + 1.3:
            cur = k
    if hide > 0:
        cur = 0
    la = ease_out(prog(t, 0.5, 0.3)) * (1 - hide)
    v.ctext((ix, OM_IN[1] + OM_BS / 2 + 26), "input: imagined frame" if cur else "input: real board", inter(16 * Z, b"SemiBold"),
            fade(ACCENT if cur else INK, la))
    ma = ease_out(prog(t, 0.35, 0.45)) * (1 - hide)
    if ma > 0:
        ia, ib = (OM_IN[0] + OM_BS / 2 + 10, CY), (OM_MID[0] - OM_W / 2 - 8, CY)
        v.line([ia, (ib[0] - 6, ib[1])], fade(LINE, ma), 2); v.head(ib[0], ib[1], fade(LINE, ma))
        oa, ob = (OM_MID[0] + OM_W / 2 + 8, CY), (OM_ACT[0] - 30, CY)
        v.line([oa, (ob[0] - 6, ob[1])], fade(LINE, ma), 2); v.head(ob[0], ob[1], fade(LINE, ma))
        fa, fb = (OM_ACT[0] + 30, CY), (OM_OUT[0] - OM_BS / 2 - 10, CY)
        v.line([fa, (fb[0] - 6, fb[1])], fade(LINE, ma), 2); v.head(fb[0], fb[1], fade(LINE, ma))
    _model_box(v, t, ma)
    v.card(_sq(SST[cur], IMAG_TINT if cur else None), ix, OM_IN[1], OM_BS, 1.0, nearest=True,
           outline=ACCENT if cur else INK, dashed=bool(cur), r=12)
    for t0, mv, k in OM_IT:
        if hide > 0:
            break
        _stream(v, (OM_IN[0] + OM_BS / 2 + 10, CY), (OM_MID[0] - OM_W / 2, CY), t, t0, INK)
        pa = prog(t, t0 + 0.45, 0.3)
        nxt = OM_IT[OM_IT.index((t0, mv, k)) + 1][0] if (t0, mv, k) != OM_IT[-1] else 99
        gone = prog(t, nxt - 0.05, 0.2)
        if pa > 0 and gone < 1:
            action_tag(v, OM_ACT[0], CY, mv, ACCENT, ease_out(pa) * (1 - gone), pop_r(pa, 26) / 15)
            v.ctext((OM_ACT[0], CY - 48), "action", inter(16 * Z, b"Bold"), fade(ACCENT, ease_out(pa) * (1 - gone)))
        pf = prog(t, t0 + 0.8, 0.35)
        fly = ease_io(prog(t, t0 + 1.25, 0.5)) if k < OM_IT[-1][2] else 0.0
        if pf > 0 and fly < 1:
            # the imagined frame appears, then loops back to become the next input
            fx = OM_OUT[0] + (OM_IN[0] - OM_OUT[0]) * fly
            fy = CY + 150 * np.sin(np.pi * fly)
            v.card(_sq(SST[k], IMAG_TINT), fx, fy, OM_BS * (1 - 0.15 * np.sin(np.pi * fly)), ease_out(pf), 0.85 + 0.15 * back_out(pf),
                   nearest=True, outline=ACCENT, dashed=True, r=12)
            if fly == 0:
                v.ctext((OM_OUT[0], CY + OM_BS / 2 + 26), "imagined frame", inter(16 * Z, b"Bold"), fade(ACCENT, ease_out(pf)))
        if fly > 0 and fly < 1:
            ctrl = [(OM_OUT[0] + (OM_IN[0] - OM_OUT[0]) * f, CY + 150 * np.sin(np.pi * f) + OM_BS / 2) for f in np.linspace(0, fly, 20)]
            v.line(ctrl, fade(ACCENT, 0.35), 2)
    fb = ease_out(prog(t, 2.0, 0.4)) * (1 - ease_out(prog(t, 3.6, 0.4))) * (1 - hide)
    if fb > 0:
        v.ctext((W / 2 + 120, 610), "…and the imagined frame becomes its next observation.", inter(20 * Z, b"SemiBold"), fade(INK, fb))


# ------------------------------------------------------------------ search: imagine, branch, act (camera follows)
SOL = V2.SOLVE_MOVES                      # 18 moves
SST = V2.SOLVE                            # 19 states
CHECK = [0, 6, 12, 18]                    # real boards: the agent sees the true board here


def _chain(start, moves):
    out = [start]
    for mv in moves:
        out.append(S.step(*out[-1], mv))
    return out[1:]


def _main_pos(m):
    return (m * 150.0, 70 * np.sin(m * 0.42) - m * 12.0)


MAIN = [_main_pos(m) for m in range(len(SST))]
# imagined dead ends: two forward-thinking rivals and one counterfactual branch (each is BFS-proven unsolvable)
DEAD_A = (0, ["left", "down", "left", "down", "right"], "row")   # from move 0: a whole rival plan, in a row above
BRANCH = (7, ["left"], (30, -170))               # off move 7
DEAD_B = (12, ["up", "left"], (118, 96))         # from move 12, falling
for m0, mv, _ in (DEAD_A, BRANCH, DEAD_B):
    assert not _solvable(*_chain(SST[m0], mv)[-1]), (m0, mv)


def _dead_frames(spec):
    m0, mv, geo = spec
    sts = _chain(SST[m0], mv)
    x0, y0 = MAIN[m0]
    if geo == "row":   # first step up and over, then straight along a row parallel to the main line
        return [(st, (x0 + 120 + 140 * k, y0 - 165), mv[k]) for k, st in enumerate(sts)]
    dx, dy = geo
    return [(st, (x0 + dx * (k + 1), y0 + dy * (k + 1)), mv[k]) for k, st in enumerate(sts)]


DEADS = [_dead_frames(DEAD_A), _dead_frames(BRANCH), _dead_frames(DEAD_B)]
_IMG = {}


def _sq(state, tint=None):
    key = (state, tint)
    if key not in _IMG:
        im = BOARD.draw(state).resize((512, 512), Image.NEAREST)
        if tint:
            im = Image.blend(im, Image.new("RGB", im.size, tint), 0.16)
        _IMG[key] = im
    return _IMG[key]


IMAG_TINT = (200, 216, 255)
# timeline (scene-local seconds)
PH = [  # phase start, caption, rival/branch spec index, main moves imagined (first, last)
    dict(t=0.0, cap=("Forward thinking", "imagine two plans, keep the one that works"), dead=0, first=1, last=6),
    dict(t=3.5, cap=("Counterfactual branching", "test a risky push, then continue"), dead=1, first=7, last=12),
    dict(t=6.9, cap=("Forward thinking", "again, all the way to the goal"), dead=2, first=13, last=18),
]
T_ZOOM = 10.1
STEP_IM = 0.21


def _phase_times(ph):
    t0 = ph["t"]
    d = ph["dead"]
    if d == 1:
        main0 = t0 + 0.35
        dead0 = main0 + STEP_IM * (BRANCH[0] - ph["first"] + 1) + 0.1
    else:
        dead0 = t0 + 0.55
        main0 = t0 + 0.85
    t_x = dead0 + 0.25 * len(DEADS[d]) + 0.25
    t_ok = main0 + STEP_IM * (ph["last"] - ph["first"] + 1) + 0.1
    t_commit = max(t_ok, t_x) + 0.3
    return main0, dead0, t_x, t_ok, t_commit


PT = [_phase_times(ph) for ph in PH]


ZOOM_OUT = None


def _act_head(t):
    """Fractional move index the agent has acted up to at time t."""
    mc = 0.0
    for i, ph in enumerate(PH):
        c = ease_io(prog(t, PT[i][4], 0.8))
        if c > 0:
            mc = max(mc, ph["first"] - 1 + c * (ph["last"] - ph["first"] + 1))
    return mc


def _main_at(mc):
    k = min(int(mc), len(MAIN) - 2); f = mc - k
    return MAIN[k][0] + (MAIN[k + 1][0] - MAIN[k][0]) * f, MAIN[k][1] + (MAIN[k + 1][1] - MAIN[k][1]) * f


def _camera(t):
    """Follow the acting head (current real board sits left of centre), then pull back to show the whole path."""
    global ZOOM_OUT
    if ZOOM_OUT is None:
        allp = MAIN + [p for d in DEADS for _, p, _ in d]
        xs, ys = [p[0] for p in allp], [p[1] for p in allp]
        z = min(W * 0.86 / (max(xs) - min(xs) + 160), H * 0.5 / (max(ys) - min(ys) + 160))
        ZOOM_OUT = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2 + 40 / z, z)
    hx, hy = _main_at(min(_act_head(t), PH[-1]["first"] - 1))
    off = 400 * ease_io(prog(t, 0.25, 0.9))
    cx, cy, zm = hx + off, hy + 20, 1.0 + (0.86 - 1.0) * ease_io(prog(t, 0.25, 0.9))
    z = ease_io(prog(t, T_ZOOM, 1.5))
    if z > 0:
        X, Y, Zf = ZOOM_OUT
        cx, cy = cx + (X - cx) * z, cy + (Y - cy) * z
        zm = np.exp(np.log(zm) + (np.log(Zf) - np.log(zm)) * z)
    return cx, cy, zm


ACT_LAG = 0.12   # an action is drawn this long before the frame it leads to


def action_tag(v, x, y, move, col, a, scale=1.0):
    """A filled pill: arrow glyph + the action's name, centred on (x, y)."""
    if a <= 0 or scale <= 0:
        return
    f = inter(max(8, int(13 * Z * scale)), b"Bold")
    tw = v.tlen(move, f); h = 22 * scale; aw = 11 * scale
    w = 10 * scale + aw + 5 * scale + tw + 10 * scale
    x0 = x - w / 2
    v.rrect((x0, y - h / 2, x0 + w, y + h / 2), h / 2, fill=fade(col, a), outline=fade(WHITE, a), width=max(1, int(1.5 * scale)))
    cx = x0 + 10 * scale + aw / 2
    dx, dy = MOVE_VEC[move]; L = aw / 2
    v.line([(cx - dx * L, y - dy * L), (cx + dx * (L - 3 * scale), y + dy * (L - 3 * scale))], fade(WHITE, a), max(1, int(2 * scale)))
    v.head(cx + dx * L, y + dy * L, fade(WHITE, a), (dx, dy), 4.5 * scale)
    v.text((cx + aw / 2 + 5 * scale, y + 0.5), move, f, fade(WHITE, a), "lm")


def pop_r(p, r):
    """Badge radius that pops in with a small overshoot."""
    return max(1.0, r * (0.4 + 0.6 * back_out(p)))


def sc_search(v, t, dur):
    cx, cy, zm = _camera(t)
    to_s = lambda p: (W / 2 + (p[0] - cx) * zm, H / 2 + (p[1] - cy) * zm)
    imag_n = 0; act_moves = []
    zoom_a = ease_out(prog(t, T_ZOOM + 0.6, 0.6))

    # committed path so far (glowing), drawn first so boards sit on top
    mc = _act_head(t)   # how far along the solution the agent has acted (fractional move index)
    done_m = int(mc)
    act_moves = SOL[:done_m]
    if mc > 0:
        pts = [to_s(MAIN[m]) for m in range(done_m + 1)]
        if done_m < len(MAIN) - 1 and mc > done_m:
            a_, b_ = to_s(MAIN[done_m]), to_s(MAIN[done_m + 1]); f = mc - done_m
            pts.append((a_[0] + (b_[0] - a_[0]) * f, a_[1] + (b_[1] - a_[1]) * f))
        glow_line(v, pts, GREEN, 4, 12, 1.0)

    badges = []
    # imagined dead ends and branches
    for i, ph in enumerate(PH):
        main0, dead0, t_x, t_ok, t_commit = PT[i]
        frames = DEADS[ph["dead"]]
        die = ease_out(prog(t, t_x, 0.5))
        prev = to_s(MAIN[BRANCH[0]] if ph["dead"] == 1 else MAIN[ph["first"] - 1])
        for k, (st, wp, mv) in enumerate(frames):
            ta = dead0 + 0.25 * k
            pa = prog(t, ta, 0.2)                     # the action appears first...
            p = ease_out(prog(t, ta + ACT_LAG, 0.3))   # ...then the frame it leads to
            if pa <= 0:
                continue
            imag_n += 1
            sp = to_s(wp)
            a = p * (1 - 0.45 * die) * (1 - 0.35 * zoom_a)
            aa = ease_out(pa) * (1 - 0.45 * die) * (1 - 0.35 * zoom_a)
            v.pline([prev, sp], ease_out(pa), fade(RED if die > 0.5 else ACCENT, 0.7 * aa), 2, dash=(6, 5))
            fr = 0.5 if k else 0.7   # keep the first action clear of the real board
            mx, my = prev[0] + (sp[0] - prev[0]) * fr, prev[1] + (sp[1] - prev[1]) * fr
            if zm > 0.5:
                badges.append((mx, my, mv, RED if die > 0.5 else ACCENT, aa, pop_r(pa, 16 * min(1, zm))))
            v.card(_sq(st, (255, 205, 195) if die > 0.5 else IMAG_TINT), sp[0], sp[1], 108 * zm, a, 0.85 + 0.15 * back_out(p),
                   nearest=True, outline=RED if die > 0.5 else ACCENT, dashed=True, r=10)
            prev = sp
        if die > 0:
            v.stamp(prev[0] + 54 * zm + 4, prev[1] - 54 * zm - 4, False, prog(t, t_x, 0.35), r=max(9, 17 * zm))

    # the main line: imagined first, solid once committed
    for i, ph in enumerate(PH):
        main0, dead0, t_x, t_ok, t_commit = PT[i]
        for m in range(ph["first"], ph["last"] + 1):
            ta = main0 + STEP_IM * (m - ph["first"])
            pa = prog(t, ta, 0.2)
            p = ease_out(prog(t, ta + ACT_LAG, 0.3))
            if pa <= 0:
                continue
            imag_n += 1
            done = m <= done_m
            a_, b_ = to_s(MAIN[m - 1]), to_s(MAIN[m])
            if not done:
                v.pline([a_, b_], ease_out(pa), fade(ACCENT, 0.7), 2, dash=(6, 5))
            if zm > 0.5:
                badges.append(((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2, SOL[m - 1], GREEN if done else ACCENT, ease_out(pa),
                               pop_r(pa, 16 * min(1, zm))))
            real = m in CHECK
            size = (150 if real and done else 108) * zm
            v.card(_sq(SST[m], None if done else IMAG_TINT), b_[0], b_[1], size, p, 0.85 + 0.15 * back_out(p), nearest=True,
                   outline=GREEN if done else ACCENT, dashed=not done, r=10)
        ok = prog(t, t_ok, 0.35)
        if ok > 0:
            e = to_s(MAIN[ph["last"]])
            v.stamp(e[0] + 54 * zm + 4, e[1] - 54 * zm - 4, True, ok, r=max(9, 17 * zm))
    # the starting real board
    p0 = to_s(MAIN[0])
    v.card(_sq(SST[0]), p0[0], p0[1], 150 * zm, 1.0, nearest=True, outline=GREEN if t > PT[0][4] else INK, r=12)
    if zm > 0.6:
        v.pill(p0[0] - 75 * zm, p0[1] - 75 * zm - 22, "real board", INK, None, 1 - zoom_a, size=13)
    for bx_, by_, mv_, col_, a__, r__ in badges:   # actions on top of the boards
        action_tag(v, bx_, by_, mv_, col_, a__, r__ / 16)

    # HUD, on a soft plate so passing boards never clash with it
    m = Image.new("L", (380 * Z, 92 * Z), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius=18 * Z, fill=232)
    v.c.paste(Image.new("RGB", m.size, BG), (50 * Z, 22 * Z), m)
    for k, (lab, val, col) in enumerate([("IMAGINED", imag_n, ACCENT), ("ACTED", len(act_moves), GREEN)]):
        x = 70 + k * 170
        v.text((x, 48), lab, inter(14 * Z, b"Bold"), MUTED, "ls")
        v.text((x, 94), f"{val}", inter(44 * Z, b"Bold"), col, "ls")
        v.text((x + v.tlen(f"{val}", inter(44 * Z, b"Bold")) + 6, 94), "frames" if k == 0 else "moves", inter(16 * Z), MUTED, "ls")
    # caption
    cap = None
    for i, ph in enumerate(PH):
        if PH[i]["t"] <= t < (PH[i + 1]["t"] if i + 1 < len(PH) else T_ZOOM):
            cap, ca = ph["cap"], ease_out(prog(t, ph["t"] + 0.1, 0.4)) * (1 - ease_out(prog(t, (PH[i + 1]["t"] if i + 1 < len(PH) else T_ZOOM) - 0.3, 0.3)))
    if cap:
        f1, f2 = inter(26 * Z, b"Bold"), inter(26 * Z)
        w1 = v.tlen(cap[0] + "  ", f1); w = w1 + v.tlen(cap[1], f2)
        v.text((W / 2 - w / 2, 612), cap[0], f1, fade(ACCENT, ca), "ls")
        v.text((W / 2 - w / 2 + w1, 612), cap[1], f2, fade(INK, ca), "ls")
    za = ease_out(prog(t, T_ZOOM + 1.0, 0.5))
    if za > 0:
        v.headline(t - T_ZOOM - 1.0, [("Imagine many futures.", INK), ("Act on one.", GREEN)], x=W / 2 - 260, y=612, size=34)
    # ACT track: every committed move, in order
    v.text((70, 672), "ACT", inter(15 * Z, b"Bold"), GREEN, "lm")
    for k, mv in enumerate(act_moves):
        v.badge(130 + k * 34, 672, mv, GREEN, 1.0, r=13)


WIPE = []
for task in ("maze2d", "sokoban", "pacman", "pusht"):
    WIPE.append(next(seq for tk, k, seq in EPIS if tk == task and k == "kinf"))
WIPE_NAMES = ["Maze2D", "Sokoban", "Pacman", "PushT"]


def sc_wipe(v, t, dur):
    v.headline(t, [("Its imagination", INK), ("tracks reality.", ACCENT)], size=44,
               sub="Each episode is imagined from a single real frame, then compared with the real one, step by step.")
    size, gap = 262, 30
    x0 = (W - 4 * size - 3 * gap) / 2
    fl = inter(16 * Z, b"Bold")
    for i, seq in enumerate(WIPE):
        p = ease_out(prog(t, 0.2 + i * 0.08, 0.5))
        x = x0 + i * (size + gap); y = 262 + 18 * (1 - p)
        k = int(max(0, t - 0.5) / 0.26) % len(seq)
        im, real = seq[k]
        s2 = int(size * Z)
        im2, re2 = im.resize((s2, s2), Image.BILINEAR), real.resize((s2, s2), Image.BILINEAR)
        u = 0.5 + 0.36 * np.sin(2 * np.pi * (t - 0.5) / 2.4 + i * 0.9) if t > 0.5 else 0.5
        cut = int(s2 * u)
        comp = re2.copy(); comp.paste(im2.crop((0, 0, cut, s2)), (0, 0))
        v.card(comp, x + size / 2, y + size / 2, size, p)
        # labels sit above the tile; a split strip shows which side is which
        v.text((x, y - 22), "Imagined", fl, fade(ACCENT, p), "ls")
        v.text((x + size, y - 22), "Real", fl, fade(INK, p), "rs")
        v.rrect((x, y - 12, x + size * u - 2, y - 6), 3, fill=fade(ACCENT, p))
        v.rrect((x + size * u + 2, y - 12, x + size, y - 6), 3, fill=fade(INK, p))
        lx = x + size * u
        v.line([(lx + 1, y + 1), (lx + 1, y + size)], fade((120, 120, 120), 0.5 * p), 3)
        v.line([(lx, y), (lx, y + size)], fade(WHITE, p), 3)
        v.d.ellipse(v.z(lx - 12, y + size / 2 - 12, lx + 12, y + size / 2 + 12), fill=fade(WHITE, p), outline=fade(INK, 0.5 * p), width=Z)
        for d in (-1, 1):
            v.head(lx + d * 7, y + size / 2, fade(INK, p), (d, 0), 5)
        v.ctext((x + size / 2, y + size + 32), WIPE_NAMES[i], inter(21 * Z, b"Bold"), fade(INK, p))


# ------------------------------------------------------------------ imagine/act in the same visual language
_TINT = {}


def _tinted(im, kind):
    """kind: None (as is), 'imag' (blue cast), 'dead' (red, washed out)."""
    if kind is None:
        return im
    key = (id(im), kind)
    if key not in _TINT:
        col, a = (IMAG_TINT, 0.16) if kind == "imag" else ((255, 205, 195), 0.38)
        _TINT[key] = Image.blend(im.convert("RGB"), Image.new("RGB", im.size, col), a)
    return _TINT[key]


def icard(v, im, pos, size, p, kind, r=10):
    """A board card: kind 'imag' (dashed blue), 'dead' (dashed red, faded), 'done' (solid green), 'real' (solid ink)."""
    if p <= 0:
        return
    tint = {"imag": "imag", "dead": "dead"}.get(kind)
    col = {"imag": ACCENT, "dead": RED, "done": GREEN, "real": INK}[kind]
    a = p * (0.6 if kind == "dead" else 1.0)
    v.card(_tinted(im, tint), pos[0], pos[1], size, a, 0.85 + 0.15 * back_out(p), nearest=True,
           outline=col, dashed=kind in ("imag", "dead"), r=r)


def ilink(v, a, b, p, kind, move=None):
    if p <= 0:
        return
    col = {"imag": ACCENT, "dead": RED, "done": GREEN}[kind]
    if kind == "done":
        glow_line(v, [a, (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)], GREEN, 3, 9, 1.0)
    else:
        v.pline([a, b], p, fade(col, 0.75), 2, dash=(6, 5))
    if move in MOVE_ICON and p > 0.6:
        v.badge((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, move, col, p, r=12)


MOVE_ICON = ("up", "down", "left", "right")


def caption(v, t, head, rest, t0=0.1, y=640):
    ca = ease_out(prog(t, t0, 0.4))
    f1, f2 = inter(26 * Z, b"Bold"), inter(26 * Z)
    w1 = v.tlen(head + "  ", f1); w = w1 + v.tlen(rest, f2)
    v.text((W / 2 - w / 2, y), head, f1, fade(ACCENT, ca), "ls")
    v.text((W / 2 - w / 2 + w1, y), rest, f2, fade(INK, ca), "ls")


def stat_card(v, t, t0, box, top, big, bottom, col=ACCENT):
    x0, y0, x1, y1 = box
    p = ease_out(prog(t, t0, 0.5))
    v.rrect(box, 24, fill=fade(WHITE, p))
    cx = (x0 + x1) / 2
    for k, line in enumerate(top):
        v.ctext((cx, y0 + 48 + 26 * k), line, inter(17 * Z), fade(MUTED, p))
    v.ctext((cx, (y0 + y1) / 2 + 22), big(ease_out(prog(t, t0 + 0.3, 1.6))), inter(84 * Z, b"Bold"), fade(col, p))
    v.ctext((cx, y1 - 40), bottom, inter(17 * Z, b"SemiBold"), fade(MUTED, p))


# --- Pacman route-order trap: forward thinking with the model's own imagined frames
_E = EP34
PAC_START = img(V2.T._pac_start(512))
PAC_IM = {c: [img(p) for p in sorted(glob.glob(f"{_E}/forward_thinking/imagined_by_candidate/{c}/*.png"))]
          for c in ("candidate_B_near_food_left_caught", "candidate_A_far_food_up_succeeds")}
PAC_IMV = {c: [os.path.basename(p)[3:].split("_")[0] for p in sorted(glob.glob(f"{_E}/forward_thinking/imagined_by_candidate/{c}/*.png"))]
           for c in PAC_IM}
PAC_GT = [img(p) for p in sorted(glob.glob(f"{_E}/ground_truth/candidate_A_far_food_up_succeeds/*.png"))]
PAC_GTV = [os.path.basename(p)[3:-4].split("_")[0] for p in sorted(glob.glob(f"{_E}/ground_truth/candidate_A_far_food_up_succeeds/*.png"))][1:]
PAC_PICK = {"candidate_B_near_food_left_caught": [0, 2, 5, 7], "candidate_A_far_food_up_succeeds": [1, 4, 7, 10]}
PAC_LANES = [("candidate_B_near_food_left_caught", 218, False, 0.5, "nearer food first", "caught"),
             ("candidate_A_far_food_up_succeeds", 518, True, 1.2, "farther food first", "all food eaten")]
PAC_COMMIT, PAC_EXEC = 3.4, 0.16


PAC_VAN = [img(p) for p in sorted(glob.glob(f"{_E}/vanilla/ground_truth/*.png"))][1:]
PAC_VANV = [os.path.basename(p)[3:-4].split("_")[0] for p in sorted(glob.glob(f"{_E}/vanilla/ground_truth/*.png"))][1:]
_NEAR, _FAR = "candidate_B_near_food_left_caught", "candidate_A_far_food_up_succeeds"
# three plans from one real board; after RL a new plan is drafted only once the previous one fails
RLP = [dict(frames=PAC_IM[_NEAR], moves=PAC_IMV[_NEAR], y=-205, name="Plan 1", desc="the nearer food first", ok=False),
       dict(frames=PAC_VAN, moves=PAC_VANV, y=0, name="Plan 2", desc="nearer food, then the far one", ok=False),
       dict(frames=PAC_IM[_FAR], moves=PAC_IMV[_FAR], y=205, name="Plan 3", desc="the farther food first", ok=True)]
RB, RF, RX0, RPITCH = 190, 108, 250, 140     # real board size, frame size, first frame x, pitch (world units)
SPD = 1 / 1.7                # the RL rollout runs 1.7x faster than the Sokoban one
R_STEP = 0.15 * SPD
_t = 0.4
for P in RLP:
    P["t0"] = _t
    P["t_end"] = _t + R_STEP * len(P["frames"])
    P["t_fail"] = P["t_end"] + 0.15 * SPD
    _t = P["t_fail"] + (1.15 if not P["ok"] else 0.75) * SPD   # back to the board before the next plan
R_ACT = RLP[-1]["t_fail"] + 0.6 * SPD
R_ACT_STEP = 0.16 * SPD
R_ZOOM = R_ACT + R_ACT_STEP * len(RLP[-1]["frames"]) + 0.5 * SPD
R_Z, VW = 0.72, 780          # the rollout fills the left VW px; the route-trap panel sits on the right
R_LEFT = (VW / 2 - 135) / R_Z


def _rx(i):
    return RX0 + i * RPITCH


def _r_keys():
    keys = [(0.0, R_LEFT, 0.0, R_Z, "io")]
    for P in RLP:
        far = max(R_LEFT, _rx(len(P["frames"]) - 1) - 300)
        keys += [(P["t0"], R_LEFT, -40, R_Z, "io"), (P["t_end"], far, -40, R_Z, "lin"),
                 (P["t_fail"] + 0.35 * SPD, far, -40, R_Z, "io")]
    far = max(R_LEFT, _rx(len(RLP[-1]["frames"]) - 1) - 300)
    keys += [(R_ACT, R_LEFT, -40, R_Z, "io"),
             (R_ACT + R_ACT_STEP * len(RLP[-1]["frames"]), far, -40, R_Z, "lin"),
             (R_ZOOM, far, -40, R_Z, "io")]
    xr = _rx(max(len(P["frames"]) for P in RLP) - 1) + RF
    z = min(VW * 0.92 / (xr + RB / 2 + 60), H * 0.62 / (2 * 205 + RF + 60))
    keys.append((R_ZOOM + 1.4 * SPD, (xr - RB / 2) / 2, -40, z, "io"))
    return keys


R_KEYS = _r_keys()


def _r_cam(t):
    for (t0, x0, y0, z0, _), (t1, x1, y1, z1, mode) in zip(R_KEYS, R_KEYS[1:]):
        if t <= t1:
            f = clamp((t - t0) / max(t1 - t0, 1e-6)); f = f if mode == "lin" else ease_io(f)
            return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, np.exp(np.log(z0) + (np.log(z1) - np.log(z0)) * f)
    return R_KEYS[-1][1:4]


def sc_rlpac(v, t, dur):
    cx, cy, zm = _r_cam(t)
    to_s = lambda p: (VW / 2 + (p[0] - cx) * zm, H / 2 + (p[1] - cy) * zm)
    zo = ease_out(prog(t, R_ZOOM + 0.6 * SPD, 0.6))
    act_n = int(clamp((t - R_ACT) / (R_ACT_STEP * len(RLP[-1]["frames"]))) * len(RLP[-1]["frames"])) if t > R_ACT else 0
    imag_n = 0; plans = 0; badges = []
    board = to_s((0, 0))
    for k, P in enumerate(RLP):
        if t < P["t0"] - 0.1:
            continue
        plans += 1
        dead = (not P["ok"]) and t > P["t_fail"] + 0.25
        y = P["y"]
        # plan label at the start of its row
        la = ease_out(prog(t, P["t0"] - 0.1, 0.3))
        lp = to_s((RX0 - RF / 2, y - RF / 2 - 22))
        col = RED if dead else (GREEN if P["ok"] and act_n else ACCENT)
        f_n = inter(int(19 * Z * max(zm, 0.55) / R_Z), b"Bold"); f_d = inter(int(17 * Z * max(zm, 0.55) / R_Z))
        v.text(lp, P["name"], f_n, fade(col, la), "ls")
        v.text((lp[0] + v.tlen(P["name"] + "  ", f_n), lp[1]), P["desc"], f_d, fade(MUTED, la), "ls")
        prev = (board[0] + RB / 2 * zm + 4, board[1])
        for i, (im, mv) in enumerate(zip(P["frames"], P["moves"])):
            pa = prog(t, P["t0"] + R_STEP * i, 0.15)
            p = ease_out(prog(t, P["t0"] + R_STEP * i + 0.07, 0.22))
            if pa <= 0:
                break
            imag_n += 1
            sp = to_s((_rx(i), y))
            acted = P["ok"] and i < act_n
            kind = "done" if acted else ("dead" if dead else "imag")
            a_ = (sp[0] - RF / 2 * zm - 4, sp[1])
            if i == 0:
                pts = V2.bezier(prev, a_, 24, 0.5)
                if kind == "done":
                    glow_line(v, pts, GREEN, 3, 9, 1.0)
                else:
                    v.pline(pts, ease_out(pa), fade(RED if dead else ACCENT, 0.75 * (0.6 if dead else 1)), 2, dash=(6, 5))
            else:
                ilink(v, prev, a_, ease_out(pa), kind, None)
            if mv in MOVE_ICON and zm > 0.55:
                badges.append(((prev[0] + a_[0]) / 2 if i else a_[0] - 22 * zm, (prev[1] + a_[1]) / 2 if i else a_[1], mv,
                               {"done": GREEN, "dead": RED, "imag": ACCENT}[kind], ease_out(pa) * (0.6 if dead else 1),
                               pop_r(pa, 15 * min(1, zm / R_Z))))
            icard(v, PAC_GT[i + 1] if acted else im, sp, RF * zm, p, kind)
            prev = (sp[0] + RF / 2 * zm + 4, sp[1])
        st = prog(t, P["t_fail"], 0.35)
        if st > 0:
            e = to_s((_rx(len(P["frames"]) - 1), y))
            v.stamp(e[0] + RF / 2 * zm + 22, e[1], P["ok"], st, r=max(10, 17 * zm / R_Z))
    for bx_, by_, mv_, col_, a__, r__ in badges:
        action_tag(v, bx_, by_, mv_, col_, a__, r__ / 16)
    # the real board (shows the acted state once acting starts)
    rb = PAC_START if act_n == 0 else PAC_GT[min(act_n, len(PAC_GT) - 1)]
    done = act_n >= len(RLP[-1]["frames"])
    icard(v, rb, board, RB * zm, 1.0, "done" if done else "real", r=12)
    v.pill(board[0] - RB / 2 * zm, board[1] - RB / 2 * zm - 20, "real board", INK, None, 1 - zo, size=13)
    # HUD
    m = Image.new("L", (520 * Z, 92 * Z), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius=18 * Z, fill=232)
    v.c.paste(Image.new("RGB", m.size, BG), (50 * Z, 22 * Z), m)
    for k, (lab, val, unit, col) in enumerate([("PLANS", plans, "drafted", ACCENT), ("IMAGINED", imag_n, "frames", ACCENT),
                                              ("ACTED", act_n, "moves", GREEN)]):
        x = 70 + k * 160
        v.text((x, 48), lab, inter(14 * Z, b"Bold"), MUTED, "ls")
        v.text((x, 94), f"{val}", inter(44 * Z, b"Bold"), col, "ls")
        v.text((x + v.tlen(f"{val}", inter(44 * Z, b"Bold")) + 6, 94), unit, inter(16 * Z), MUTED, "ls")
    # ACT track, under the counters
    if act_n:
        v.text((70, 138), "ACT", inter(14 * Z, b"Bold"), GREEN, "lm")
        for i, mv in enumerate(RLP[-1]["moves"][:act_n]):
            if mv in MOVE_ICON:
                v.badge(116 + i * 28, 138, mv, GREEN, 1.0, r=11)
    # the RL improvement, at the bottom of the rollout
    v.headline(t, [("After RL, it plans", INK), ("again", ACCENT), ("only when a plan fails.", INK)], x=VW / 2 - 330, y=636, size=30,
               t0=0.2)
    q = ease_out(prog(t, 0.6, 0.5)); c = ease_out(prog(t, 0.8, 2.2))
    v.ctext((VW / 2, 676), f"A second plan follows a real failure {int(round(74 + 24 * c))}% of the time (74% before RL).",
            inter(17 * Z, b"SemiBold"), fade(MUTED, q))
    trap_panel(v, t)


# --- the right-hand panel: the route-order trap, as two clips with glowing trails
TP_X0, TP_X1, TP_Y0, TP_Y1 = 806, 1250, 30, 690


def _pac_path(frames):
    pts = []
    for f in frames:
        a = np.asarray(f).astype(int)
        m = (a[..., 0] > 200) & (a[..., 1] > 190) & (a[..., 2] < 90)
        ys, xs = np.nonzero(m)
        pts.append((xs.mean() / a.shape[1], ys.mean() / a.shape[0]) if len(xs) else pts[-1])
    return pts


TP_CLIPS = [(PAC_LIN, _pac_path(PAC_LIN), "Nearer food first", RED, "caught by the ghost", False),
            (PAC_FWD, _pac_path(PAC_FWD), "Farther food first", ACCENT, "all food eaten", True)]
TP_STEP, TP_GO = 0.24, 0.7


def trap_panel(v, t):
    v.c.paste(BG2.crop((int(VW * Z), 0, W * Z, H * Z)), (int(VW * Z), 0))
    p = ease_out(prog(t, 0.1, 0.5))
    v.rrect((TP_X0, TP_Y0 + 14 * (1 - p), TP_X1, TP_Y1), 26, fill=fade(WHITE, p))
    x0 = TP_X0 + 26
    v.text((x0, TP_Y0 + 52), "Route-order trap", inter(24 * Z, b"Bold"), fade(INK, p), "ls")
    v.text((x0, TP_Y0 + 80), "Pacman: which food to eat first?", inter(16 * Z), fade(MUTED, p), "ls")
    size = 190
    for i, (frames, path, title, col, verdict, ok) in enumerate(TP_CLIPS):
        cx = TP_X0 + 26 + size / 2 + i * (size + 14); cy = 330
        q = ease_out(prog(t, 0.3 + i * 0.12, 0.5))
        tt = max(0.0, t - TP_GO) / TP_STEP
        k = min(len(frames) - 1, int(tt))
        v.card(frames[k], cx, cy, size, q, r=10)
        u = min(tt, len(path) - 1); n = int(u); f = u - n
        pts = path[:n + 1] + ([(path[n][0] + (path[n + 1][0] - path[n][0]) * f, path[n][1] + (path[n + 1][1] - path[n][1]) * f)]
                              if n + 1 < len(path) else [])
        sp = [(cx - size / 2 + x * size, cy - size / 2 + y * size) for x, y in pts]
        if len(sp) >= 2 and t > TP_GO:
            glow_line(v, sp, col, 3, 9, 1.0)
            glow_dot(v, sp[-1][0], sp[-1][1], 7, col)
        v.ctext((cx, cy - size / 2 - 22), title, inter(17 * Z, b"Bold"), fade(INK, q))
        if k == len(frames) - 1:
            te = TP_GO + (len(frames) - 1) * TP_STEP
            v.stamp(cx + size / 2 - 6, cy - size / 2 + 6, ok, prog(t, te, 0.35), r=17)
            v.ctext((cx, cy + size / 2 + 24), verdict, inter(16 * Z, b"SemiBold"), fade(GREEN if ok else RED, ease_out(prog(t, te, 0.3))))
    # the result
    c = ease_out(prog(t, 3.6, 1.4)); r_ = ease_out(prog(t, 3.4, 0.5))
    v.line([(x0, 540), (TP_X1 - 26, 540)], fade(FAINT, r_), 1)
    v.text((x0, 628), f"{int(round(54 * c))}%", inter(64 * Z, b"Bold"), fade(ACCENT, r_), "ls")
    v.text((x0 + 158, 588), "solved by Agima", inter(17 * Z, b"SemiBold"), fade(INK, r_), "ls")
    v.text((x0 + 158, 614), f"vs {int(round(19 * c))}% for GPT-5.6 Sol", inter(16 * Z), fade(MUTED, r_), "ls")
    v.text((x0 + 158, 638), "8×8 grid, 4 food", inter(14 * Z), fade(MUTED, r_), "ls")


# --- easy-to-hard, each task's result rising under it right after the flip to the test episode
ENV_RES = [("Maze2D", "maze2d", 1), ("Sokoban", "sokoban", 2), ("Pacman", "pacman", 0), ("PushT", "pusht", 3)]
ENV_DESC = {k: (tr, te) for _, k, tr, te in V2.ENVS}


def sc_envs(v, t, dur):
    v.headline(t, [("Trained on", INK), ("easy.", MUTED), ("Tested on", INK), ("hard.", ACCENT)], size=44,
               sub="Success on the hard test sets (K = 5): Agima vs the best baseline.")
    size, pitch = 196, 292
    x0 = W / 2 - 1.5 * pitch
    base, hmax = 640, 150
    for i, (name, key, ri) in enumerate(ENV_RES):
        cx = x0 + i * pitch; cy = 280
        trf, tef = ENV[key]
        t_flip = 1.4 + i * 0.35
        f = prog(t, t_flip, 0.5)
        sx = abs(np.cos(np.pi * f)) if 0 < f < 1 else 1.0
        test = f >= 0.5
        frames = tef if test else trf
        kf = int(max(0, t - (t_flip + 0.25 if test else 0)) / 0.2) % len(frames)
        p = ease_out(prog(t, 0.3 + i * 0.1, 0.5))
        v.card(frames[kf], cx, cy + 16 * (1 - p), size, p, sx=max(sx, 0.02), outline=ACCENT if test else None)
        v.ctext((cx, cy - size / 2 - 28), name, inter(22 * Z, b"Bold"), fade(INK, p))
        tr, te = ENV_DESC[key]
        lab, desc, col = ("TEST", te, ACCENT) if test else ("TRAIN", tr, MUTED)
        q = p if not test else ease_out(prog(t, t_flip + 0.25, 0.3))
        fb = inter(14 * Z, b"Bold"); fr = inter(16 * Z)
        w = v.tlen(lab, fb) + 9 + v.tlen(desc, fr)
        v.text((cx - w / 2, cy + size / 2 + 26), lab, fb, fade(col, q), "lm")
        v.text((cx - w / 2 + v.tlen(lab, fb) + 9, cy + size / 2 + 26), desc, fr, fade(INK, q), "lm")
        # results for this task, rising once its test episode is showing
        tb = t_flip + 0.55
        v.line([(cx - 80, base), (cx + 80, base)], fade(FAINT, ease_out(prog(t, tb - 0.2, 0.3))), 2)
        for j, (val, colb, lab2) in enumerate([(V2.BEST_BASE[ri], FAINT, "best baseline"), (V2.AGIMA[ri], ACCENT, "Agima")]):
            g = ease_out(prog(t, tb + 0.12 * j, 0.8))
            if g <= 0:
                continue
            bw = 50; x = cx - bw - 5 if j == 0 else cx + 5
            hgt = hmax * val / 45 * g
            if hgt > 2:
                v.rrect((x, base - hgt, x + bw, base), min(7, hgt / 2 - 0.5), fill=colb, corners=(True, True, False, False))
            v.ctext((x + bw / 2, base - hgt - 16), f"{int(round(val * g))}%", inter((22 if j else 16) * Z, b"Bold"),
                    fade(ACCENT if j else MUTED, min(1, g * 2)))
    q = ease_out(prog(t, 2.6, 0.4))
    v.rrect((W / 2 - 150, 676, W / 2 - 136, 690), 4, fill=fade(ACCENT, q)); v.text((W / 2 - 128, 683), "Agima", inter(15 * Z), fade(INK, q), "lm")
    v.rrect((W / 2 - 40, 676, W / 2 - 26, 690), 4, fill=fade(FAINT, q)); v.text((W / 2 - 18, 683), "best baseline", inter(15 * Z), fade(INK, q), "lm")


SCENES = [  # (scene, duration, fade in, fade out)
    (sc_wall, WALL_FLY + 1.25, True, False),
    (sc_omni, OM_END + 0.65, False, False),
    (sc_search, T_ZOOM + 2.8, False, True),
    (sc_rlpac, R_ZOOM + 1.4 * SPD + 2.0, True, True),
    (sc_envs, 1.4 + 3 * 0.35 + 0.55 + 0.9 + 1.6, True, True),
    (sc_wipe, 3.8, True, True),
    (sc_end, 4.2, True, False),
]
XF = 0.35


def frame(fn, t, dur, fin=True, fout=True):
    v = V(); fn(v, t, dur); im = v.out()
    a_in = ease_out(t / XF) if fin else 1.0
    a_out = ease_out((dur - t) / XF) if fout else 1.0
    if a_in >= 1 and a_out >= 1:
        return im
    bg = BG2.resize((W, H))
    s = 1.0 + 0.03 * (1 - a_in) - 0.03 * (1 - a_out)
    if abs(s - 1) > 1e-3:
        im2 = im.resize((int(W * s), int(H * s)), Image.BILINEAR)
        if s > 1:
            l, tp = (im2.width - W) // 2, (im2.height - H) // 2
            im = im2.crop((l, tp, l + W, tp + H))
        else:
            im = bg.copy(); im.paste(im2, ((W - im2.width) // 2, (H - im2.height) // 2))
    return Image.blend(bg, im, min(a_in, a_out))


def render_video(out):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for fn, dur, fi, fo in SCENES:
        for i in range(int(round(dur * FPS))):
            ff.stdin.write(frame(fn, i / FPS, dur, fi, fo).tobytes()); n += 1
    ff.stdin.close(); ff.wait()
    print("frames", n, "seconds", round(n / FPS, 1), "->", out, os.path.getsize(out) // 1024, "KB")


def render_stills(outdir):
    os.makedirs(outdir, exist_ok=True)
    for k, (fn, dur, fi, fo) in enumerate(SCENES):
        for frac in (0.2, 0.45, 0.7, 0.95):
            frame(fn, dur * frac, dur, fi, fo).save(f"{outdir}/{k:02d}_{fn.__name__}_{int(frac * 100)}.png")
    print("stills ->", outdir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render the Agima teaser video.")
    ap.add_argument("-o", "--out", default=OUT, help="output .mp4 (default: the site's assets/video/agima_teaser_v3.mp4)")
    ap.add_argument("--stills", metavar="DIR", help="save preview stills of every scene to DIR instead of rendering the video")
    args = ap.parse_args()
    if args.stills:
        render_stills(args.stills)
    else:
        render_video(args.out)
