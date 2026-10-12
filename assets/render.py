"""Raymarch a Mexican golden eagle perched on a nopal, snake in beak, and plot it
as rotating colour ASCII in an animated SVG (wings folded, or raised like on the flag).

    python3 assets/render.py > assets/eagle.svg   (WINGS = "down" or "up")
"""
import sys
from multiprocessing import Pool

import numpy as np

COLS, ROWS, FRAMES, SECONDS = 84, 60, 48, 6.0
RAMP = " .,:;-=+*x#%@"
FONT, CW, LH = 10, 6.0, 10.0
SPAN, CY = 2.3, 1.29
WINGS = "down"  # "down" = folded at its sides, "up" = raised like the flag
CS, ES = 0.85, 1.4  # nopal and eagle scale  # world width shown, vertical centre

V = lambda *x: np.array(x, dtype=float)
D = np.radians


def rx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def sphere(p, c, r):
    return np.linalg.norm(p - c, axis=-1) - r


def ellipsoid(p, c, r, R=None):
    q = p - c
    if R is not None:
        q = q @ R
    q = q / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def cone(p, a, b, r1, r2):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - ba * h[..., None], axis=-1) - (r1 + (r2 - r1) * h)


def smin(a, b, k=0.1):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


# ---------------------------------------------------------------- the nopal
PADS = [(V(0, 0.72, 0), V(0.30, 0.40, 0.09), ry(D(10))),
        (V(-0.42, 1.02, 0.05), V(0.25, 0.32, 0.08), ry(D(-35)) @ rz(D(35))),
        (V(0.42, 0.98, -0.05), V(0.24, 0.30, 0.08), ry(D(55)) @ rz(D(-32))),
        (V(0.0, 1.18, 0.0), V(0.27, 0.27, 0.08), ry(D(80))),
        (V(-0.27, 0.52, -0.24), V(0.18, 0.24, 0.07), ry(D(120)) @ rz(D(28))),
        (V(0.33, 0.52, 0.22), V(0.17, 0.22, 0.07), ry(D(-60)) @ rz(D(-30)))]
# prickly-pear fruit (tunas) sitting on the pad rims
TUNAS = [c + R @ V(dx, r[1] * 0.95, 0) for (c, r, R), dx in
         [(PADS[1], -0.08), (PADS[1], 0.1), (PADS[2], 0.0), (PADS[4], 0.05), (PADS[5], -0.04), (PADS[0], 0.17)]]

# ---------------------------------------------------------------- the eagle
Y0 = 1.44  # talons on the top pad
EYES = [V(0.095 * s, Y0 + 0.86, 0.19) for s in (-1, 1)]
BEAK_TIP = V(0, Y0 + 0.75, 0.40)
FLAP = D(36)  # wing angle, fixed


def wing_frames(side, e):
    """Bone line shoulder -> elbow -> wrist, plus each segment's (axis, chord, normal)."""
    sh = V(0.23 * side, Y0 + 0.6, -0.06)
    d1 = V(np.cos(e) * side, np.sin(e), -0.15); d1 /= np.linalg.norm(d1)
    el = sh + d1 * 0.42
    e2 = e + D(14)
    d2 = V(np.cos(e2) * side, np.sin(e2), -0.18); d2 /= np.linalg.norm(d2)
    wr = el + d2 * 0.34
    segs = []
    for a, b in ((sh, el), (el, wr)):
        ax = (b - a) / np.linalg.norm(b - a)
        z = V(0, 0, 1) - ax * ax[2]; z /= np.linalg.norm(z)
        n = np.cross(ax, z)
        tw = D(68) * side  # twist the feathered face toward the front, like the flag
        ch = z * np.cos(tw) + n * np.sin(tw)
        segs.append((a, b, ax, ch, np.cross(ax, ch)))
    prims = []
    for i in range(5):
        ang = e2 + D(-38 + 15 * i)
        dd = V(np.cos(ang) * side, np.sin(ang), -0.10 - 0.03 * i); dd /= np.linalg.norm(dd)
        base = el + d2 * (0.22 + 0.04 * i)
        prims.append((base, base + dd * (0.28 - 0.012 * abs(i - 2))))
    return segs, prims


def folded_wing(side):
    """Closed wing hugging the flank: shoulder at the top, tips crossing over the tail."""
    return [(V(0.22 * side, Y0 + 0.42, -0.06), V(0.10, 0.36, 0.25), rx(D(-18)) @ rz(D(-7 * side))),
            (V(0.17 * side, Y0 + 0.14, -0.20), V(0.08, 0.26, 0.15), rx(D(-28)) @ rz(D(-14 * side)))]


def folded_prims(side):
    out = []
    for i in range(3):
        a = V((0.19 - 0.02 * i) * side, Y0 + 0.24 - 0.03 * i, -0.18 - 0.03 * i)
        out.append((a, V((0.06 - 0.02 * i) * side, Y0 - 0.16 - 0.02 * i, -0.36 - 0.02 * i)))
    return out


def snake_path():
    t = np.linspace(0, 1, 15)
    x = 0.30 * np.sin(2.4 * np.pi * t) * (0.35 + t)
    y = Y0 + 0.73 - 0.55 * t - 0.1 * np.sin(np.pi * t)
    z = 0.40 + 0.12 * np.sin(2 * np.pi * t + 0.8) - 0.04 * t
    return np.stack([x, y, z], -1)


SNAKE = snake_path()
SNAKE_HEAD = V(0.13, Y0 + 0.78, 0.40)


def eagle_frame(p):
    """World -> eagle space: talons pinned to the top pad, eagle scaled up."""
    return (p - V(0, Y0 * CS, 0)) / ES + V(0, Y0, 0)


def groups(p):
    """Distance to each part, kept apart so each one can be painted on its own."""
    g = {}
    P = p
    p = P / CS  # nopal, rock and water live in their own scaled frame
    rock = ellipsoid(p, V(0, 0.12, 0), V(0.62, 0.21, 0.48))
    rock += 0.03 * np.sin(9 * p[..., 0]) * np.sin(7 * p[..., 2] + 1) * np.sin(11 * p[..., 1])
    rock = smin(rock, ellipsoid(p, V(0.48, 0.07, 0.3), V(0.18, 0.09, 0.15)), 0.05)
    rock = smin(rock, ellipsoid(p, V(-0.5, 0.06, -0.25), V(0.2, 0.08, 0.16)), 0.05)
    g["rock"] = rock
    g["water"] = ellipsoid(p, V(0, 0.02, 0), V(0.95, 0.03, 0.7))
    cact = cone(p, V(0, 0.2, 0), V(0, 0.5, 0), 0.13, 0.11)
    for c, r, R in PADS:
        cact = smin(cact, ellipsoid(p, c, r, R), 0.04)
    g["cactus"] = cact
    g["tuna"] = np.min([ellipsoid(p, c, V(0.055, 0.075, 0.055)) for c in TUNAS], 0)
    for k in g:
        g[k] = g[k] * CS
    p = eagle_frame(P)

    body = ellipsoid(p, V(0, Y0 + 0.42, -0.03), V(0.28, 0.38, 0.25), rx(D(-12)))
    body = smin(body, ellipsoid(p, V(0, Y0 + 0.50, 0.08), V(0.23, 0.29, 0.20)), 0.08)
    body = smin(body, cone(p, V(0, Y0 + 0.62, 0), V(0, Y0 + 0.78, 0.08), 0.13, 0.11), 0.08)
    head = ellipsoid(p, V(0, Y0 + 0.83, 0.10), V(0.16, 0.15, 0.17))
    head = smin(head, ellipsoid(p, V(0, Y0 + 0.88, 0.14), V(0.13, 0.05, 0.10)), 0.04)  # brow ridge
    body = smin(body, head, 0.06)
    for e in EYES:
        body = np.maximum(body, -sphere(p, e + V(0, 0, 0.02), 0.035))
    for s in (-1, 1):
        body = smin(body, ellipsoid(p, V(0.13 * s, Y0 + 0.2, 0.04), V(0.13, 0.15, 0.13)), 0.06)
    body = smin(body, ellipsoid(p, V(0, Y0 + 0.12, -0.30), V(0.15, 0.30, 0.035), rx(D(-40))), 0.05)
    g["body"] = body

    wing = np.full(p.shape[:-1], 9.0)
    for s in (-1, 1):
        if WINGS == "down":
            for c, r, R in folded_wing(s):
                wing = smin(wing, ellipsoid(p, c, r, R), 0.05)
            for a, b in folded_prims(s):
                wing = smin(wing, cone(p, a, b, 0.045, 0.014), 0.03)
            continue
        segs, prims = wing_frames(s, FLAP)
        for (a, b, ax, ch, n), (L, w) in zip(segs, ((0.27, 0.25), (0.23, 0.20))):
            R = np.stack([ax, ch, n], 1)
            wing = smin(wing, ellipsoid(p, (a + b) / 2, V(L, w, 0.035), R), 0.06)
        for a, b in prims:
            wing = smin(wing, cone(p, a, b, 0.04, 0.012), 0.03)
    g["wing"] = wing * ES

    beak = cone(p, V(0, Y0 + 0.83, 0.22), BEAK_TIP, 0.065, 0.02)
    beak = smin(beak, cone(p, BEAK_TIP, BEAK_TIP + V(0, -0.05, -0.015), 0.02, 0.01), 0.02)
    talon = np.full(p.shape[:-1], 9.0)
    for s in (-1, 1):
        f = V(0.10 * s, Y0 + 0.01, 0.07)
        talon = np.minimum(talon, cone(p, V(0.10 * s, Y0 + 0.13, 0.05), f, 0.035, 0.03))
        for k in (-1, 0, 1):
            talon = np.minimum(talon, cone(p, f, f + V(0.05 * k, -0.02, 0.10), 0.022, 0.008))
        talon = np.minimum(talon, cone(p, f, f + V(0, -0.02, -0.08), 0.02, 0.008))
    g["beak"] = np.minimum(beak, talon) * ES

    sn = ellipsoid(p, SNAKE_HEAD, V(0.065, 0.04, 0.05))
    for i, (a, b) in enumerate(zip(SNAKE[:-1], SNAKE[1:])):
        r = 0.05 - 0.028 * i / len(SNAKE)
        sn = smin(sn, cone(p, a, b, r, r - 0.002), 0.02)
    g["snake"] = sn * ES
    g["body"] = g["body"] * ES
    return g


NAMES = ["rock", "water", "cactus", "tuna", "body", "wing", "beak", "snake"]


def scene(p):
    return np.min(np.stack([groups(p)[k] for k in NAMES]), 0)


# materials: the first ten come in shadow / mid / lit
ROCK, WATER, CACT, TUNA, BROWN, DARK, GOLD, BEAK, SNAKE_G, SNAKE_B = range(10)
SPINE, EYE, PUPIL = 30, 31, 32


def paint(p):
    g = groups(p)
    which = np.argmin(np.stack([g[k] for k in NAMES]), 0)
    P = p
    p = P / CS
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    m = np.full(x.shape, ROCK)
    m = np.where(which == 1, WATER, m)
    # nopal: areoles as a dot lattice drawn in each pad's own plane
    cm = np.full(x.shape, CACT)
    for c, r, R in PADS:
        q = (p - c) @ R
        on = ellipsoid(p, c, r, R) < 0.02
        u, v = q[..., 0] / 0.085, q[..., 1] / 0.1 + 0.5 * (np.round(q[..., 0] / 0.085) % 2)
        dot = np.hypot(u - np.round(u), v - np.round(v)) < 0.24
        cm = np.where(on & dot, SPINE, cm)
    m = np.where(which == 2, cm, m)
    m = np.where(which == 3, TUNA, m)
    p = eagle_frame(P)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    # body: scalloped breast feathers, golden nape and crown, banded tail
    yl = y - Y0
    scallop = np.sin(34 * yl + 5 * np.cos(14 * x)) > 0.7
    bm = np.where(scallop, DARK, BROWN)
    nape = (yl > 0.66) & ~((z > 0.17) & (yl < 0.86))
    bm = np.where(nape, GOLD, bm)
    tail = (yl < 0.3) & (z < -0.15)
    bm = np.where(tail, np.where(np.sin(40 * z) > 0.2, DARK, BROWN), bm)
    for e in EYES:
        d = np.linalg.norm(p - e, axis=-1)
        bm = np.where(d < 0.06, np.where(np.abs(x - e[0]) < 0.018, PUPIL, EYE), bm)
    m = np.where(which == 4, bm, m)
    # wings: rows of coverts along the bone, dark flight feathers at the edge
    wm = np.full(x.shape, BROWN)
    if WINGS == "down":
        yl = y - Y0
        rows_ = np.sin(46 * yl + 9 * z + 4 * np.abs(x)) > 0.5
        wm = np.where(rows_, BROWN, DARK)
        wm = np.where((yl > 0.5) & (z > -0.12), GOLD, wm)  # pale shoulder coverts
        for s in (-1, 1):
            for a, b in folded_prims(s):
                wm = np.where(cone(p, a, b, 0.045, 0.014) < 0.02, DARK, wm)
    for s in (-1, 1) if WINGS == "up" else ():
        segs, prims = wing_frames(s, FLAP)
        side = (x * s) > 0
        for a, b, ax, ch, n in segs:
            cc = (p - a) @ ch
            along = (p - a) @ ax
            row = np.sin(48 * cc + 6 * along) > 0.55
            trailing = cc < -0.09
            seg = np.where(trailing | ~row, DARK, BROWN)
            near = np.linalg.norm(p - (a + b) / 2, axis=-1) < 0.36
            wm = np.where(side & near, seg, wm)
        for a, b in prims:
            wm = np.where(side & (cone(p, a, b, 0.04, 0.012) < 0.015), DARK, wm)
    m = np.where(which == 5, wm, m)
    m = np.where(which == 6, BEAK, m)
    # snake: green with darker bands, the head held in the beak
    seg = np.argmin(np.stack([np.linalg.norm(p - c, axis=-1) for c in SNAKE]), 0)
    band = np.sin(3.4 * seg + 18 * y) > 0.3
    sm = np.where(band, SNAKE_B, SNAKE_G)
    m = np.where(which == 7, sm, m)
    return m, which


def normal(p, e=1e-3):
    n = np.stack([scene(p + V(*o)) - scene(p - V(*o)) for o in np.eye(3) * e], -1)
    return np.nan_to_num(n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9))


def frame(theta):
    xs = (np.arange(COLS) + 0.5 - COLS / 2) / COLS * SPAN
    ys = CY - (np.arange(ROWS) + 0.5 - ROWS / 2) / ROWS * SPAN * (ROWS * LH) / (COLS * CW)
    X, Y = np.meshgrid(xs, ys)
    c, s = np.cos(theta), np.sin(theta)
    tilt = 0.22
    fwd = V(-s * np.cos(tilt), -np.sin(tilt), -c * np.cos(tilt))
    right = V(c, 0, -s)
    up = np.cross(right, fwd)
    ro = X[..., None] * right + (Y[..., None] - CY) * up + V(0, CY, 0) - fwd * 5
    t = np.zeros(X.shape)
    hit = np.zeros(X.shape, bool)
    for _ in range(110):
        d = scene(ro + fwd * t[..., None])
        hit |= d < 1e-3
        t = np.where(hit | (t > 10), t, t + d * 0.8)
    p = ro + fwd * t[..., None]
    n = normal(p)
    light = V(-0.5, 0.8, 0.35); light /= np.linalg.norm(light)
    rot_light = V(light[0] * c + light[2] * s, light[1], -light[0] * s + light[2] * c)
    diff = np.clip(n @ rot_light, 0, 1)
    rim = (1 - np.abs(n @ fwd)) ** 3
    lit = np.clip(0.26 + 0.74 * diff - 0.25 * rim, 0, 1)
    ink = 0.3 + 0.7 * lit
    idx = np.where(hit, 1 + np.nan_to_num(ink * (len(RAMP) - 2)).round().astype(int), 0)
    mat, which = paint(p)
    shade = np.digitize(lit, [0.22, 0.5])
    cls = np.where(mat < 10, mat * 3 + shade, mat)
    out = []
    for r in range(ROWS):
        row = []
        for k in range(COLS):
            ch = RAMP[idx[r, k]]
            if ch != " ":
                mk = mat[r, k]
                if mk == WATER:
                    ch = "~" if np.sin(9 * p[r, k, 0] + 13 * p[r, k, 2] + 2 * theta) > -0.2 else "-"
                elif mk == SPINE:
                    ch = "'" if lit[r, k] < 0.5 else "*"
                elif mk in (EYE, PUPIL):
                    ch = "@" if mk == PUPIL else "o"
            row.append((ch, int(cls[r, k])))
        out.append(row)
    return out


# class -> (light theme, dark theme)
SHADED = [  # shadow, mid, lit
    [("#4f4a44", "#5d574f"), ("#6e675e", "#8a8378"), ("#8f877b", "#bdb5a7")],   # rock
    [("#1d4e89", "#2f6db5"), ("#2f6db5", "#4c93e0"), ("#4c93e0", "#8cc4ff")],   # water
    [("#1d5e26", "#2a7d34"), ("#2e8a38", "#47b84a"), ("#4fae46", "#86e06a")],   # nopal
    [("#8e1240", "#b02a5c"), ("#c2185b", "#ef4f88"), ("#e0457a", "#ff8fb4")],   # tuna
    [("#4e2c10", "#7a4a22"), ("#6e4220", "#a26a34"), ("#8c5a2c", "#c88e50")],   # brown
    [("#24160a", "#4a3424"), ("#3a2512", "#634630"), ("#52361c", "#82624a")],   # dark feathers
    [("#8a5a1e", "#b07a34"), ("#b07a30", "#d9a552"), ("#c99440", "#f2c87c")],   # golden nape
    [("#b08a0a", "#c99a14"), ("#e0a812", "#f2be30"), ("#f4c830", "#ffe070")],   # beak & talons
    [("#4f6b10", "#6e9020"), ("#7a9a1a", "#a8c83a"), ("#98b82a", "#d4ee6a")],   # snake
    [("#2e3e08", "#4a6012"), ("#44580c", "#6a8a1e"), ("#5a7212", "#8aaa2e")],   # snake bands
]
PALETTE = [c for mat in SHADED for c in mat] + [
    ("#c8b47a", "#f2e6b8"),  # spines
    ("#c77a0a", "#ffb02e"),  # amber eye
    ("#111111", "#0d0d0d")]  # pupil


def svg(only=None):
    W = int(COLS * CW) + 80
    H = int(ROWS * LH) + 110
    ox, oy = 40, 40
    anim = only is None
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           "<style>",
           ":root{--ink:#1b1b1b;--pen:#1f7a3a;--faint:#1b1b1b33}",
           "@media (prefers-color-scheme:dark){:root{--ink:#e8e6e1;--pen:#5fd27a;--faint:#e8e6e133}}",
           f"text{{font:{FONT}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:var(--ink);white-space:pre}}",
           ".m{font-size:10px;letter-spacing:.12em;fill:var(--pen)}",
           ".l{stroke:var(--pen);stroke-width:.8;fill:none}",
           ".h{stroke:var(--faint);stroke-width:.6}",
           "".join(f".c{i}{{fill:{a}}}" for i, (a, _) in enumerate(PALETTE)),
           "@media (prefers-color-scheme:dark){" + "".join(f".c{i}{{fill:{b}}}" for i, (_, b) in enumerate(PALETTE)) + "}",
           f"g.f{{visibility:hidden;animation:k {SECONDS}s step-end infinite}}" if anim else "",
           f"@keyframes k{{0%{{visibility:visible}}{100 / FRAMES:.4f}%,100%{{visibility:hidden}}}}" if anim else "",
           "</style>"]
    out.append(f'<rect class="l" x="{ox - 16}" y="{oy - 16}" width="{W - 2 * ox + 32}" height="{H - 2 * oy + 2}" stroke-dasharray="2 4"/>')
    for x, y, dx, dy in [(ox - 24, oy - 24, 1, 1), (W - ox + 24, oy - 24, -1, 1),
                         (ox - 24, H - oy + 10, 1, -1), (W - ox + 24, H - oy + 10, -1, -1)]:
        out.append(f'<path class="l" d="M{x} {y + 14 * dy}V{y}H{x + 14 * dx}"/>')
    hv = SPAN * (ROWS * LH) / (COLS * CW) / 2  # half the world height on the page
    cx, cy = W / 2, oy + (CY + hv) / (2 * hv) * ROWS * LH + 2  # pinned under the water line
    rx_, ry_ = 0.9 * CS / SPAN * COLS * CW, 14
    out.append(f'<clipPath id="s"><ellipse cx="{cx}" cy="{cy}" rx="{rx_}" ry="{ry_}"/></clipPath><g clip-path="url(#s)">')
    out += [f'<line class="h" x1="{x:.1f}" y1="{cy - ry_}" x2="{x + 30:.1f}" y2="{cy + ry_}"/>'
            for x in np.arange(cx - rx_ - 30, cx + rx_, 5)]
    out.append("</g>")
    out.append(f'<ellipse class="l" cx="{cx}" cy="{cy}" rx="{rx_}" ry="{ry_}" stroke-dasharray="1 3"/>')
    out.append(f'<text class="m" x="{ox}" y="{H - oy + 4}">viva méxico</text>')
    idxs = list(range(FRAMES)) if anim else [only]
    thetas = [2 * np.pi * i / FRAMES for i in idxs]
    with Pool() as pool:  # frames are independent, render them in parallel
        rendered = pool.map(frame, thetas)
    for i, theta, rows in zip(idxs, thetas, rendered):
        out.append(f'<g class="f" style="animation-delay:{SECONDS * i / FRAMES:.3f}s">' if anim else "<g>")
        for r, row in enumerate(rows):
            cells = [j for j, (ch, _) in enumerate(row) if ch != " "]
            if not cells:
                continue
            lead, end = cells[0], cells[-1] + 1
            runs, prev = [], None
            for ch, k in row[lead:end]:
                ch = ch.replace("&", "&amp;").replace("<", "&lt;")
                if ch != " " and k != prev:
                    runs.append([k, ""])
                    prev = k
                runs[-1][1] += ch
            spans = "".join(f'<tspan class="c{k}">{s}</tspan>' for k, s in runs)
            out.append(f'<text x="{ox + lead * CW:.1f}" y="{oy + (r + 1) * LH:.1f}" '
                       f'textLength="{(end - lead) * CW:.1f}" lengthAdjust="spacingAndGlyphs">{spans}</text>')
        out.append(f'<text class="m" x="{W - ox}" y="{H - oy + 4}" text-anchor="end">θ = {round(np.degrees(theta)):03d}°</text>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    print(svg(int(sys.argv[1]) if len(sys.argv) > 1 else None))
