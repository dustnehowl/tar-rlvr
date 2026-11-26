"""Stage 1b: CoVA GT 첫 이미지가 '상태만' 담고 있는지 픽셀로 확인 (OCR 없음, CPU).

maze  : 격자를 읽어 문제의 미로와 칸 단위로 같은지, 경로/탐색 표시 색(파랑·주황·분홍)이 없는지.
clutrr: 이름 상자를 찾아 질문의 이름과 맞는지 (닫힌 어휘 템플릿 매칭), 빨간 라벨 상자(= 새로 그린 관계) 개수,
        두 질의 인물(초록 상자)이 서로 이어졌는지.
  - 그림 렌더 색 (CoVA 고정): 질의 인물 채움 (187,247,208), 새 관계 인물 채움 (254,226,226), 빨강 (220,38,38).
  - 이름 매칭: Pillow 기본 글꼴로 이름을 그려 굵게 만든 뒤 상자 글자와 상관계수. 글꼴이 달라 (원본은 DejaVu Sans Bold)
    완벽하지 않다 → 상자마다 margin = (질문 안 이름 최고점) − (질문 밖 이름 최고점) 의 평균으로
    match / mismatch / uncertain 세 단계로 나누고, uncertain 은 표본에서 뺀다.
  - 손으로 확인한 24 행에서: mismatch 7 행은 평균 margin ≤ −0.29, match 17 행은 ≥ −0.17 (stage1b_prep 로그 참고).
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

# ------------------------------------------------------------------ maze


def _rgb(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)


def maze_read_grid(png: bytes, W: int, H: int):
    """→ (rows, n_unknown_cells, marks). rows 는 '1'(벽) '0'(열림) 'S' 'E' '?' 로 된 H 개 문자열."""
    a = _rgb(png)
    ys, xs = np.where(a.min(-1) < 250)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cw, ch = (x1 - x0 + 1) / W, (y1 - y0 + 1) / H
    rows, unk = [], 0
    for r in range(H):
        s = ""
        for c in range(W):
            R, G, B = a[int(y0 + (r + 0.25) * ch), int(x0 + (c + 0.25) * cw)]   # S/E 글자를 피해 칸의 1/4 지점
            if R + G + B < 200:
                s += "1"
            elif min(R, G, B) > 225 and max(R, G, B) - min(R, G, B) < 12:
                s += "0"
            elif G > 130 and R < 80 and B < 120:
                s += "S"
            elif R > 190 and G < 80 and B < 80:
                s += "E"
            else:
                s += "?"; unk += 1
        rows.append(s)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    marks = dict(blue=float(((B > R + 40) & (B > G + 20)).mean()),
                 orange=float(((R > 200) & (G > 100) & (G < 190) & (B < 90)).mean()),
                 pink=float(((R > 200) & (B > 150) & (G < 150)).mean()))
    return rows, unk, marks


def maze_check(png: bytes, raw: str):
    """→ (state_only: bool, reason)."""
    R = raw.strip().split("\n")
    rows, unk, marks = maze_read_grid(png, len(R[0]), len(R))
    if any(v > 1e-4 for v in marks.values()):
        return False, "path_or_search_marks " + ",".join(k for k, v in marks.items() if v > 1e-4)
    if unk:
        return False, f"unknown_cells_{unk}"
    if rows != R:
        return False, "grid_differs_from_question"
    return True, ""


# ------------------------------------------------------------------ clutrr
FILL_QUERY, FILL_NEW, RED = (187, 247, 208), (254, 226, 226), (220, 38, 38)
_font = ImageFont.load_default(size=48)
_cache: dict = {}


def _trim(m):
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _vec(m, H=20, W=80):
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
    v = np.asarray(im, float)
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def _template(name):
    if name not in _cache:
        im = Image.new("L", (48 * len(name) + 20, 80), 0)
        ImageDraw.Draw(im).text((10, 10), name, fill=255, font=_font)
        m = _trim(ndimage.binary_dilation(np.asarray(im) > 128, iterations=2))   # Bold 굵기 흉내
        _cache[name] = (_vec(m), m.shape[1] / m.shape[0])
    return _cache[name]


def _name_scores(ink, names):
    v, ar = _vec(ink), ink.shape[1] / ink.shape[0]
    out = {}
    for n in names:
        tv, tar = _template(n)
        out[n] = float((v * tv).sum()) - 0.6 * abs(np.log(ar / tar))
    return out


def _boxes(a):
    out = []
    for kind, col in (("query", FILL_QUERY), ("new", FILL_NEW)):
        lab, _ = ndimage.label(np.abs(a - np.array(col)).max(-1) <= 6)
        for sl in ndimage.find_objects(lab):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if h < 8 or w < 15 or h > 120:
                continue
            dark = a[sl].sum(-1) < 450
            dark[:3] = dark[-3:] = False
            dark[:, :3] = dark[:, -3:] = False
            ink = _trim(dark)
            if ink is None or ink.shape[0] < 5:
                continue
            out.append(dict(kind=kind, sl=sl, ink=ink))
    return out


def _label_boxes(a):
    """흰 바탕 작은 라벨 상자 → 테두리 색으로 red(새 관계) / gray(이전 관계) 분류."""
    white = a.min(-1) >= 250
    lab, n = ndimage.label(white)
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    bg = int(np.argmax(sizes)) + 1
    # 테두리는 1px + 안티앨리어싱이라 색이 옅다 → '붉은 기' / '회색 기' 로 판정, 1–3px 바깥 고리 중 최대값
    R_, G_, B_ = a[..., 0], a[..., 1], a[..., 2]
    red = (R_ - G_ > 40) & (R_ - B_ > 40)
    gray = (np.abs(R_ - B_) < 25) & (np.abs(R_ - G_) < 25) & (a.max(-1) < 245)
    res = {"red": 0, "gray": 0}
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if i == bg or sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (6 <= h <= 40 and 15 <= w <= 160):
            continue
        fr, fg = 0.0, 0.0
        for d in (1, 2, 3):
            y0, y1, x0, x1 = max(sl[0].start - d, 0), sl[0].stop + d, max(sl[1].start - d, 0), sl[1].stop + d
            ring = np.zeros((y1 - y0, x1 - x0), bool)
            ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
            fr = max(fr, red[y0:y1, x0:x1][ring].mean())
            fg = max(fg, gray[y0:y1, x0:x1][ring].mean())
        if fr > 0.6:
            res["red"] += 1
        elif fg > 0.6:
            res["gray"] += 1
    return res


def clutrr_check(png: bytes, q_names: set, pool: list, query: tuple):
    """→ dict(names=match|mismatch|uncertain, margin, n_red_labels, n_gray_labels, query_pair_edge, state_only, reason)."""
    a = _rgb(png)
    bx = _boxes(a)
    margins, reads = [], []
    for b in bx:
        sc = _name_scores(b["ink"], pool)
        inq = max((s, n) for n, s in sc.items() if n in q_names)
        outq = max(s for n, s in sc.items() if n not in q_names)
        margins.append(inq[0] - outq)
        reads.append((b["kind"], inq[1]))
    m = float(np.mean(margins)) if margins else -9.0
    names = "match" if m >= -0.12 else ("mismatch" if m <= -0.25 else "uncertain")
    lb = _label_boxes(a)
    # 두 질의 인물(초록 상자)이 모두 빨간 선에 닿으면 질의 쌍 사이의 관계를 그린 것
    red = np.abs(a - np.array(RED)).max(-1) <= 40
    touch = 0
    for b in bx:
        if b["kind"] != "query":
            continue
        sl = b["sl"]
        y0, y1, x0, x1 = max(sl[0].start - 8, 0), sl[0].stop + 8, max(sl[1].start - 8, 0), sl[1].stop + 8
        touch += red[y0:y1, x0:x1].sum() > 20
    reason = []
    if names != "match":
        reason.append(f"names_{names}")
    if lb["red"] != 1:
        reason.append(f"red_labels_{lb['red']}")
    if lb["gray"]:
        reason.append(f"gray_labels_{lb['gray']}")
    if touch >= 2:
        reason.append("query_pair_edge")
    return dict(names=names, margin=m, reads=reads, n_red_labels=lb["red"], n_gray_labels=lb["gray"],
                query_pair_edge=touch >= 2, state_only=not reason, reason=",".join(reason))
