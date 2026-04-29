"""End-to-end real-video collision-prediction demo with the dashboard CCTV
overlay style requested by the instructor mockup:
  - FKL_xxx / P_xxxx multi-line labels
  - solid-blue actual path + dashed-blue 2.5 s predicted path
  - red filled "위험 영역" polygon in front of moving forklifts
  - ✕ collision-prediction point
  - bottom-right "충돌 위험!" panel with TTC + Risk Score
  - top-left 4-item legend, top-right CAM label / time

Stack: YOLO (Ultralytics) for detection, Deep SORT (deep-sort-realtime)
for persistent ID tracking — one tracker per class to avoid ID collision.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PERSON_CLASS_NAMES = {"person"}
FORKLIFT_CLASS_NAMES = {"forklift", "fork lift", "fork-lift"}

REACTION_S = 1.0
BRAKE_DECEL = 1.5
MIN_R = 1.5
HORIZONS = (1.0, 2.0, 3.0)
PRED_T = 2.5  # seconds for path prediction & danger zone

KFONT = r"C:\Windows\Fonts\malgun.ttf"
KFONT_BD = r"C:\Windows\Fonts\malgunbd.ttf"

# RGB colors (mockup palette)
COLOR_FK_DANGER = (239, 68, 68)
COLOR_FK_CAUTION = (250, 204, 21)
COLOR_FK_NORMAL = (148, 163, 184)
COLOR_WORKER = (96, 165, 250)
COLOR_TRAJ = (96, 165, 250)
COLOR_DANGER_ZONE = (239, 68, 68)
COLOR_CROSS = (239, 68, 68)
COLOR_PANEL = (15, 23, 42)
COLOR_TEXT = (226, 232, 240)
COLOR_TEXT_DIM = (148, 163, 184)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    p = KFONT_BD if bold else KFONT
    if Path(p).exists():
        return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def class_role(name: str) -> str | None:
    n = name.lower()
    if n in PERSON_CLASS_NAMES:
        return "person"
    if n in FORKLIFT_CLASS_NAMES:
        return "forklift"
    return None


def pseudo_meter_per_px(y_pixel: float, img_h: int) -> float:
    return max(0.05, (img_h - y_pixel) / img_h)


def adaptive_radius(v: float) -> float:
    v = max(0.0, v)
    return MIN_R + REACTION_S * v + (v ** 2) / (2 * BRAKE_DECEL)


def predict_distance(p1, p2, v1, v2, T):
    fx = (p1[0] + v1[0] * T) - (p2[0] + v2[0] * T)
    fy = (p1[1] + v1[1] * T) - (p2[1] + v2[1] * T)
    return math.hypot(fx, fy)


def is_likely_forklift_box(x1, y1, x2, y2, w_img, h_img,
                           max_aspect=1.8, max_area_frac=0.45,
                           min_area=200) -> bool:
    """Heuristic: drop bboxes that are too tall+narrow (rack columns), too
    small, or covering most of the frame — common keremberke FP modes."""
    bw = x2 - x1; bh = y2 - y1
    if bw < 10 or bh < 10:
        return False
    if bh / max(1.0, bw) > max_aspect:
        return False
    if bw * bh > max_area_frac * w_img * h_img:
        return False
    if bw * bh < min_area:
        return False
    return True


def is_static_track(history, fps, min_seconds=1.5, max_disp_px=8.0) -> bool:
    """A track that has existed for >= min_seconds but moved < max_disp_px
    total — almost certainly a stationary false-positive (rack/pallet)."""
    if len(history) < int(fps * min_seconds):
        return False
    xs = [p[1] for p in history]
    ys = [p[2] for p in history]
    return (max(xs) - min(xs) < max_disp_px) and (max(ys) - min(ys) < max_disp_px)


def draw_dashed(draw: ImageDraw.ImageDraw, p1, p2, color, width=2, dash=10, gap=6):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    d = 0.0
    drawing = True
    rgba = color + (255,) if len(color) == 3 else color
    while d < L:
        seg = dash if drawing else gap
        nd = min(L, d + seg)
        if drawing:
            draw.line([(x1 + ux * d, y1 + uy * d),
                       (x1 + ux * nd, y1 + uy * nd)], fill=rgba, width=width)
        d = nd
        drawing = not drawing


def draw_arrow_head(draw, p_from, p_to, color, size=10):
    x1, y1 = p_from; x2, y2 = p_to
    angle = math.atan2(y2 - y1, x2 - x1)
    a1 = angle + math.radians(150); a2 = angle - math.radians(150)
    pts = [(x2, y2),
           (x2 + size * math.cos(a1), y2 + size * math.sin(a1)),
           (x2 + size * math.cos(a2), y2 + size * math.sin(a2))]
    draw.polygon(pts, fill=color + (255,))


def draw_x_marker(draw, center, size=18, color=(239, 68, 68), width=4):
    cx, cy = center
    s = size / 2
    draw.line([(cx - s, cy - s), (cx + s, cy + s)], fill=color + (255,), width=width)
    draw.line([(cx + s, cy - s), (cx - s, cy + s)], fill=color + (255,), width=width)


def danger_polygon(anchor_px, v_px, T=PRED_T, half_w_near=28,
                   half_w_far_min=70, half_w_far_per_speed=6, max_len=320):
    cx, cy = anchor_px
    vx, vy = v_px
    v_mag = math.hypot(vx, vy)
    if v_mag < 5:
        return None
    ux, uy = vx / v_mag, vy / v_mag
    L = min(v_mag * T, max_len)
    half_w_far = half_w_far_min + min(120, v_mag * half_w_far_per_speed * 0.05)
    px, py = -uy, ux
    near_l = (cx - px * half_w_near, cy - py * half_w_near)
    near_r = (cx + px * half_w_near, cy + py * half_w_near)
    far_c = (cx + ux * L, cy + uy * L)
    far_l = (far_c[0] - px * half_w_far, far_c[1] - py * half_w_far)
    far_r = (far_c[0] + px * half_w_far, far_c[1] + py * half_w_far)
    return [near_l, near_r, far_r, far_l]


def annotate_dashboard(
    frame_bgr: np.ndarray,
    per_track_state: dict,
    alarm_pairs: list[dict],
    track_pix_history: dict,
    frame_idx: int, fps: float, w: int, h: int,
    cam_label: str, scale_speed_kmh: float,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil, "RGBA")

    f_lbl_id = get_font(12, bold=True)
    f_lbl = get_font(11)
    f_zone = get_font(16, bold=True)
    f_legend_title = get_font(12, bold=True)
    f_legend = get_font(11)
    f_cam = get_font(14, bold=True)
    f_alarm_t = get_font(16, bold=True)
    f_alarm_b = get_font(12, bold=True)
    f_alarm_s = get_font(11)
    f_time = get_font(13, bold=True)
    header_h_offset = 36  # bottom of top header bar; labels avoid covering it

    # ── 1. danger zones (drawn first, under everything) ──
    drawn_zones = set()
    for ap in alarm_pairs:
        fk_id = ap["fk_id"]
        if fk_id in drawn_zones:
            continue
        drawn_zones.add(fk_id)
        fk = per_track_state[fk_id]
        poly = danger_polygon(fk["anchor_floor"], fk["v_px"])
        if poly is None:
            continue
        draw.polygon(poly, fill=COLOR_DANGER_ZONE + (90,))
        for i in range(len(poly)):
            draw_dashed(draw, poly[i], poly[(i + 1) % len(poly)],
                        COLOR_DANGER_ZONE, width=2, dash=8, gap=5)
        cx_p = sum(p[0] for p in poly) / 4
        cy_p = sum(p[1] for p in poly) / 4
        txt = "위험 영역"
        bb = draw.textbbox((0, 0), txt, font=f_zone)
        tw = bb[2] - bb[0]; th = bb[3] - bb[1]
        draw.text((cx_p - tw / 2, cy_p - th / 2), txt,
                  font=f_zone, fill=COLOR_DANGER_ZONE + (255,))

    # ── 2. trajectory history (solid blue) + prediction (dashed blue) ──
    for tid, st in per_track_state.items():
        history = list(track_pix_history.get(tid, []))
        if len(history) >= 2:
            pts = [(h2[1], h2[2]) for h2 in history[-30:]]
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]],
                          fill=COLOR_TRAJ + (200,), width=2)
        v_px = st["v_px"]
        v_mag = math.hypot(*v_px)
        if v_mag > 5:
            cx, cy = st["anchor_floor"]
            ex = cx + v_px[0] * PRED_T
            ey = cy + v_px[1] * PRED_T
            draw_dashed(draw, (cx, cy), (ex, ey), COLOR_TRAJ,
                        width=2, dash=9, gap=5)
            draw_arrow_head(draw, (cx, cy), (ex, ey), COLOR_TRAJ, size=8)

    # ── 3. collision X markers ──
    drawn_x = set()
    for ap in alarm_pairs:
        key = (ap["fk_id"], ap["ps_id"])
        if key in drawn_x:
            continue
        drawn_x.add(key)
        cx, cy = ap["collision_x_pix"]
        draw_x_marker(draw, (cx, cy), size=18, color=COLOR_CROSS, width=4)

    # ── 4. bboxes + labels ──
    alarmed_fk = {ap["fk_id"] for ap in alarm_pairs}
    for tid, st in per_track_state.items():
        x1, y1, x2, y2 = [int(v) for v in st["bbox"]]
        if st["role"] == "forklift":
            v_mag = math.hypot(*st["v_px"])
            if tid in alarmed_fk:
                color = COLOR_FK_DANGER
            elif v_mag > 8:
                color = COLOR_FK_CAUTION
            else:
                color = COLOR_FK_NORMAL
            id_short = (tid - 10000) % 1000 if tid >= 10000 else tid % 1000
            seq = (tid - 10000) % 1000 if tid >= 10000 else tid
            line1 = f"FKL_{seq:03d}"
        else:
            color = COLOR_WORKER
            seq = tid
            id_short = tid
            line1 = f"P_{seq:04d}"
        line2 = f"ID: {id_short}"
        line3 = f"{st['speed_kmh']:.1f} km/h"

        for k in range(3):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k],
                           outline=color + (max(60, 255 - k * 60),))

        # label panel above bbox: forklift inherits its bbox color, person uses dark
        if st["role"] == "forklift":
            label_fill = color + (240,)
        else:
            label_fill = (15, 23, 42, 235)

        lines = [line1, line2, line3]
        widths = [draw.textbbox((0, 0), s, font=(f_lbl_id if i == 0 else f_lbl))[2]
                  for i, s in enumerate(lines)]
        heights = [draw.textbbox((0, 0), s, font=(f_lbl_id if i == 0 else f_lbl))[3]
                   for i, s in enumerate(lines)]
        tw = max(widths) + 12
        th = sum(heights) + 8
        ly1 = max(header_h_offset, y1 - th - 4)
        draw.rounded_rectangle([x1, ly1, x1 + tw, ly1 + th],
                               radius=4, fill=label_fill)
        ty = ly1 + 3
        for i, s in enumerate(lines):
            f = f_lbl_id if i == 0 else f_lbl
            draw.text((x1 + 6, ty), s, font=f, fill=(255, 255, 255))
            ty += heights[i] + (1 if i == 0 else 0)

    # ── 5. top header bar: CAM label (left) + 실시간 / AI 분석 ON (right) ──
    header_h = 36
    draw.rectangle([0, 0, w, header_h], fill=COLOR_PANEL + (235,))
    draw.text((14, 9), cam_label, font=f_cam, fill=COLOR_TEXT + (255,))

    # right-side badges
    badge_pad = 8
    badges = [("실시간", (34, 197, 94)), ("AI 분석 ON", (59, 130, 246))]
    bx = w - 14
    for txt, c in reversed(badges):
        bb = draw.textbbox((0, 0), txt, font=f_legend_title)
        tw = bb[2] - bb[0]; th = bb[3] - bb[1]
        bw = tw + 22
        bh = 22
        bx1 = bx - bw; by1 = 7
        draw.rounded_rectangle([bx1, by1, bx, by1 + bh],
                               radius=4, fill=c + (60,),
                               outline=c + (255,), width=1)
        draw.ellipse([bx1 + 5, by1 + bh / 2 - 4, bx1 + 13, by1 + bh / 2 + 4],
                     fill=c + (255,))
        draw.text((bx1 + 17, by1 + 3), txt, font=f_legend_title, fill=c + (255,))
        bx = bx1 - badge_pad

    # ── 6. legend (top-left) ──
    legend_items = [
        ("실제 이동 경로", "solid", COLOR_TRAJ),
        ("예측 이동 경로 (2.5s)", "dashed", COLOR_TRAJ),
        ("위험 영역", "fill", COLOR_DANGER_ZONE),
        ("충돌 예측 지점", "x", COLOR_CROSS),
    ]
    leg_x = 10; leg_y = 46
    leg_w = 200; leg_h = 110
    draw.rounded_rectangle([leg_x, leg_y, leg_x + leg_w, leg_y + leg_h],
                           radius=8, fill=COLOR_PANEL + (220,),
                           outline=(71, 85, 105, 255), width=1)
    for i, (txt, kind, c) in enumerate(legend_items):
        ly = leg_y + 12 + i * 24
        ix = leg_x + 12
        if kind == "solid":
            draw.line([(ix, ly + 7), (ix + 30, ly + 7)],
                      fill=c + (255,), width=2)
        elif kind == "dashed":
            draw_dashed(draw, (ix, ly + 7), (ix + 30, ly + 7),
                        c, width=2, dash=5, gap=3)
        elif kind == "fill":
            draw.rectangle([ix + 2, ly + 2, ix + 22, ly + 14],
                           fill=c + (110,), outline=c + (255,))
        elif kind == "x":
            draw_x_marker(draw, (ix + 12, ly + 8), size=12, color=c, width=2)
        draw.text((leg_x + 50, ly), txt, font=f_legend, fill=COLOR_TEXT + (255,))

    # ── 7. bottom-LEFT alarm panel (per mockup) ──
    if alarm_pairs:
        top = max(alarm_pairs, key=lambda a: a["risk_score"])
        fk_seq = (top["fk_id"] - 10000) % 1000 if top["fk_id"] >= 10000 else top["fk_id"]
        title = "충돌 위험!"
        body = f"FKL_{fk_seq:03d} (ID:{fk_seq})  ↔  P_{top['ps_id']:04d} (ID:{top['ps_id']})"
        stats = f"TTC {top['ttc_s']:.1f} sec   |   Risk Score {int(top['risk_score'])}%"

        bb_t = draw.textbbox((0, 0), title, font=f_alarm_t)
        bb_b = draw.textbbox((0, 0), body, font=f_alarm_b)
        bb_s = draw.textbbox((0, 0), stats, font=f_alarm_s)
        tw = max(bb_t[2], bb_b[2], bb_s[2]) + 60
        th = (bb_t[3] - bb_t[1]) + (bb_b[3] - bb_b[1]) + (bb_s[3] - bb_s[1]) + 32

        bx1 = 16; by1 = h - th - 16
        bx2 = bx1 + tw; by2 = h - 16
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=10,
                               fill=(15, 23, 42, 235),
                               outline=COLOR_DANGER_ZONE + (255,), width=2)
        # red left accent bar
        draw.rounded_rectangle([bx1, by1, bx1 + 5, by2], radius=2,
                               fill=COLOR_DANGER_ZONE + (255,))
        # warning triangle
        tri_x = bx1 + 24; tri_y = by1 + 16
        s = 12
        draw.polygon([(tri_x, tri_y), (tri_x + s, tri_y + s * 1.6),
                      (tri_x - s, tri_y + s * 1.6)],
                     fill=COLOR_DANGER_ZONE + (255,))
        draw.text((tri_x - 3, tri_y + 6), "!", font=f_alarm_b,
                  fill=(255, 255, 255))
        draw.text((bx1 + 50, by1 + 8), title, font=f_alarm_t,
                  fill=COLOR_DANGER_ZONE + (255,))
        draw.text((bx1 + 50, by1 + 32), body, font=f_alarm_b,
                  fill=COLOR_TEXT + (255,))
        draw.text((bx1 + 50, by1 + 52), stats, font=f_alarm_s,
                  fill=COLOR_TEXT_DIM + (255,))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./outputs/realvideo_alarm.mp4"))
    ap.add_argument("--tracks-out", type=Path, default=Path("./outputs/realvideo_tracks.csv"))
    ap.add_argument("--person-weights", default="yolo11n.pt")
    ap.add_argument("--forklift-weights", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--cam-label", default="CAM 01 - 창고")
    ap.add_argument("--scale-speed-kmh", type=float, default=0.6,
                    help="display-only multiplier from pixel-velocity to km/h")
    args = ap.parse_args()

    from deep_sort_realtime.deepsort_tracker import DeepSort

    model_p = load_model(args.person_weights)
    model_f = load_model(args.forklift_weights) if args.forklift_weights and args.forklift_weights != args.person_weights else None
    combined = model_f is None and any(class_role(n) for n in model_p.names.values())
    print(f"using {'combined' if combined else 'person+forklift'} model. classes: {model_p.names}"
          + (f"  forklift_model: {model_f.names}" if model_f else ""))

    # Two Deep SORT trackers (one per class) to keep IDs disjoint
    tracker_p = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.4)
    tracker_f = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.4)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {width}x{height}@{fps:.1f}fps  {n_total} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=8))
    track_pix_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
    rows = []
    n_max = int(args.max_seconds * fps) if args.max_seconds else n_total

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx + 1 >= n_max:
            break
        frame_idx += 1

        # ── YOLO detection (no built-in tracker) ──
        p_dets, f_dets = [], []
        models_used = [(model_p, "all" if combined else "person")]
        if model_f is not None:
            models_used.append((model_f, "forklift"))
        for model, role_filter in models_used:
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.int().cpu().tolist()
            names = res.names
            for box, c, cl in zip(xyxy, confs, cls):
                cname = class_role(names[cl])
                if cname is None and names[cl].lower() == "truck" and model_f is None and not combined:
                    cname = "forklift"
                if cname is None:
                    continue
                if role_filter != "all" and cname != role_filter:
                    continue
                x1, y1, x2, y2 = box
                # detection-time shape filter for forklift FPs
                if cname == "forklift" and not is_likely_forklift_box(
                        x1, y1, x2, y2, width, height):
                    continue
                bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                if cname == "person":
                    p_dets.append((bbox_xywh, float(c), "person"))
                else:
                    f_dets.append((bbox_xywh, float(c), "forklift"))

        # ── Deep SORT tracking ──
        p_tracks = tracker_p.update_tracks(p_dets, frame=frame)
        f_tracks = tracker_f.update_tracks(f_dets, frame=frame)

        def sanitize(ltrb):
            x1, y1, x2, y2 = [float(v) for v in ltrb]
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            x1 = max(0.0, min(width - 1, x1))
            x2 = max(0.0, min(width - 1, x2))
            y1 = max(0.0, min(height - 1, y1))
            y2 = max(0.0, min(height - 1, y2))
            return [x1, y1, x2, y2]

        tracks_now = []
        for t in p_tracks:
            if not t.is_confirmed():
                continue
            box = sanitize(t.to_ltrb())
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                continue
            tracks_now.append((int(t.track_id), "person", box))
        for t in f_tracks:
            if not t.is_confirmed():
                continue
            box = sanitize(t.to_ltrb())
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                continue
            tracks_now.append((int(t.track_id) + 10000, "forklift", box))

        per_track_state = {}
        for tid, role, (x1, y1, x2, y2) in tracks_now:
            cx_px = (x1 + x2) / 2.0
            cy_floor = y2  # bbox bottom-center
            scale = pseudo_meter_per_px(cy_floor, height)
            wx = cx_px * scale
            wy = (height - cy_floor) * scale
            track_history[tid].append((frame_idx, wx, wy))
            track_pix_history[tid].append((frame_idx, cx_px, cy_floor))
            if len(track_history[tid]) >= 2:
                f0, x0, y0 = track_history[tid][0]
                f1, x1w, y1w = track_history[tid][-1]
                dt = max((f1 - f0) / fps, 1e-3)
                vx = (x1w - x0) / dt; vy = (y1w - y0) / dt
            else:
                vx = vy = 0.0
            if len(track_pix_history[tid]) >= 2:
                f0, px0, py0 = track_pix_history[tid][0]
                f1, px1, py1 = track_pix_history[tid][-1]
                dtp = max((f1 - f0) / fps, 1e-3)
                vpx = (px1 - px0) / dtp; vpy = (py1 - py0) / dtp
            else:
                vpx = vpy = 0.0
            v_pix_mag = math.hypot(vpx, vpy)
            speed_kmh = v_pix_mag * args.scale_speed_kmh
            per_track_state[tid] = {
                "role": role, "wx": wx, "wy": wy,
                "vx": vx, "vy": vy, "v_px": (vpx, vpy),
                "speed_kmh": speed_kmh,
                "bbox": (x1, y1, x2, y2),
                "anchor_floor": (cx_px, cy_floor),
            }

        # Drop forklift tracks that look like rack/pallet false positives:
        #   (a) currently has a tall/narrow / oversized bbox shape, or
        #   (b) has been stationary for >= 1.5 s
        bad_fks = set()
        for tid, st in per_track_state.items():
            if st["role"] != "forklift":
                continue
            x1, y1, x2, y2 = st["bbox"]
            if not is_likely_forklift_box(x1, y1, x2, y2, width, height):
                bad_fks.add(tid)
            elif is_static_track(track_pix_history.get(tid, []), fps):
                bad_fks.add(tid)
        for tid in bad_fks:
            per_track_state.pop(tid, None)

        alarm_pairs: list[dict] = []
        forklifts = [t for t, s in per_track_state.items() if s["role"] == "forklift"]
        persons = [t for t, s in per_track_state.items() if s["role"] == "person"]
        for f_id in forklifts:
            f = per_track_state[f_id]
            v_speed = math.hypot(f["vx"], f["vy"])
            r = adaptive_radius(v_speed)
            for p_id in persons:
                p = per_track_state[p_id]
                p_w = (p["wx"], p["wy"]); f_w = (f["wx"], f["wy"])
                d_now = math.hypot(p_w[0] - f_w[0], p_w[1] - f_w[1])
                d_min = d_now
                breach_T = None
                if d_now <= r:
                    breach_T = 0.0
                else:
                    for T in HORIZONS:
                        d_t = predict_distance(f_w, p_w, (f["vx"], f["vy"]),
                                               (p["vx"], p["vy"]), T)
                        d_min = min(d_min, d_t)
                        if d_t <= r:
                            breach_T = T
                            break
                if breach_T is not None and v_speed >= 0.05:
                    rel_v = math.hypot(f["vx"] - p["vx"], f["vy"] - p["vy"])
                    ttc = d_now / rel_v if rel_v > 1e-3 else 9.9
                    risk_score = max(20, min(99, (1 - d_min / max(r, 1e-3)) * 100))

                    f_pix = f["anchor_floor"]; p_pix = p["anchor_floor"]
                    fvx, fvy = f["v_px"]; pvx, pvy = p["v_px"]
                    cx_X = ((f_pix[0] + fvx * ttc) + (p_pix[0] + pvx * ttc)) / 2
                    cy_X = ((f_pix[1] + fvy * ttc) + (p_pix[1] + pvy * ttc)) / 2

                    alarm_pairs.append({
                        "fk_id": f_id, "ps_id": p_id,
                        "fk_pix": f_pix, "ps_pix": p_pix,
                        "fk_v_pix": f["v_px"],
                        "dist_m": d_min,  # pseudo-m, not used for display now
                        "ttc_s": min(ttc, 9.9),
                        "rel_speed": rel_v,
                        "red_radius": r,
                        "risk_score": risk_score,
                        "collision_x_pix": (cx_X, cy_X),
                    })
                    rows.append({
                        "frame": frame_idx, "fk_id": f_id, "ps_id": p_id,
                        "fk_speed_proxy": round(v_speed, 3),
                        "distance_proxy": round(d_now, 3),
                        "red_radius_proxy": round(r, 3),
                        "ttc_s": round(min(ttc, 99.9), 2),
                        "risk_score": round(risk_score, 1),
                    })

        out = annotate_dashboard(frame, per_track_state, alarm_pairs,
                                 track_pix_history, frame_idx, fps,
                                 width, height, args.cam_label,
                                 args.scale_speed_kmh)
        writer.write(out)

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{n_max}  pers={len(persons)} fk={len(forklifts)} "
                  f"pairs_alarm={len(alarm_pairs)}")

    cap.release(); writer.release()
    pd.DataFrame(rows).to_csv(args.tracks_out, index=False)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"wrote {args.tracks_out}  ({len(rows)} alarm-frame rows)")


if __name__ == "__main__":
    main()
