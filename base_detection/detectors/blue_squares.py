#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Tuple


def _order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos (x,y) como TL, TR, BR, BL."""
    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    idx = np.argsort(angles)
    ordered = pts[idx]

    # iniciar em TL (menor y, depois menor x)
    top_left_idx = np.lexsort((ordered[:,0], ordered[:,1]))[0]
    ordered = np.roll(ordered, -top_left_idx, axis=0)

    # garantir sentido horário TL->TR->BR->BL
    v1 = ordered[1] - ordered[0]
    v2 = ordered[2] - ordered[0]
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return ordered.astype(np.float32)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = v1.astype(np.float32); b = v2.astype(np.float32)
    na = np.linalg.norm(a) + 1e-6; nb = np.linalg.norm(b) + 1e-6
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def square_score(pts_ord: np.ndarray, area: float, bbox: Tuple[int,int,int,int]) -> float:
    """Score heurístico [0..1] combinando extent, aspecto e ângulos ~90°."""
    x, y, w, h = bbox
    extent = 0.0 if w * h <= 0 else float(area) / float(w * h)
    extent = max(0.0, min(1.0, extent))

    aspect = float(min(w, h)) / float(max(w, h) + 1e-6)
    aspect = max(0.0, min(1.0, aspect))

    p0, p1, p2, p3 = pts_ord
    e0 = p1 - p0; e1 = p2 - p1; e2 = p3 - p2; e3 = p0 - p3
    angles = [
        _angle_between(e0, e1),
        _angle_between(e1, e2),
        _angle_between(e2, e3),
        _angle_between(e3, e0),
    ]
    mean_dev = float(np.mean([abs(a - 90.0) for a in angles]))
    angle_score = max(0.0, min(1.0, 1.0 - (mean_dev / 45.0)))

    score = 0.4 * extent + 0.3 * aspect + 0.3 * angle_score
    return max(0.0, min(1.0, score))


def detect_blue_squares(mask: np.ndarray,
                        min_square_area: float = 500.0,
                        approx_epsilon_frac: float = 0.04):
    """
    Recebe máscara (0/255) do AZUL e retorna:
    - detections: List[dict] com keys: x,y,w,h,cx,cy,score,corners(np.float32[4,2])
    - contours: lista de contornos para debug
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_square_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon_frac * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(np.float32)
            pts_ord = _order_corners_clockwise(pts)

            x, y, w, h = cv2.boundingRect(cnt)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            score = square_score(pts_ord, area, (x, y, w, h))

            detections.append({
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "cx": cx, "cy": cy, "score": float(score), "corners": pts_ord
            })

    return detections, contours
