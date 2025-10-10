#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Dict, Tuple


def detect_yellow_circles(mask: np.ndarray,
                          min_radius: int = 5,
                          max_radius: int = 300,
                          min_area: float = 50.0,
                          min_circularity: float = 0.8) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Detecta círculos a partir da máscara binária usando contornos.

    - Usa circularidade = 4π*area / perimeter² para distinguir círculos (~1) de quadrados/polígonos.
    - Apenas contornos com circularidade >= min_circularity são aceitos.

    Retorna:
      - detections: List[dict] com {x, y, r, score}
      - contours: lista de contornos (debug)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Dict] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 1e-6:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        if circularity < min_circularity:
            continue  # descarta quadrados e polígonos

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        r = int(radius)

        if r < min_radius or r > max_radius:
            continue

        detections.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "score": float(circularity)
        })

    return detections, contours
