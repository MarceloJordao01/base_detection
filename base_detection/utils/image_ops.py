#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def gaussian_blur(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)


def clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    b, g, r = cv2.split(img_bgr)
    b_eq = clahe.apply(b)
    g_eq = clahe.apply(g)
    r_eq = clahe.apply(r)
    return cv2.merge([b_eq, g_eq, r_eq])


def hsv_mask(img_bgr: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask


def morph_open_close(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    if open_k < 1:
        open_k = 1
    if close_k < 1:
        close_k = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return mask


def gray_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def clamp_hsv_triplet(vals) -> np.ndarray:
    """Clamp para H[0..179], S[0..255], V[0..255] -> np.uint8[3]."""
    if len(vals) != 3:
        raise ValueError("HSV triplet must have 3 elements")
    h = max(0, min(int(vals[0]), 179))
    s = max(0, min(int(vals[1]), 255))
    v = max(0, min(int(vals[2]), 255))
    return np.array([h, s, v], dtype=np.uint8)
