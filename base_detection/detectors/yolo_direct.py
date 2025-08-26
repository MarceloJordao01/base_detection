#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detector YOLO (Ultralytics) sem pré-processamento.
Recebe uma imagem BGR (OpenCV), envia direto ao modelo YOLO e retorna detecções.

Saída de cada detecção:
  - class_id (int)
  - label (str)
  - conf (float)          -> "score"
  - x, y, w, h (float)    -> bbox no formato xywh (x,y = canto superior esquerdo)
  - cx, cy (float)        -> centróide = (x + w/2, y + h/2)

Também retorna metadados:
  - names (dict id->label)  -> classes que o modelo fornece
  - raw (obj Results)       -> resultados brutos da Ultralytics (para debug)
"""

from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import cv2

from ultralytics import YOLO
import torch

# Cache simples de modelos: (model_path, device, half) -> YOLO
_MODEL_CACHE: Dict[Tuple[str, str, bool], YOLO] = {}


def _resolve_device(device_param: str) -> str:
    if device_param == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_param


def _get_model(model_path: str, device: str, use_half: bool) -> YOLO:
    key = (model_path, device, bool(use_half))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[YOLO] Modelo não encontrado: {model_path}")
    model = YOLO(model_path)
    _MODEL_CACHE[key] = model
    return model


def detect_yolo_direct(
    bgr: np.ndarray,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    device: str = "auto",
    half: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Executa YOLO diretamente sobre a imagem BGR (sem pré-processamento).

    Retorna (detections, meta):
      detections: List[dict] com chaves:
        - 'class_id', 'label', 'conf', 'x','y','w','h','cx','cy'
      meta: dict com 'names' (id->label) e 'raw' (obj Results)
    """
    if bgr is None or bgr.size == 0:
        return [], {}

    dev = _resolve_device(device)
    model = _get_model(model_path, dev, half)

    # YOLO da Ultralytics espera RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    res = model.predict(
        source=rgb,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        classes=(classes if classes and len(classes) > 0 else None),
        device=dev,
        half=bool(half and ("cuda" in dev) and torch.cuda.is_available()),
        verbose=False
    )

    detections: List[Dict] = []
    names = getattr(model, "names", {})

    if len(res) > 0:
        r = res[0]
        if r.boxes is not None and len(r.boxes) > 0:
            # xyxy (N,4), conf (N,1), cls (N,1)
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()

            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i].astype(float)
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                cx = x1 + w * 0.5
                cy = y1 + h * 0.5

                conf_i = float(confs[i])
                cls_id = int(clss[i]) if not np.isnan(clss[i]) else -1
                label  = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "conf": conf_i,
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(w),
                    "h": float(h),
                    "cx": float(cx),
                    "cy": float(cy),
                })

    meta = {"names": names, "raw": res}
    return detections, meta
