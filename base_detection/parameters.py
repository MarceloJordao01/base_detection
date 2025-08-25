# -*- coding: utf-8 -*-
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List
import rclpy
from rclpy.node import Node


@dataclass
class HSVFilterParams:
    lower: List[int]
    upper: List[int]


@dataclass
class ImageInferencerParams:
    model_path: str
    hsv_filter: HSVFilterParams
    detection_threshold: float


def hsv_bounds_from_list(vals) -> list:
    """Clamp & cast an HSV triple into valid [0..H,S,V] ranges used by OpenCV."""
    # OpenCV HSV ranges are H: [0..179], S: [0..255], V: [0..255]
    if len(vals) != 3:
        raise ValueError("HSV list must have exactly 3 elements.")
    h = max(0, min(int(vals[0]), 179))
    s = max(0, min(int(vals[1]), 255))
    v = max(0, min(int(vals[2]), 255))
    return [h, s, v]


def get_image_inferencer_params(node: Node) -> ImageInferencerParams:
    """
    Declare and fetch parameters for the ImageInferencer node.

    Parameters (with defaults):
      - model_path (string): path for YOLO model, e.g. 'yolov8n.pt'
      - hsv_lower (double[3]): lower HSV bound
      - hsv_upper (double[3]): upper HSV bound
      - detection_threshold (double): YOLO confidence threshold
      - input_image_topic (string): image topic to subscribe
    """

    node.declare_parameter("model_path", "yolov8n.pt")
    node.declare_parameter("hsv_lower", [0.0, 0.0, 0.0])
    node.declare_parameter("hsv_upper", [179.0, 255.0, 255.0])
    node.declare_parameter("detection_threshold", 0.25)
    node.declare_parameter("input_image_topic", "")

    model_path = node.get_parameter("model_path").get_parameter_value().string_value
    lower = node.get_parameter("hsv_lower").get_parameter_value().double_array_value
    upper = node.get_parameter("hsv_upper").get_parameter_value().double_array_value
    thr = node.get_parameter("detection_threshold").get_parameter_value().double_value

    hsv = HSVFilterParams(
        lower=hsv_bounds_from_list(lower),
        upper=hsv_bounds_from_list(upper),
    )

    return ImageInferencerParams(
        model_path=model_path,
        hsv_filter=hsv,
        detection_threshold=float(thr),
    )
