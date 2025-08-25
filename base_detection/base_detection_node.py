#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 node: detecção de quadrados AZUIS com saída PB + anotações AZUIS.
Compatível com base_detection_params.yaml (namespace: base_detection).

Lê:
  - color_image_topic
  - inferred_image_topic
  - detected_coords_topic
  - num_bases_topic
  - bool_detector_topic
  - blue_hsv_lower (double[3], H[0..179], S/V[0..255])
  - blue_hsv_upper (double[3])
  - yolo.model_path (ignorado aqui, mantido por compatibilidade)
  - detection_threshold (ignorado aqui, mantido por compatibilidade)

Publica:
  - inferred_image_topic (sensor_msgs/Image, bgr8): máscara PB em BGR com bbox/centróide AZUIS
  - detected_coords_topic (std_msgs/Float32MultiArray):
        por detecção: [x, y, w, h, cx, cy, score]
  - num_bases_topic (std_msgs/Int32): número de quadrados
  - bool_detector_topic (std_msgs/Bool): True se count > 0
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Int32, Bool
from cv_bridge import CvBridge

import cv2
import numpy as np
from typing import List


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
    """Ângulo (em graus) entre dois vetores v1 e v2."""
    a = v1.astype(np.float32)
    b = v2.astype(np.float32)
    na = np.linalg.norm(a) + 1e-6
    nb = np.linalg.norm(b) + 1e-6
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _square_score(pts_ord: np.ndarray, area: float, bbox: tuple) -> float:
    """
    Score heurístico [0..1] de "parece quadrado".
    Combina: extent, aspect ratio e ângulos ~90°.
    """
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
    angle_score = max(0.0, min(1.0, 1.0 - (mean_dev / 45.0)))  # 45° de desvio médio -> 0

    score = 0.4 * extent + 0.3 * aspect + 0.3 * angle_score
    return max(0.0, min(1.0, score))


class BaseDetectionNode(Node):
    def __init__(self):
        # nome do nó = 'base_detection' para casar com o namespace do YAML
        super().__init__('base_detection')

        # ---------- parâmetros ----------
        self.declare_parameter('color_image_topic', '/drone1_espcam/image_raw/compressed')
        self.declare_parameter('inferred_image_topic', '/base_detection/inferred_image')
        self.declare_parameter('detected_coords_topic', '/base_detection/detected_coords')
        self.declare_parameter('num_bases_topic', '/base_detection/num_bases')
        self.declare_parameter('bool_detector_topic', '/base_detection/has_base')

        # compat: presentes no YAML original (não usados aqui)
        self.declare_parameter('yolo.model_path', '')
        self.declare_parameter('detection_threshold', 0.50)

        # HSV azul (pode vir invertido no seu YAML atual; aqui tratamos como AZUL mesmo)
        self.declare_parameter('blue_hsv_lower', [100.0, 40.0, 80.0])
        self.declare_parameter('blue_hsv_upper', [135.0, 255.0, 255.0])

        input_topic = self.get_parameter('color_image_topic').get_parameter_value().string_value
        self.out_img_topic = self.get_parameter('inferred_image_topic').get_parameter_value().string_value
        self.out_coords_topic = self.get_parameter('detected_coords_topic').get_parameter_value().string_value
        self.out_count_topic = self.get_parameter('num_bases_topic').get_parameter_value().string_value
        self.out_bool_topic = self.get_parameter('bool_detector_topic').get_parameter_value().string_value

        # blur/clahe/morfologia fixos e simples (poderiam virar parâmetros depois)
        self.kernel_blur_size = 5
        if self.kernel_blur_size % 2 == 0:
            self.kernel_blur_size += 1

        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid = 8
        self.min_square_area = 500.0
        self.approx_epsilon_frac = 0.04

        # ---------- publishers ----------
        self.pub_image = self.create_publisher(Image, self.out_img_topic, 10)
        self.pub_coords = self.create_publisher(Float32MultiArray, self.out_coords_topic, 10)
        self.pub_count = self.create_publisher(Int32, self.out_count_topic, 10)
        self.pub_bool = self.create_publisher(Bool, self.out_bool_topic, 10)

        # ---------- subscriber único (CompressedImage) ----------
        self.bridge = CvBridge()
        self.sub = self.create_subscription(CompressedImage, input_topic, self._on_image, 10)

        # ---------- faixa AZUL do YAML (clamp p/ uint8) ----------
        def _clamp(v):
            h = max(0, min(int(v[0]), 179))
            s = max(0, min(int(v[1]), 255))
            br = max(0, min(int(v[2]), 255))
            return np.array([h, s, br], dtype=np.uint8)

        lower_cfg = self.get_parameter('blue_hsv_lower').get_parameter_value().double_array_value
        upper_cfg = self.get_parameter('blue_hsv_upper').get_parameter_value().double_array_value
        self.lower_blue = _clamp(lower_cfg if len(lower_cfg) == 3 else [100, 40, 80])
        self.upper_blue = _clamp(upper_cfg if len(upper_cfg) == 3 else [135, 255, 255])

        # cor azul para anotações (BGR)
        self.draw_blue = (255, 0, 0)

    # ---------- pipeline ----------
    def _on_image(self, msg: CompressedImage):
        # 1) decode -> BGR
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        # 2) Gaussian blur
        bgr_blur = cv2.GaussianBlur(bgr, (self.kernel_blur_size, self.kernel_blur_size), 0)

        # 3) CLAHE por canal (B,G,R)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_tile_grid, self.clahe_tile_grid))
        b, g, r = cv2.split(bgr_blur)
        b_eq = clahe.apply(b)
        g_eq = clahe.apply(g)
        r_eq = clahe.apply(r)
        bgr_eq = cv2.merge([b_eq, g_eq, r_eq])

        # 4) HSV + máscara azul (faixa do YAML)
        hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # 5) morfologia
        mask = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # 6) contornos & quadriláteros
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Overlay PB (expandir GRAY->BGR) para anotar em azul
        annotated = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        coords_out: List[float] = []
        count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_square_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.approx_epsilon_frac * peri, True)

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

                score = _square_score(pts_ord, area, (x, y, w, h))

                # saída por detecção: [x, y, w, h, cx, cy, score]
                coords_out.extend([float(x), float(y), float(w), float(h), cx, cy, float(score)])
                count += 1

                # anotações visuais (AZUL)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), self.draw_blue, 2)
                cv2.polylines(annotated, [pts_ord.astype(np.int32)], True, self.draw_blue, 2)
                cv2.circle(annotated, (int(cx), int(cy)), 4, self.draw_blue, -1)
                cv2.putText(
                    annotated, f"score: {score:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.draw_blue, 1, cv2.LINE_AA
                )

        # 7) publicar
        self.pub_count.publish(Int32(data=count))
        self.pub_bool.publish(Bool(data=(count > 0)))
        self.pub_coords.publish(Float32MultiArray(data=coords_out))

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header = msg.header
        self.pub_image.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BaseDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
