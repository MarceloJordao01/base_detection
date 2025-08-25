#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Int32, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

from base_detection.utils.image_ops import (
    gaussian_blur, clahe_bgr, hsv_mask, morph_open_close, gray_to_bgr, clamp_hsv_triplet
)
from base_detection.detectors.blue_squares import detect_blue_squares
from base_detection.detectors.yellow_circles import detect_yellow_circles


class BaseDetectionNode(Node):
    def __init__(self):
        super().__init__('base_detection')

        # ---------- parâmetros ----------
        self.declare_parameter('color_image_topic', '/drone1_espcam/image_raw/compressed')
        self.declare_parameter('inferred_image_topic', '/base_detection/inferred_image')
        self.declare_parameter('detected_coords_topic', '/base_detection/detected_coords')
        self.declare_parameter('num_bases_topic', '/base_detection/num_bases')
        self.declare_parameter('bool_detector_topic', '/base_detection/has_base')

        # HSV ranges
        self.declare_parameter('blue_hsv_lower', [100.0, 40.0, 80.0])
        self.declare_parameter('blue_hsv_upper', [135.0, 255.0, 255.0])
        self.declare_parameter('yellow_hsv_lower', [15.0, 40.0, 120.0])
        self.declare_parameter('yellow_hsv_upper', [35.0, 255.0, 255.0])

        # Preprocess
        self.declare_parameter('kernel_blur_size', 5)
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_grid', 8)
        self.declare_parameter('morph_open_k', 3)
        self.declare_parameter('morph_close_k', 5)

        # Quadrados
        self.declare_parameter('min_square_area', 500.0)
        self.declare_parameter('approx_epsilon_frac', 0.04)

        # Círculos (contornos)
        self.declare_parameter('min_circle_radius', 5)
        self.declare_parameter('max_circle_radius', 300)
        self.declare_parameter('min_circle_area', 50.0)
        self.declare_parameter('min_circle_circularity', 0.8)

        # Score thresholds
        self.declare_parameter('square_score_threshold', 0.8)
        self.declare_parameter('circle_score_threshold', 0.5)

        # ---------- leitura de params ----------
        input_topic = self.get_parameter('color_image_topic').value
        self.out_img_topic = self.get_parameter('inferred_image_topic').value
        self.out_coords_topic = self.get_parameter('detected_coords_topic').value
        self.out_count_topic = self.get_parameter('num_bases_topic').value
        self.out_bool_topic = self.get_parameter('bool_detector_topic').value

        self.lower_blue = clamp_hsv_triplet(self.get_parameter('blue_hsv_lower').value)
        self.upper_blue = clamp_hsv_triplet(self.get_parameter('blue_hsv_upper').value)
        self.lower_yel  = clamp_hsv_triplet(self.get_parameter('yellow_hsv_lower').value)
        self.upper_yel  = clamp_hsv_triplet(self.get_parameter('yellow_hsv_upper').value)

        self.square_thr = float(self.get_parameter('square_score_threshold').value)
        self.circle_thr = float(self.get_parameter('circle_score_threshold').value)

        self.kernel_blur_size = int(self.get_parameter('kernel_blur_size').value)
        self.clahe_clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        self.clahe_tile_grid = int(self.get_parameter('clahe_tile_grid').value)
        self.morph_open_k = int(self.get_parameter('morph_open_k').value)
        self.morph_close_k = int(self.get_parameter('morph_close_k').value)

        self.min_square_area = float(self.get_parameter('min_square_area').value)
        self.approx_epsilon_frac = float(self.get_parameter('approx_epsilon_frac').value)

        self.min_circle_radius = int(self.get_parameter('min_circle_radius').value)
        self.max_circle_radius = int(self.get_parameter('max_circle_radius').value)
        self.min_circle_area   = float(self.get_parameter('min_circle_area').value)
        self.min_circle_circularity = float(self.get_parameter('min_circle_circularity').value)

        # ---------- publishers ----------
        self.pub_image = self.create_publisher(Image, self.out_img_topic, 10)
        self.pub_coords = self.create_publisher(Float32MultiArray, self.out_coords_topic, 10)
        self.pub_count = self.create_publisher(Int32, self.out_count_topic, 10)
        self.pub_bool = self.create_publisher(Bool, self.out_bool_topic, 10)

        self.pub_debug_blue = self.create_publisher(Image, "/base_detection/debug_blue", 10)
        self.pub_debug_yellow = self.create_publisher(Image, "/base_detection/debug_yellow", 10)

        # ---------- subscriber ----------
        self.bridge = CvBridge()
        self.sub = self.create_subscription(CompressedImage, input_topic, self._on_image, 10)

        # cores de anotação
        self.blue_draw = (255, 0, 0)
        self.yellow_draw = (0, 255, 255)

    def _on_image(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        # Pré-processamento
        bgr_blur = gaussian_blur(bgr, self.kernel_blur_size)
        bgr_eq   = clahe_bgr(bgr_blur, self.clahe_clip_limit, self.clahe_tile_grid)

        # Máscaras
        blue_mask = morph_open_close(hsv_mask(bgr_eq, self.lower_blue, self.upper_blue),
                                     self.morph_open_k, self.morph_close_k)
        yellow_mask = morph_open_close(hsv_mask(bgr_eq, self.lower_yel, self.upper_yel),
                                       self.morph_open_k, self.morph_close_k)

        # Detecção
        sq_dets, _ = detect_blue_squares(blue_mask, self.min_square_area, self.approx_epsilon_frac)
        ci_dets, _ = detect_yellow_circles(
            yellow_mask,
            min_radius=self.min_circle_radius,
            max_radius=self.max_circle_radius,
            min_area=self.min_circle_area,
            min_circularity=self.min_circle_circularity
        )

        # Filtra pelo threshold
        sq_dets = [d for d in sq_dets if d["score"] >= self.square_thr]
        ci_dets = [d for d in ci_dets if d["score"] >= self.circle_thr]

        coords = []
        count_total = 0

        # imagens de debug
        blue_dbg = gray_to_bgr(blue_mask)
        yellow_dbg = gray_to_bgr(yellow_mask)
        annotated = gray_to_bgr(cv2.bitwise_or(blue_mask, yellow_mask))

        # --- quadrados ---
        for d in sq_dets:
            x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
            cx, cy = int(d["cx"]), int(d["cy"])
            score = d["score"]
            corners = d["corners"].astype(np.int32)
            for img in [blue_dbg, annotated]:
                cv2.rectangle(img, (x, y), (x+w, y+h), self.blue_draw, 2)
                cv2.polylines(img, [corners], True, self.blue_draw, 2)
                cv2.circle(img, (cx, cy), 4, self.blue_draw, -1)
                cv2.putText(img, f"{score:.2f}", (x, max(0,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.blue_draw, 1)
            coords.extend([1.0, float(x), float(y), float(w), float(h), float(cx), float(cy), score])
            count_total += 1

        # --- círculos ---
        for d in ci_dets:
            x, y, r = int(d["x"]), int(d["y"]), int(d["r"])
            score = d["score"]
            for img in [yellow_dbg, annotated]:
                cv2.circle(img, (x,y), r, self.yellow_draw, 2)
                cv2.circle(img, (x,y), 3, self.yellow_draw, -1)
                cv2.putText(img, f"{score:.2f}", (max(0,x-r), max(0,y-r-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.yellow_draw, 1)
            coords.extend([2.0, float(x), float(y), float(r), score])
            count_total += 1

        # Publicações
        self.pub_count.publish(Int32(data=count_total))
        self.pub_bool.publish(Bool(data=(count_total > 0)))
        self.pub_coords.publish(Float32MultiArray(data=coords))

        self.pub_image.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        self.pub_debug_blue.publish(self.bridge.cv2_to_imgmsg(blue_dbg, encoding="bgr8"))
        self.pub_debug_yellow.publish(self.bridge.cv2_to_imgmsg(yellow_dbg, encoding="bgr8"))


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


if __name__ == "__main__":
    main()
