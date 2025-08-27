#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from math import sqrt

from base_detection.utils.image_ops import (
    gaussian_blur, clahe_bgr, hsv_mask, morph_open_close, gray_to_bgr, clamp_hsv_triplet
)
from base_detection.detectors.blue_squares import detect_blue_squares
from base_detection.detectors.yellow_circles import detect_yellow_circles
from base_detection.detectors.yolo_direct import detect_yolo_direct


class BaseDetectionNode(Node):
    def __init__(self):
        super().__init__('base_detection')

        # ---------- parâmetros ----------
        self.declare_parameter('color_image_topic', '/drone1_espcam/image_raw/compressed')
        self.declare_parameter('inferred_image_topic', '/base_detection/inferred_image')
        self.declare_parameter('detected_coords_topic', '/base_detection/detected_coords')
        self.declare_parameter('num_bases_topic', '/base_detection/num_bases')

        # HSV ranges
        self.declare_parameter('blue_hsv_lower', [100.0, 40.0, 80.0])
        self.declare_parameter('blue_hsv_upper', [135.0, 255.0, 255.0])
        self.declare_parameter('yellow_hsv_lower', [15.0, 40.0, 120.0])
        self.declare_parameter('yellow_hsv_upper', [35.0, 255.0, 255.0])

        # Preprocess (para HSV; YOLO não usa)
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

        # YOLO (detector 3) — SEM PREPROCESSAMENTO
        self.declare_parameter('enable_yolo', True)
        self.declare_parameter('yolo.model_path', '<share>/models/best.pt')
        self.declare_parameter('yolo.conf_threshold', 0.25)
        self.declare_parameter('yolo.iou_threshold', 0.45)
        self.declare_parameter('yolo.imgsz', 640)
        self.declare_parameter('yolo.max_det', 300)
        self.declare_parameter(
            'yolo.classes',
            [],
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY)
        )
        self.declare_parameter('yolo.device', 'auto')  # auto|cpu|cuda:0
        self.declare_parameter('yolo.half', False)

        # Score thresholds
        self.declare_parameter('square_score_threshold', 0.8)
        self.declare_parameter('circle_score_threshold', 0.5)
        self.declare_parameter('yolo_conf_threshold', 0.25)

        # ---------- Fusão ----------
        self.declare_parameter('fusion.enable', True)
        self.declare_parameter('fusion.max_dist_px', 40)
        self.declare_parameter('fusion.max_dist_frac', 0.05)

        # ---------- Bias por detector ----------
        self.declare_parameter('bias.square', 1.0)
        self.declare_parameter('bias.circle', 1.0)
        self.declare_parameter('bias.yolo', 1.0)

        # ---------- leitura ----------
        input_topic = self.get_parameter('color_image_topic').value
        self.out_img_topic = self.get_parameter('inferred_image_topic').value
        self.out_coords_topic = self.get_parameter('detected_coords_topic').value
        self.out_count_topic = self.get_parameter('num_bases_topic').value

        self.lower_blue = clamp_hsv_triplet(self.get_parameter('blue_hsv_lower').value)
        self.upper_blue = clamp_hsv_triplet(self.get_parameter('blue_hsv_upper').value)
        self.lower_yel  = clamp_hsv_triplet(self.get_parameter('yellow_hsv_lower').value)
        self.upper_yel  = clamp_hsv_triplet(self.get_parameter('yellow_hsv_upper').value)

        self.square_thr = float(self.get_parameter('square_score_threshold').value)
        self.circle_thr = float(self.get_parameter('circle_score_threshold').value)
        self.yolo_thr   = float(self.get_parameter('yolo_conf_threshold').value)

        self.kernel_blur_size = int(self.get_parameter('kernel_blur_size').value)
        self.clahe_clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        self.clahe_tile_grid  = int(self.get_parameter('clahe_tile_grid').value)
        self.morph_open_k     = int(self.get_parameter('morph_open_k').value)
        self.morph_close_k    = int(self.get_parameter('morph_close_k').value)

        self.min_square_area     = float(self.get_parameter('min_square_area').value)
        self.approx_epsilon_frac = float(self.get_parameter('approx_epsilon_frac').value)

        self.min_circle_radius      = int(self.get_parameter('min_circle_radius').value)
        self.max_circle_radius      = int(self.get_parameter('max_circle_radius').value)
        self.min_circle_area        = float(self.get_parameter('min_circle_area').value)
        self.min_circle_circularity = float(self.get_parameter('min_circle_circularity').value)

        # YOLO params
        self.enable_yolo     = bool(self.get_parameter('enable_yolo').value)
        self.yolo_model_path = self.get_parameter('yolo.model_path').value
        self.yolo_conf       = float(self.get_parameter('yolo.conf_threshold').value)
        self.yolo_iou        = float(self.get_parameter('yolo.iou_threshold').value)
        self.yolo_imgsz      = int(self.get_parameter('yolo.imgsz').value)
        self.yolo_max_det    = int(self.get_parameter('yolo.max_det').value)
        classes_param        = self.get_parameter_or('yolo.classes', Parameter('yolo.classes', value=[]))
        self.yolo_classes    = list(classes_param.value) if classes_param.value else []
        self.yolo_device     = self.get_parameter('yolo.device').value
        self.yolo_half       = bool(self.get_parameter('yolo.half').value)

        # Fusão params
        self.fusion_enable      = bool(self.get_parameter('fusion.enable').value)
        self.fusion_max_px      = int(self.get_parameter('fusion.max_dist_px').value)
        self.fusion_max_frac    = float(self.get_parameter('fusion.max_dist_frac').value)

        # Bias
        self.bias_square = float(self.get_parameter('bias.square').value)
        self.bias_circle = float(self.get_parameter('bias.circle').value)
        self.bias_yolo   = float(self.get_parameter('bias.yolo').value)

        # ---------- pubs/subs ----------
        self.pub_image = self.create_publisher(Image, self.out_img_topic, 10)
        self.pub_coords = self.create_publisher(Float32MultiArray, self.out_coords_topic, 10)
        self.pub_count = self.create_publisher(Int32, self.out_count_topic, 10)

        self.pub_debug_blue = self.create_publisher(Image, "/base_detection/debug_blue", 10)
        self.pub_debug_yellow = self.create_publisher(Image, "/base_detection/debug_yellow", 10)
        self.pub_debug_yolo = self.create_publisher(Image, "/base_detection/debug_yolo", 10)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(CompressedImage, input_topic, self._on_image, 10)

        self.blue_draw = (255, 0, 0)
        self.yellow_draw = (0, 255, 255)
        self.yolo_draw = (0, 255, 0)

        self.get_logger().info(
            f"[base_detection] Fusão habilitada={self.fusion_enable} | "
            f"bias(square,circle,yolo)=({self.bias_square},{self.bias_circle},{self.bias_yolo})"
        )

    # ------------------ utilidades de fusão ------------------
    @staticmethod
    def _euclid(a, b):
        return float(sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

    def _cluster_by_distance(self, points, diag):
        """
        Agrupa índices por proximidade usando um limiar baseado em:
        max(fusion_max_px, fusion_max_frac * diagonal_da_imagem)
        """
        thr = max(float(self.fusion_max_px), float(self.fusion_max_frac) * float(diag))
        assigned = [-1] * len(points)
        clusters = []
        cid = 0
        for i, p in enumerate(points):
            if assigned[i] != -1:
                continue
            assigned[i] = cid
            cluster = [i]
            changed = True
            while changed:
                changed = False
                for j, q in enumerate(points):
                    if assigned[j] == -1:
                        if any(self._euclid(q, points[k]) <= thr for k in cluster):
                            assigned[j] = cid
                            cluster.append(j)
                            changed = True
            clusters.append(cluster)
            cid += 1
        return clusters

    # ------------------ callback principal ------------------
    def _on_image(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        H, W = bgr.shape[:2]
        diag = sqrt(W*W + H*H)

        # Pré-processamento para HSV (YOLO NÃO USA)
        bgr_blur = gaussian_blur(bgr, self.kernel_blur_size)
        bgr_eq   = clahe_bgr(bgr_blur, self.clahe_clip_limit, self.clahe_tile_grid)

        blue_mask = morph_open_close(hsv_mask(bgr_eq, self.lower_blue, self.upper_blue),
                                     self.morph_open_k, self.morph_close_k)
        yellow_mask = morph_open_close(hsv_mask(bgr_eq, self.lower_yel, self.upper_yel),
                                       self.morph_open_k, self.morph_close_k)

        # Detectores HSV
        sq_dets, _ = detect_blue_squares(blue_mask, self.min_square_area, self.approx_epsilon_frac)
        ci_dets, _ = detect_yellow_circles(
            yellow_mask,
            min_radius=self.min_circle_radius,
            max_radius=self.max_circle_radius,
            min_area=self.min_circle_area,
            min_circularity=self.min_circle_circularity
        )
        sq_dets = [d for d in sq_dets if float(d.get("score", 0.0)) >= self.square_thr]
        ci_dets = [d for d in ci_dets if float(d.get("score", 0.0)) >= self.circle_thr]

        # YOLO sem pré-processamento → usa BGR cru
        yo_dets = []
        if self.enable_yolo:
            try:
                yo_raw, meta = detect_yolo_direct(
                    bgr,
                    model_path=self.yolo_model_path,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    imgsz=self.yolo_imgsz,
                    max_det=self.yolo_max_det,
                    classes=self.yolo_classes,
                    device=self.yolo_device,
                    half=self.yolo_half
                )
                yo_dets = [d for d in yo_raw if d["conf"] >= self.yolo_thr]
                if hasattr(self, "_logged_yolo_names") is False:
                    names = meta.get("names", {})
                    self.get_logger().info(f"[YOLO] classes: {names}")
                    self._logged_yolo_names = True
            except Exception as e:
                self.get_logger().error(f"[YOLO] erro: {e}")
                yo_dets = []

        # ------------ debug images separadas ------------
        blue_dbg = gray_to_bgr(blue_mask)
        yellow_dbg = gray_to_bgr(yellow_mask)
        yolo_dbg = bgr.copy()

        for d in sq_dets:
            x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
            cx, cy = int(d["cx"]), int(d["cy"])
            score = float(d["score"])
            cv2.rectangle(blue_dbg, (x, y), (x+w, y+h), self.blue_draw, 2)
            cv2.polylines(blue_dbg, [d["corners"].astype(np.int32)], True, self.blue_draw, 2)
            cv2.circle(blue_dbg, (cx, cy), 4, self.blue_draw, -1)
            cv2.putText(blue_dbg, f"{score:.2f}", (x, max(0, y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.blue_draw, 1, cv2.LINE_AA)

        for d in ci_dets:
            x, y, r = int(d["x"]), int(d["y"]), int(d["r"])
            score = float(d["score"])
            cv2.circle(yellow_dbg, (x, y), r, self.yellow_draw, 2)
            cv2.circle(yellow_dbg, (x, y), 3, self.yellow_draw, -1)
            cv2.putText(yellow_dbg, f"{score:.2f}", (max(0, x - r), max(0, y - r - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.yellow_draw, 1, cv2.LINE_AA)

        if self.enable_yolo:
            for d in yo_dets:
                x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
                cx, cy = int(d["cx"]), int(d["cy"])
                conf   = float(d["conf"])
                cls_id = int(d["class_id"])
                label  = str(d.get("label", cls_id))
                cv2.rectangle(yolo_dbg, (x, y), (x+w, y+h), self.yolo_draw, 2)
                cv2.circle(yolo_dbg, (cx, cy), 3, self.yolo_draw, -1)
                txt = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(yolo_dbg, (x, max(0, y - th - 4)), (x + tw + 4, y), self.yolo_draw, -1)
                cv2.putText(yolo_dbg, txt, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # ----------- FUSÃO DE DETECÇÕES -----------
        candidates = []

        # tipo: 1=quadrado azul, 2=círculo amarelo, 3=yolo
        if self.bias_square > 0.0:
            for d in sq_dets:
                candidates.append({
                    "type": 1,
                    "cx": float(d["cx"]), "cy": float(d["cy"]),
                    "score": float(d["score"]),
                    "score_eff": float(d["score"]) * self.bias_square,
                    "data": d
                })
        if self.bias_circle > 0.0:
            for d in ci_dets:
                candidates.append({
                    "type": 2,
                    "cx": float(d["x"]), "cy": float(d["y"]),
                    "score": float(d["score"]),
                    "score_eff": float(d["score"]) * self.bias_circle,
                    "data": d
                })
        if self.bias_yolo > 0.0:
            for d in yo_dets:
                candidates.append({
                    "type": 3,
                    "cx": float(d["cx"]), "cy": float(d["cy"]),
                    "score": float(d["conf"]),
                    "score_eff": float(d["conf"]) * self.bias_yolo,
                    "data": d
                })

        fused = []
        if self.fusion_enable and len(candidates) > 0:
            pts = [(c["cx"], c["cy"]) for c in candidates]
            clusters = self._cluster_by_distance(pts, diag)
            for cl in clusters:
                # Seleciona pelo score efetivo (score * bias)
                best_idx = max(cl, key=lambda i: candidates[i]["score_eff"])
                fused.append(candidates[best_idx])
        else:
            fused = candidates  # sem fusão, usa tudo que passou no bias>0

        # ----------- Monta imagem anotada APENAS com as fusões -----------
        annotated = bgr.copy()

        # ======= NOVA PUBLICAÇÃO DE COORDENADAS (formato x1,y1,x2,y2,score) =======
        frame_detections = []  # lista de [x1,y1,x2,y2,score]

        for c in fused:
            t = c["type"]
            d = c["data"]

            if t == 1:
                # quadrado (usar bbox)
                x1 = float(d["x"])
                y1 = float(d["y"])
                x2 = float(d["x"] + d["w"])
                y2 = float(d["y"] + d["h"])
                score = float(d["score"])

                # desenho
                cx, cy = int(d["cx"]), int(d["cy"])
                corners = d["corners"].astype(np.int32)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), self.blue_draw, 2)
                cv2.polylines(annotated, [corners], True, self.blue_draw, 2)
                cv2.circle(annotated, (cx, cy), 4, self.blue_draw, -1)
                cv2.putText(annotated, f"{score:.2f}", (int(x1), max(0, int(y1)-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.blue_draw, 1, cv2.LINE_AA)

                frame_detections.append([x1, y1, x2, y2, score])

            elif t == 2:
                # círculo → publica bbox do círculo
                cx = float(d["x"])
                cy = float(d["y"])
                r  = float(d["r"])
                score = float(d["score"])
                x1 = cx - r
                y1 = cy - r
                x2 = cx + r
                y2 = cy + r

                # desenho
                cv2.circle(annotated, (int(cx), int(cy)), int(r), self.yellow_draw, 2)
                cv2.circle(annotated, (int(cx), int(cy)), 3, self.yellow_draw, -1)
                cv2.putText(annotated, f"{score:.2f}", (max(0, int(cx - r)), max(0, int(cy - r - 6))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.yellow_draw, 1, cv2.LINE_AA)

                frame_detections.append([x1, y1, x2, y2, score])

            else:
                # YOLO (usar bbox do modelo)
                x1 = float(d["x"])
                y1 = float(d["y"])
                x2 = float(d["x"] + d["w"])
                y2 = float(d["y"] + d["h"])
                score = float(d["conf"])
                cx, cy = int(d["cx"]), int(d["cy"])
                cls_id = int(d["class_id"])
                label  = str(d.get("label", cls_id))

                # desenho
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), self.yolo_draw, 2)
                cv2.circle(annotated, (cx, cy), 3, self.yolo_draw, -1)
                txt = f"{label} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (int(x1), max(0, int(y1) - th - 4)), (int(x1 + tw + 4), int(y1)), self.yolo_draw, -1)
                cv2.putText(annotated, txt, (int(x1 + 2), int(y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                frame_detections.append([x1, y1, x2, y2, score])

        # ----------- Publicações -----------
        # número total (após fusão)
        count_total = len(frame_detections)
        self.pub_count.publish(Int32(data=count_total))

        # coords no formato solicitado: [x1,y1,x2,y2,score, ...]
        if frame_detections:
            flat = [item for det in frame_detections for item in det]
            self.pub_coords.publish(Float32MultiArray(data=flat))

        # imagens
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        self.pub_debug_blue.publish(self.bridge.cv2_to_imgmsg(blue_dbg, encoding="bgr8"))
        self.pub_debug_yellow.publish(self.bridge.cv2_to_imgmsg(yellow_dbg, encoding="bgr8"))
        if self.enable_yolo:
            self.pub_debug_yolo.publish(self.bridge.cv2_to_imgmsg(yolo_dbg, encoding="bgr8"))


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
