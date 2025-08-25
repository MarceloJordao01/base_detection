#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 node super simples: assina UM tópico de imagem (qualquer um dos dois tipos comuns)
e republica em outro tópico como sensor_msgs/Image.

Regras:
- Não há troca de parâmetros em runtime.
- Um único callback para a imagem (sem separar raw/compressed).
- Se a mensagem parecer CompressedImage (tem .data bytes), decodifica via OpenCV.
- Caso contrário, tenta tratar como sensor_msgs/Image via CvBridge.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import Optional


class UltraSimpleImageRepublisher(Node):
    def __init__(self):
        super().__init__('ultra_simple_image_republisher')

        # Parâmetros (apenas leitura na inicialização)
        self.declare_parameter('color_image_topic', '/drone1_espcam/image_raw/compressed')
        self.declare_parameter('inferred_image_topic', '/base_detection/inferred_image')

        input_topic = self.get_parameter('color_image_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('inferred_image_topic').get_parameter_value().string_value

        # Publisher fixo
        self.pub_image = self.create_publisher(Image, output_topic, 10)

        # Bridge para conversões
        self.bridge = CvBridge()

        # ÚNICA subscription (sem distinção por tipo)
        # Observação: usamos CompressedImage como tipo da subscription por padrão porque
        # seu callback recebe o payload bruto (data). Dentro do callback, tentamos
        # detectar se é CompressedImage ou Image e tratar conforme.
        # Se você preferir fixar o tipo, troque CompressedImage por Image abaixo.
        self.sub = self.create_subscription(
            CompressedImage,  # escolha um tipo; o callback lida com ambos
            input_topic,
            self._on_image,
            10
        )

    def _on_image(self, msg):
        """
        Callback único. Tenta:
        1) Decodificar como CompressedImage (campo .data).
        2) Se falhar, tenta converter como sensor_msgs/Image via CvBridge.
        """
        img_msg: Optional[Image] = None

        # Tentativa 1: tratar como CompressedImage
        try:
            if hasattr(msg, 'data') and isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
                    img_msg.header = msg.header
        except Exception:
            img_msg = None

        # Tentativa 2: tratar como sensor_msgs/Image (caso a subscription tenha sido feita para Image)
        if img_msg is None:
            try:
                # Se msg já for Image, apenas garanta o encoding bgr8
                if isinstance(msg, Image):
                    # Converte para cv2 e volta, para padronizar encoding
                    img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    img_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
                    img_msg.header = msg.header
            except (CvBridgeError, Exception):
                img_msg = None

        # Publica se deu tudo certo
        if img_msg is not None:
            self.pub_image.publish(img_msg)
        # Caso contrário, ignora silenciosamente (sem logs, como pedido)

def main(args=None):
    rclpy.init(args=args)
    node = UltraSimpleImageRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
