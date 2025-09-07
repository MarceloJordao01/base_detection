# Base Detection System

![License](https://img.shields.io/badge/License-MIT-blue.svg)![ROS Version](https://img.shields.io/badge/ROS-2%20Humble-blueviolet)![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Computer Vision](https://img.shields.io/badge/CV-OpenCV-orange.svg)![YOLO](https://img.shields.io/badge/YOLO-v8-blueviolet.svg)

Este repositório contém uma solução baseada em ROS2 para detecção autônoma de bases. O sistema utiliza uma camera OV2660 da espcam e cv2.


## Estrutura do repositorio

```
/base_detection
├── base_detection/              # Código-fonte principal em Python
│   ├── __init__.py
│   ├── base_detection_node.py   # Nó principal ROS2 para detecção das bases
│   ├── detectors/               # Algoritmos de detecção específicos
│   │   ├── __init__.py
│   │   ├── blue_squares.py      # Detector de quadrados azuis
│   │   ├── yellow_circles.py    # Detector de círculos amarelos
│   │   └── yolo_direct.py       # Detector usando YOLO diretamente
│   ├── utils/                   # Funções auxiliares
│   │   ├── __init__.py
│   │   └── image_ops.py         # Operações de processamento de imagem
├── config/                      # Arquivos de configuração YAML
│   └── base_detection_params.yaml  # Parâmetros de detecção
├── launch/                      # Arquivos de lançamento ROS2
│   └── base_detection.launch.py # Launch file para inicializar o sistema
├── models/                      # Modelos treinados
│   └── best.pt                  # Modelo YOLO pré-treinado
├── resource/                    # Índice de recursos do ament
├── package.xml                  # Manifesto do pacote ROS2
├── README.md                    # Documentação do pacote
├── setup.cfg                    # Configurações adicionais de setup
└── setup.py                     # Script de instalação do pacote Python

```

## Arquitetura do sistema

### Visão geral
O nó usa três detectores em paralelo (dois clássicos com HSV + um com YOLO). Depois, fundem as detecções com base na proximidade e em um peso (bias) por detector, publicando:
 - imagem anotada,
 - bounding boxes,
 - centróides,
 - contagem de alvos,
 - três imagens de debug (uma por detector)

### Topicos (I/O)

#### Assinatura (input)

 - `color_image_topic` (default: `/drone1_espcam/image_raw/compressed`)
 
  Tipo: sensor_msgs/CompressedImage

#### Publicação (output)

 - `inferred_image_topic` (default: /base_detection/inferred_image) → sensor_msgs/Image (bgr8)
Imagem original anotada apenas com as detecções finais (após fusão).

 - `detected_coords_topic` (default: /base_detection/detected_coords) → std_msgs/Float32MultiArray
Vetor achatado no formato [x1,y1,x2,y2,score, …] por detecção.

 - `num_bases_topic` (default: /base_detection/num_bases) → std_msgs/Int32
Número de detecções após a fusão.

 - `centroids_topic` (default: /base_detection/centroids) → std_msgs/Float32MultiArray
Vetor achatado [cx1,cy1, cx2,cy2, …] (útil para projeção 3D/tracking).

#### Tópicos de debug (imagens por detector)

 - `/base_detection/debug_blue` → máscara/resultado do detector quadrados azuis (HSV).

 - `/base_detection/debug_yellow` → máscara/resultado do detector círculos amarelos (HSV).

 - `/base_detection/debug_yolo` → caixas e rótulos do YOLO.

### Fusão de detectores
Depois de obter as listas de detecções, o nó monta uma lista de candidatos com:
 - `type` (1=quadrado, 2=círculo, 3=yolo),
 - `cx, cy,`
 - `score` (confiança do detector),
 - `score_eff` = score * bias_do_detector.

A fusão é feita por agrupamento espacial (single-link-like):
 - Agrupa centróides cuja distância euclidiana seja ≤
`max(fusion.max_dist_px, fusion.max_dist_frac * diagonal_da_imagem)`.
 - Para cada cluster, escolhe apenas uma detecção: a de maior score_eff.
Assim, você pode priorizar um detector ajustando `bias.square`, `bias.circle`, `bias.yolo`.
Se `fusion.enable=false`, não agrupa: publica todos os candidatos aprovados no bias.


### Pré-processamento (aplicado somente aos detectores HSV)

1. `gaussian_blur(kernel_blur_size)`
2. `clahe_bgr(clip_limit, tile_grid)`
3. `hsv_mask` com faixas configuráveis:
   - `blue_hsv_lower/upper`
   - `yellow_hsv_lower/upper`
4. `morph_open_close(open_k, close_k)`

## Formatos publicados

 - `detected_coords_topic`:
   - `Float32MultiArray` com `N` detecções → comprimento 5*N.
  Cada detecção: `[x1, y1, x2, y2, score]` (em pixels).

 - `centroids_topic`:
   - `Float32MultiArray` com `N` centróides → comprimento 2*N.
Cada centróide: `[cx, cy]` (em pixels).

- num_bases_topic: Int32 com `N`.


## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if available), or refer to the standard MIT License text.
