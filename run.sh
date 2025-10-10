xhost +local:root
docker run --rm -it \
  --name base_detection_marcelo \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$(pwd)":/base_detection/src/base_detection:rw \
  -w /base_detection \
  base_detection_marcelo bash
