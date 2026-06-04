#!/bin/bash

mkdir -p ~/ros1_ws/src

xhost +local:docker

docker run -it  \
  --gpus all \
  --name morai_rl \
  --net=host \
  --pid=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v ~/ros1_ws:/root/catkin_ws:rw \
  -v /dev/shm:/dev/shm:rw \
  -v /dev/bus/usb:/dev/bus/usb:rw \
  -v /dev/input:/dev/input:rw \
  -v /run/udev:/run/udev:ro \
  misys:RL
