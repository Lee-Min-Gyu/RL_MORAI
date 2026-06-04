FROM osrf/ros:noetic-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
SHELL ["/bin/bash", "-c"]

# 기본 시스템 / ROS / 개발 패키지
RUN apt update && apt install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    vim \
    curl \
    wget \
    net-tools \
    iputils-ping \
    python3-catkin-tools \
    python3-rospkg \
    python3-catkin-pkg \
    python3-rosdep \
    python3-vcstool \
    ros-noetic-rospy \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-catkin \
    ros-noetic-rosbridge-server \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# rosdep 업데이트
RUN rosdep update || true

# RL용 Python 3.8 가상환경
RUN python3 -m venv /opt/rl_venv

# pip 업데이트
RUN source /opt/rl_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel 

#netifaces 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install netifaces

# PyTorch CUDA 11.8 wheel 먼저 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

#Scikit-learn 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install scikit-learn

#empy 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install empy==3.3.4

#pyproj 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install pyproj

#catkin_pkg 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install catkin_pkg rospkg

#twisted 설치
RUN source /opt/rl_venv/bin/activate && \
    pip install twisted \
    autobahn \
    tornado \
    bson \
    pymongo

# RL 패키지 버전 고정
RUN source /opt/rl_venv/bin/activate && \
    pip install \
    numpy==1.24.4 \
    gymnasium==0.29.1 \
    "stable-baselines3[extra]==2.4.1" \
    tensorboard \
    opencv-python-headless==4.10.0.84 \
    pygame==2.6.1 \
    tomli

# catkin workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# ROS-MORAI setting
RUN cd /root/catkin_ws/src && \
    git clone --branch 26.R1 --recurse-submodules https://github.com/MORAI-Autonomous/MORAI-DriveExample_ROS.git && \
    cd .. && \
    source /opt/ros/noetic/setup.bash && \
    catkin_make

# bash 시작 시 자동 source
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /opt/rl_venv/bin/activate" >> /root/.bashrc && \
    echo "export ROS_MASTER_URI=http://localhost:11311" >> /root/.bashrc && \
    echo "export ROS_HOSTNAME=localhost" >> /root/.bashrc && \
    echo "alias underlay='source /opt/ros/noetic/setup.bash' >>/root/.bashrc && \
    echo "alias overlay='source devel/setup.bash' >> /root/.bashrc

CMD ["bash"]
