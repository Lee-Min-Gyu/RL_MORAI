[![MORAILog](./docs/MORAI_Logo.png)](https://www.morai.ai)
===
# MORAI - Drive example (ROS2)

First step to enjoy the `MORAI Sim: Drive` with ROS2.

This example support Ubuntu 20.04 or later
```
./
├── morai_ros2_msgs              # [ROS2 msgs] MORAI Simulator ROS2 message set
└── morai_standard               # [Simulator Example] MORAI Sim: Drive example project
     ├── launch                    # example launch files
     ├── rviz                      # rviz preset configuration file
     └── morai_standard              # example script files
          ├── autonomous_driving     # [Autonomous Driving] autonomous driving module
          ├── network                # ROS2 network connection
          └── main.py                # [Entry] example excuter
```

These example contains the below list.
  - Trajectory following lateral control
  - Smart(adaptive) Cruise Control
  - ROS2 communication

# Requirement

- ROS2 desktop-full >= foxy

- python >= 3.8

# Installation

Install packages which basically need

```
mkdir -p ~/colcon_ws/src
cd ~/colcon_ws/src
git clone https://github.com/MORAI-Autonomous/MORAI-DriveExample_ROS2.git
cd MORAI-DriveExample_ROS2
git submodule update --init --recursive
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash
```

# Usage

Enjoy the example which follow the trajectory with smart cruise control.
```
ros2 launch morai_standard morai_standard.launch.py
```

# License
- MORAI Drive Example license info:  [Drive Example License](./docs/License.md)
- MORAI Autonomous Driving license info: [Autonomous Driving License](./morai_standard/morai_standard/autonomous_driving/docs/License.md)
- MGeo Module license info: [MGeo module License](./morai_standard/morai_standard/autonomous_driving/mgeo/lib/mgeo/docs/License.md)

