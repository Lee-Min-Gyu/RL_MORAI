#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from morai_standard.network.ros_manager import run_node
from morai_standard.autonomous_driving.autonomous_driving import AutonomousDriving


def main():
    autonomous_driving = AutonomousDriving()
    node = run_node(autonomous_driving)

if __name__ == '__main__':
    main()
