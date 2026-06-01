#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os, signal
from lib.controller import *
from lib.launcher_start_api import *
from lib.read_text import *

import rospkg
rospack = rospkg.RosPack()
morai_pkg_path = rospack.get_path('morai_standard')
workspace_src_path = os.path.abspath(os.path.join(morai_pkg_path, '..'))
if workspace_src_path not in sys.path:
    sys.path.insert(0, workspace_src_path)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../ros_drive/morai_standard/scripts"))
print(os.path.join(os.path.dirname(__file__), "../../ros_drive/morai_standard/scripts"))
from morai_standard.scripts.main import main as ros_main
"""
https://docs.google.com/spreadsheets/d/1jHbR_JoZFYfxMirwSp-48peWkJf1xUmMZyFIwetxcZM/edit#gid=0

"""

class api :

    def __init__(self):         

        signal.signal(signal.SIGINT, self.signal_handler) #handle ctrl-c
        
        api = launcher_start()
        api.launcher_start()

        ros_main()

    def signal_handler(self, signal, frame):        
        sys.exit(0)                                           

if __name__ == "__main__":
    start=api()
    
