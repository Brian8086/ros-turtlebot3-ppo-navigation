#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
python3 train.py --episodes "${1:-200}"
