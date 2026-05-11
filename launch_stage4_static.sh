#!/usr/bin/env bash
set -eo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

export TURTLEBOT3_MODEL=${TURTLEBOT3_MODEL:-burger}
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}

TB3_PKG_PREFIX=$(ros2 pkg prefix turtlebot3_gazebo)

gzserver --verbose \
  "$TB3_PKG_PREFIX/share/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage4_static.world" \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so
