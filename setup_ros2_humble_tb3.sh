#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-gazebo-* \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  git curl

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init || true
fi
rosdep update

mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src

clone_if_missing () {
  local branch="$1"
  local repo="$2"
  local dir="$3"
  if [ ! -d "$dir" ]; then
    git clone -b "$branch" "$repo" "$dir"
  fi
}

clone_if_missing humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git DynamixelSDK
clone_if_missing humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git turtlebot3_msgs
clone_if_missing humble https://github.com/ROBOTIS-GIT/turtlebot3.git turtlebot3
clone_if_missing humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git turtlebot3_simulations

cd ~/turtlebot3_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

grep -qxF 'source /opt/ros/humble/setup.bash' ~/.bashrc || echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
grep -qxF 'source ~/turtlebot3_ws/install/setup.bash' ~/.bashrc || echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
grep -qxF 'export ROS_DOMAIN_ID=30' ~/.bashrc || echo 'export ROS_DOMAIN_ID=30' >> ~/.bashrc
grep -qxF 'export TURTLEBOT3_MODEL=burger' ~/.bashrc || echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc

echo
echo "Setup complete. Open a new shell or run:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ~/turtlebot3_ws/install/setup.bash"
