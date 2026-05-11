# TurtleBot3 PPO Navigation on Gazebo Stage4

This project trains a TurtleBot3 Burger robot to navigate from a fixed start pose to a target position in the Gazebo `turtlebot3_dqn_stage4_static.world` environment using Proximal Policy Optimization (PPO).

## Files

- `train.py`: PPO training entry point.
- `ppo.py`: PPO update with clipped policy loss and GAE.
- `model.py`: LiDAR-goal actor-critic policy network.
- `env.py`: ROS2/Gazebo TurtleBot3 environment wrapper.
- `tasks_stage4.json`: start and goal configuration.
- `maps/stage4_map_clean.yaml`: map used for trajectory rendering.
- `launch_stage4_static.sh`: starts the Gazebo stage4 static world.
- `setup_ros2_humble_tb3.sh`: optional ROS2 Humble and TurtleBot3 dependency setup script.

## Basic usage

Terminal 1:

```bash
cd ros-turtlebot3-ppo-navigation
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
bash launch_stage4_static.sh
```

Terminal 2:

```bash
cd ros-turtlebot3-ppo-navigation
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
python3 train.py --episodes 200
```

Results are written to `results/`, including episode logs, trajectory images, and training curves.

## Observation and action space

The policy observes a compact LiDAR-goal representation with 96 LiDAR bins and goal-relative features. The action space has five discrete actions: move forward, forward-left, forward-right, rotate-left, and rotate-right.

## Reward

The environment reward contains a small step penalty, progress reward toward the goal, collision penalty, and success reward when the robot reaches the target.

## 项目声明 Project Statement

本项目的作者及单位信息：

项目名称（Project Name）：ros-turtlebot3-ppo-navigation

项目作者（Author）：Tao Yang

作者单位（Affiliation）：暨南大学网络空间学院

项目说明（Description）：

本项目是一个基于 ROS2 Humble、Gazebo Classic 和 TurtleBot3 Burger 的移动机器人强化学习导航训练项目。项目使用 PPO（Proximal Policy Optimization）算法训练机器人在 TurtleBot3 stage4 仿真地图中完成从起点到目标点的自主导航任务。

项目主要包括 ROS/Gazebo 仿真环境封装、TurtleBot3 机器人控制接口、LiDAR 传感器状态构建、Actor-Critic 策略网络、PPO 训练算法、训练日志保存和轨迹可视化等模块。机器人通过订阅 `/scan` 和 `/odom` 获取激光雷达与位姿信息，并通过发布 `/cmd_vel` 控制运动，从而实现基于强化学习的端到端导航策略训练。

本项目主要用于移动机器人路径规划、强化学习导航算法验证、ROS2/Gazebo 仿真实验教学以及机器人智能控制相关研究。
