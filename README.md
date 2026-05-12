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

