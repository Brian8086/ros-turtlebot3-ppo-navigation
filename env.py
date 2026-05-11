import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS_PATH = PROJECT_DIR / "tasks_stage4.json"


@dataclass
class TaskSpec:
    start: Tuple[float, float, float]
    goal: Tuple[float, float]


@dataclass
class RenderConfig:
    margin: int = 32
    line_width: int = 6
    point_radius: int = 4
    collision_mark_half: int = 6
    collision_mark_width: int = 2
    start_goal_font_size: int = 22
    bg_color: Tuple[int, int, int] = (250, 250, 250)
    obstacle_color: Tuple[int, int, int] = (30, 30, 30)
    observed_obstacle_color: Tuple[int, int, int] = (110, 110, 110)
    observed_obstacle_radius: int = 2
    grid_color: Tuple[int, int, int] = (225, 225, 225)
    start_color: Tuple[int, int, int] = (70, 130, 255)
    goal_color: Tuple[int, int, int] = (235, 60, 60)


@dataclass
class RosTurtleBot3Config:
    node_name: str = "tb3_rl_env"
    robot_entity_name: str = "burger"
    cmd_vel_topic: str = "cmd_vel"
    scan_topic: str = "scan"
    odom_topic: str = "odom"
    reset_world_service: str = "/reset_world"
    spawn_entity_service: str = "/spawn_entity"
    delete_entity_service: str = "/delete_entity"
    pause_physics_service: str = "/pause_physics"
    unpause_physics_service: str = "/unpause_physics"
    model_sdf_path: str = ""
    spawn_z: float = 0.05
    service_timeout_sec: float = 40.0
    control_rate_hz: float = 10.0
    
    action_duration: float = 0.35
    settle_duration: float = 0.08
    max_steps: int = 120
    lidar_bins: int = 96
    scan_clip_max: float = 3.5
    collision_distance: float = 0.20
    goal_tolerance: float = 0.25
    step_penalty: float = -0.02
    progress_reward_scale: float = 1.2
    collision_penalty: float = -1.5
    success_reward: float = 8.0
    terminate_on_collision: bool = True
    linear_speed: float = 0.12
    angular_speed: float = 0.65
    small_turn_scale: float = 0.55
    idle_after_reset_sec: float = 0.3
    startup_timeout_sec: float = 10.0
    wait_for_topics_timeout_sec: float = 10.0
    seed: int = 123
    map_yaml_path: str = ""
    tasks_json_path: str = ""
    default_view_half_span: float = 3.0


ACTION_SPECS: Dict[int, Tuple[float, float]] = {
    0: (0.80, 0.0),
    1: (0.45, 0.55),
    2: (0.45, -0.55),
    3: (0.0, 0.45),
    4: (0.0, -0.45),
}


def _try_load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _lerp_color(c1, c2, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _resample_scan(ranges: np.ndarray, target_bins: int) -> np.ndarray:
    if ranges.size == target_bins:
        return ranges.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=ranges.size, dtype=np.float32)
    x = np.linspace(0.0, 1.0, num=target_bins, dtype=np.float32)
    return np.interp(x, xp, ranges).astype(np.float32)


def _load_tasks(path: str) -> List[TaskSpec]:
    task_path = Path(path).expanduser() if path else DEFAULT_TASKS_PATH
    if not task_path.exists():
        raise FileNotFoundError(f"tasks json not found: {task_path}")
    with open(task_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks: List[TaskSpec] = []
    for item in raw:
        start = tuple(item["start"])
        goal = tuple(item["goal"])
        if len(start) != 3 or len(goal) != 2:
            raise ValueError(f"Bad task spec: {item}")
        tasks.append(TaskSpec(start=(float(start[0]), float(start[1]), float(start[2])), goal=(float(goal[0]), float(goal[1]))))
    if not tasks:
        raise ValueError("tasks json contains no tasks")
    return tasks


def _load_map_yaml(path: str):
    if not path:
        return None
    if yaml is None:
        raise RuntimeError("PyYAML is required to load a map yaml. Install pyyaml first.")

    map_path = Path(path).expanduser().resolve()
    if not map_path.exists():
        raise FileNotFoundError(f"map yaml not found: {map_path}")

    with open(map_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    image_path = Path(meta["image"])
    if not image_path.is_absolute():
        image_path = (map_path.parent / image_path).resolve()
    else:
        image_path = image_path.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"map image not found: {image_path}")

    img = Image.open(str(image_path)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    if negate:
        occ_prob = arr.astype(np.float32) / 255.0
    else:
        occ_prob = 1.0 - arr.astype(np.float32) / 255.0

    occupancy = np.full(arr.shape, fill_value=-1, dtype=np.int8)
    occupancy[occ_prob >= occupied_thresh] = 100
    occupancy[occ_prob <= free_thresh] = 0

    return {
        "occupancy": occupancy,
        "resolution": float(meta["resolution"]),
        "origin": tuple(float(x) for x in meta["origin"][:3]),
        "width": int(occupancy.shape[1]),
        "height": int(occupancy.shape[0]),
    }


class TurtleBot3RosEnv:
    def __init__(self, config: Optional[RosTurtleBot3Config] = None):
        self.config = config or RosTurtleBot3Config()
        self.rng = np.random.default_rng(self.config.seed)
        self.tasks = _load_tasks(self.config.tasks_json_path)
        self.map_info = _load_map_yaml(self.config.map_yaml_path)
        self.num_actions = len(ACTION_SPECS)

        self.current_task: Optional[TaskSpec] = None
        self.current_goal: Tuple[float, float] = (0.0, 0.0)
        self.start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.steps = 0
        self.last_collision = False
        self._scan_msg = None
        self._odom_msg = None

        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan
            from std_srvs.srv import Empty
            from gazebo_msgs.srv import SpawnEntity, DeleteEntity
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ROS 2 Humble Python packages are not available in the current environment. "
                "Please source /opt/ros/humble/setup.bash and your turtlebot3 workspace before training."
            ) from exc

        self.rclpy = rclpy
        self.Node = Node
        self.Twist = Twist
        self.Odometry = Odometry
        self.LaserScan = LaserScan
        self.Empty = Empty
        self.SpawnEntity = SpawnEntity
        self.DeleteEntity = DeleteEntity

        if not self.rclpy.ok():
            self.rclpy.init(args=None)
        self.node = self.Node(self.config.node_name)

        qos = 10
        self.scan_sub = self.node.create_subscription(self.LaserScan, self.config.scan_topic, self._scan_cb, qos)
        self.odom_sub = self.node.create_subscription(self.Odometry, self.config.odom_topic, self._odom_cb, qos)
        self.cmd_pub = self.node.create_publisher(self.Twist, self.config.cmd_vel_topic, qos)

        self.reset_world_client = self.node.create_client(self.Empty, self.config.reset_world_service)
        self.spawn_entity_client = self.node.create_client(self.SpawnEntity, self.config.spawn_entity_service)
        self.delete_entity_client = self.node.create_client(self.DeleteEntity, self.config.delete_entity_service)
        self.pause_physics_client = self.node.create_client(self.Empty, self.config.pause_physics_service)
        self.unpause_physics_client = self.node.create_client(self.Empty, self.config.unpause_physics_service)

        self._wait_for_service(self.reset_world_client, self.config.startup_timeout_sec, self.config.reset_world_service)
        self._wait_for_service(self.spawn_entity_client, self.config.startup_timeout_sec, self.config.spawn_entity_service)
        self._wait_for_service(self.delete_entity_client, self.config.startup_timeout_sec, self.config.delete_entity_service)
        self._wait_for_service(self.pause_physics_client, self.config.startup_timeout_sec, self.config.pause_physics_service)
        self._wait_for_service(self.unpause_physics_client, self.config.startup_timeout_sec, self.config.unpause_physics_service)

        if not self.config.model_sdf_path:
            pkg_prefix = os.popen("ros2 pkg prefix turtlebot3_gazebo").read().strip()
            self.config.model_sdf_path = str(Path(pkg_prefix) / "share" / "turtlebot3_gazebo" / "models" / "turtlebot3_burger" / "model.sdf")

    def _scan_cb(self, msg):
        self._scan_msg = msg

    def _odom_cb(self, msg):
        self._odom_msg = msg

    def _spin_for(self, duration: float):
        end = time.time() + duration
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def _wait_for_service(self, client, timeout: float, name: str):
        start = time.time()
        while time.time() - start < timeout:
            if client.wait_for_service(timeout_sec=0.5):
                return
        raise RuntimeError(f"ROS service not available: {name}")

    def _wait_for_initial_topics(self, timeout: float):
        start = time.time()
        while time.time() - start < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            if self._scan_msg is not None and self._odom_msg is not None:
                return
        raise RuntimeError("Did not receive /scan and /odom in time. Start Gazebo + TurtleBot3 world first.")

    def _call_empty(self, client):
        req = self.Empty.Request()
        future = client.call_async(req)
        self._spin_until_future(future, timeout=self.config.service_timeout_sec)
        return future.result()

    def _spawn_entity(self, x: float, y: float, yaw: float):
        last_err = None
        for attempt in range(3):
            try:
                if not self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
                    raise RuntimeError("/spawn_entity service is not ready")

                req = self.SpawnEntity.Request()
                req.name = self.config.robot_entity_name
                req.xml = Path(self.config.model_sdf_path).read_text(encoding="utf-8")
                req.robot_namespace = ""
                req.reference_frame = "world"
                req.initial_pose.position.x = float(x)
                req.initial_pose.position.y = float(y)
                req.initial_pose.position.z = float(self.config.spawn_z)
                qx, qy, qz, qw = _yaw_to_quaternion(yaw)
                req.initial_pose.orientation.x = qx
                req.initial_pose.orientation.y = qy
                req.initial_pose.orientation.z = qz
                req.initial_pose.orientation.w = qw

                future = self.spawn_entity_client.call_async(req)
                self._spin_until_future(future, timeout=self.config.service_timeout_sec)
                result = future.result()
                if result is None or not getattr(result, "success", False):
                    msg = "" if result is None else getattr(result, "status_message", "")
                    raise RuntimeError(f"SpawnEntity failed: {msg}")
                return result
            except Exception as e:
                last_err = e
                try:
                    self._try_delete_entity(self.config.robot_entity_name)
                except Exception:
                    pass
                self._spin_for(0.6)

        raise RuntimeError(f"SpawnEntity failed after retries: {last_err}")

    def _try_delete_entity(self, name: str):
        req = self.DeleteEntity.Request()
        req.name = name
        future = self.delete_entity_client.call_async(req)
        self._spin_until_future(future, timeout=self.config.service_timeout_sec)
        return future.result()

    def _spin_until_future(self, future, timeout: float):
        start = time.time()
        while time.time() - start < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if future.done():
                return
        raise RuntimeError("ROS service call timed out")

    def _publish_twist(self, linear_x: float, angular_z: float):
        msg = self.Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def _drive_for(self, linear_x: float, angular_z: float, duration: float):
        """
        Hold the twist command for real wall-clock time.
        The previous implementation used spin_once(timeout=period) as if it were a sleep,
        but spin_once returns immediately whenever callbacks are ready, so commands could be
        published back-to-back and then stopped almost instantly.
        """
        period = 1.0 / max(self.config.control_rate_hz, 1e-6)
        end_t = time.monotonic() + max(duration, period)
        next_t = time.monotonic()

        while time.monotonic() < end_t:
            self._publish_twist(linear_x, angular_z)
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

            next_t += period
            sleep_dt = next_t - time.monotonic()
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)
            else:
                next_t = time.monotonic()

        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def _stop_robot(self):
        for _ in range(3):
            self._publish_twist(0.0, 0.0)
            self._spin_for(0.03)

    def sample_task(self):
        self.current_task = self.tasks[int(self.rng.integers(0, len(self.tasks)))]
        self.start_pose = self.current_task.start
        self.current_goal = self.current_task.goal

    def build_map_id(self) -> str:
        import hashlib
        payload = {
            "map_yaml_path": self.config.map_yaml_path,
            "start": [float(x) for x in self.start_pose],
            "goal": [float(x) for x in self.current_goal],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def _current_pose(self) -> Tuple[float, float, float]:
        if self._odom_msg is None:
            raise RuntimeError("No odometry received")
        pose = self._odom_msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = _quaternion_to_yaw(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        return x, y, yaw

    def _goal_relative(self) -> Tuple[float, float]:
        x, y, yaw = self._current_pose()
        dx = self.current_goal[0] - x
        dy = self.current_goal[1] - y
        goal_dist = math.hypot(dx, dy)
        goal_heading = _wrap_angle(math.atan2(dy, dx) - yaw)
        return goal_dist, goal_heading

    def _scan_array(self) -> np.ndarray:
        if self._scan_msg is None:
            raise RuntimeError("No laser scan received")
        msg = self._scan_msg
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        max_range = float(msg.range_max) if math.isfinite(msg.range_max) and msg.range_max > 0 else self.config.scan_clip_max
        ranges[~np.isfinite(ranges)] = max_range
        ranges = np.clip(ranges, 0.0, min(self.config.scan_clip_max, max_range))
        return _resample_scan(ranges, self.config.lidar_bins)

    def _observed_obstacle_points(self, dedup_resolution: float = 0.05) -> List[Tuple[float, float]]:
        if self._scan_msg is None or self._odom_msg is None:
            return []
        msg = self._scan_msg
        scan = self._scan_array()
        if scan.size == 0:
            return []

        angle_min = float(msg.angle_min) if math.isfinite(msg.angle_min) else -math.pi
        angle_max = float(msg.angle_max) if math.isfinite(msg.angle_max) and msg.angle_max > angle_min else (angle_min + 2.0 * math.pi)
        angles = np.linspace(angle_min, angle_max, num=scan.size, dtype=np.float32)
        x, y, yaw = self._current_pose()

        msg_max = float(msg.range_max) if math.isfinite(msg.range_max) and msg.range_max > 0 else self.config.scan_clip_max
        valid_max = min(self.config.scan_clip_max, msg_max) - 0.03
        points: List[Tuple[float, float]] = []
        seen = set()
        for rng, ang in zip(scan.tolist(), angles.tolist()):
            if rng <= 0.05 or rng >= valid_max:
                continue
            wx = float(x + rng * math.cos(yaw + ang))
            wy = float(y + rng * math.sin(yaw + ang))
            key = (int(round(wx / dedup_resolution)), int(round(wy / dedup_resolution)))
            if key in seen:
                continue
            seen.add(key)
            points.append((wx, wy))
        return points

    def get_obs(self) -> np.ndarray:
        scan = self._scan_array()
        norm_scan = scan / max(self.config.scan_clip_max, 1e-6)
        inv_prox = 1.0 - norm_scan
        goal_dist, goal_heading = self._goal_relative()
        goal_dist_norm = np.clip(goal_dist / 5.0, 0.0, 1.0)

        channels = np.zeros((6, self.config.lidar_bins), dtype=np.float32)
        channels[0] = norm_scan
        channels[1] = inv_prox
        channels[2].fill(goal_dist_norm)
        channels[3].fill(math.sin(goal_heading))
        channels[4].fill(math.cos(goal_heading))
        channels[5].fill(1.0 if self.last_collision else 0.0)
        return channels

    def reset(self, regenerate: bool = True) -> np.ndarray:
        if regenerate or self.current_task is None:
            self.sample_task()

        self._scan_msg = None
        self._odom_msg = None
        self._stop_robot()

        # Use delete + spawn for stable resets in the static Gazebo world.
        try:
            self._try_delete_entity(self.config.robot_entity_name)
        except Exception:
            pass

        # Give Gazebo a short moment to remove the previous robot entity.
        self._spin_for(0.60)

        self._spawn_entity(*self.start_pose)

        # Wait for sensors and plugins to start publishing.
        self._spin_for(max(0.50, self.config.idle_after_reset_sec))
        self._wait_for_initial_topics(timeout=self.config.wait_for_topics_timeout_sec)

        self._stop_robot()
        self.steps = 0
        self.last_collision = False
        return self.get_obs()

    def step(self, action: int):
        if action not in ACTION_SPECS:
            raise ValueError(f"Invalid action {action}; valid actions are {sorted(ACTION_SPECS)}")

        prev_goal_dist, _ = self._goal_relative()
        prev_min_scan = float(np.min(self._scan_array()))

        lin_scale, ang_scale = ACTION_SPECS[action]
        linear_x = self.config.linear_speed * lin_scale
        angular_z = self.config.angular_speed * ang_scale

        if abs(lin_scale) > 1e-6 and abs(ang_scale) > 1e-6:
            angular_z *= self.config.small_turn_scale

        # 靠墙时不允许继续前冲，避免顶墙翻滚
        if prev_min_scan < (self.config.collision_distance + 0.03) and linear_x > 0.0:
            linear_x = 0.0

        self._drive_for(linear_x, angular_z, self.config.action_duration)
        self._stop_robot()
        self._spin_for(self.config.settle_duration)

        self.steps += 1
        x, y, yaw = self._current_pose()
        goal_dist, goal_heading = self._goal_relative()
        min_scan = float(np.min(self._scan_array()))
        collided = bool(min_scan <= self.config.collision_distance)

        reward = float(self.config.step_penalty)
        reward += float(self.config.progress_reward_scale * (prev_goal_dist - goal_dist))

        done = False
        success = False

        if collided:
            reward += float(self.config.collision_penalty)
            self._stop_robot()
            if self.config.terminate_on_collision:
                done = True

        if goal_dist <= self.config.goal_tolerance:
            reward += float(self.config.success_reward)
            done = True
            success = True
        elif self.steps >= self.config.max_steps:
            done = True

        self.last_collision = collided
        info = {
            "pos": (round(x, 3), round(y, 3)),
            "yaw": float(yaw),
            "goal_dist": float(goal_dist),
            "goal_heading": float(goal_heading),
            "collided": collided,
            "success": success,
            "min_scan": min_scan,
            "start": self.start_pose,
            "goal": self.current_goal,
            "map_id": self.build_map_id(),
            "observed_obstacle_points": self._observed_obstacle_points(),
        }
        return self.get_obs(), reward, done, info

        

    def _map_bounds(self, path_positions: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in path_positions] + [self.current_goal[0], self.start_pose[0]]
        ys = [p[1] for p in path_positions] + [self.current_goal[1], self.start_pose[1]]

        if self.map_info is not None:
            ox, oy, _ = self.map_info["origin"]
            resolution = self.map_info["resolution"]
            width = self.map_info["width"]
            height = self.map_info["height"]
            xs.extend([ox, ox + width * resolution])
            ys.extend([oy, oy + height * resolution])
            pad = 0.6
            return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

        cx = 0.5 * (self.start_pose[0] + self.current_goal[0])
        cy = 0.5 * (self.start_pose[1] + self.current_goal[1])
        half = self.config.default_view_half_span
        return cx - half, cx + half, cy - half, cy + half

    def _world_to_canvas(self, x: float, y: float, x_min: float, y_max: float, scale: float, margin: int) -> Tuple[int, int]:
        px = int(round((x - x_min) * scale)) + margin
        py = int(round((y_max - y) * scale)) + margin
        return px, py

    def _world_to_canvas_flip_x(self, x: float, y: float, x_max: float, y_max: float, scale: float, margin: int) -> Tuple[int, int]:
        px = int(round((x_max - x) * scale)) + margin
        py = int(round((y_max - y) * scale)) + margin
        return px, py

    def render_trajectory_image(self, meta_list: Sequence[Any], save_path: str, config: Optional[RenderConfig] = None) -> str:
        config = config or RenderConfig()
        path_positions = [self.start_pose[:2]] + [tuple(m.pos) for m in meta_list]
        observed_points = [tuple(p) for m in meta_list for p in getattr(m, "obstacle_points", ())]
        if self.map_info is None and observed_points:
            xs = [p[0] for p in path_positions] + [self.current_goal[0], self.start_pose[0]] + [p[0] for p in observed_points]
            ys = [p[1] for p in path_positions] + [self.current_goal[1], self.start_pose[1]] + [p[1] for p in observed_points]
            pad = 0.6
            x_min, x_max, y_min, y_max = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        else:
            x_min, x_max, y_min, y_max = self._map_bounds(path_positions)
        span_x = max(1e-6, x_max - x_min)
        span_y = max(1e-6, y_max - y_min)
        target_size = 800
        scale = min(
            (target_size - 2 * config.margin) / span_x,
            (target_size - 2 * config.margin) / span_y,
        )
        width = int(round(span_x * scale)) + 2 * config.margin
        height = int(round(span_y * scale)) + 2 * config.margin

        img = Image.new("RGB", (max(width, 400), max(height, 400)), config.bg_color)
        draw = ImageDraw.Draw(img)

        grid_step = 0.5
        gx = math.floor(x_min / grid_step) * grid_step
        while gx <= x_max + 1e-9:
            p0 = self._world_to_canvas(gx, y_min, x_min, y_max, scale, config.margin)
            p1 = self._world_to_canvas(gx, y_max, x_min, y_max, scale, config.margin)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=config.grid_color, width=1)
            gx += grid_step
        gy = math.floor(y_min / grid_step) * grid_step
        while gy <= y_max + 1e-9:
            p0 = self._world_to_canvas(x_min, gy, x_min, y_max, scale, config.margin)
            p1 = self._world_to_canvas(x_max, gy, x_min, y_max, scale, config.margin)
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=config.grid_color, width=1)
            gy += grid_step

        if self.map_info is not None:
            occ = self.map_info["occupancy"]
            res = self.map_info["resolution"]
            ox, oy, _ = self.map_info["origin"]
            for row in range(occ.shape[0]):
                for col in range(occ.shape[1]):
                    if int(occ[row, col]) < 50:
                        continue
                    wx0 = ox + col * res
                    wy1 = oy + (occ.shape[0] - row) * res
                    wx1 = wx0 + res
                    wy0 = wy1 - res
                    px0, py1 = self._world_to_canvas(wx0, wy0, x_min, y_max, scale, config.margin)
                    px1, py0 = self._world_to_canvas(wx1, wy1, x_min, y_max, scale, config.margin)
                    draw.rectangle((px0, py0, px1, py1), fill=config.obstacle_color)
        elif observed_points:
            seen = set()
            rr = max(1, int(config.observed_obstacle_radius))
            for wx, wy in observed_points:
                key = (round(wx, 2), round(wy, 2))
                if key in seen:
                    continue
                seen.add(key)
                px, py = self._world_to_canvas_flip_x(wx, wy, x_max, y_max, scale, config.margin)
                draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=config.observed_obstacle_color)

        centers = [self._world_to_canvas_flip_x(x, y, x_max, y_max, scale, config.margin) for x, y in path_positions]
        num_segments = max(1, len(centers) - 1)
        for i in range(len(centers) - 1):
            color = _lerp_color(config.start_color, config.goal_color, i / max(1, num_segments - 1))
            draw.line([centers[i], centers[i + 1]], fill=color, width=max(config.line_width, 6))

        for i, p in enumerate(centers):
            color = _lerp_color(config.start_color, config.goal_color, i / max(1, len(centers) - 1))
            rr = max(config.point_radius, 4) + (2 if i in {0, len(centers) - 1} else 0)
            draw.ellipse((p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr), fill=color, outline=(0, 0, 0))

        half = config.collision_mark_half
        for m in meta_list:
            if getattr(m, "collided", False):
                cx, cy = self._world_to_canvas_flip_x(m.pos[0], m.pos[1], x_max, y_max, scale, config.margin)
                draw.line((cx - half, cy - half, cx + half, cy + half), fill=(200, 0, 0), width=config.collision_mark_width)
                draw.line((cx - half, cy + half, cx + half, cy - half), fill=(200, 0, 0), width=config.collision_mark_width)

        font = _try_load_font(config.start_goal_font_size)
        start_px = self._world_to_canvas_flip_x(self.start_pose[0], self.start_pose[1], x_max, y_max, scale, config.margin)
        goal_px = self._world_to_canvas_flip_x(self.current_goal[0], self.current_goal[1], x_max, y_max, scale, config.margin)
        draw.ellipse((start_px[0] - 8, start_px[1] - 8, start_px[0] + 8, start_px[1] + 8), fill=(70, 130, 255), outline=(0, 0, 0))
        draw.ellipse((goal_px[0] - 8, goal_px[1] - 8, goal_px[0] + 8, goal_px[1] + 8), fill=(235, 60, 60), outline=(0, 0, 0))
        draw.text((start_px[0] - 22, start_px[1] + 10), "S", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        draw.text((goal_px[0] + 10, goal_px[1] - 24), "G", fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path)
        return save_path

    def close(self):
        try:
            self._stop_robot()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.rclpy.ok():
            self.rclpy.shutdown()
