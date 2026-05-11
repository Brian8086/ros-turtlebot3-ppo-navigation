import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from env import RosTurtleBot3Config, TurtleBot3RosEnv
from model import LidarGoalActorCritic
from ppo import RolloutBuffer, ppo_update


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


@dataclass
class StepMeta:
    pos: Tuple[float, float]
    action: int
    reward: float
    goal_dist: float
    collided: bool
    done: bool
    obstacle_points: Tuple[Tuple[float, float], ...] = ()


@dataclass
class EpisodeStats:
    success: bool
    steps: int
    total_reward: float
    final_goal_dist: float
    min_goal_dist: float
    collision_count: int
    path_length: float
    avg_entropy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: Sequence[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    if window > 1:
        smoothed = np.concatenate([np.full(window - 1, smoothed[0], dtype=np.float32), smoothed])
    return smoothed


def build_action_hist(actions: Sequence[int], num_actions: int) -> Dict[str, int]:
    hist = {str(i): 0 for i in range(num_actions)}
    for action in actions:
        hist[str(int(action))] += 1
    return hist


def collect_episode(env: TurtleBot3RosEnv, model, device: str) -> Dict:
    obs_np = env.reset(regenerate=True)
    done = False

    buffer = RolloutBuffer()
    meta_list: List[StepMeta] = []
    rewards: List[float] = []
    entropies: List[float] = []

    start_pose = tuple(float(x) for x in env.start_pose[:3])
    goal = tuple(float(x) for x in env.current_goal[:2])
    previous_pos = start_pose[:2]
    path_length = 0.0

    while not done:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, value, entropy, _, _ = model.act(obs, return_stats=True)

        next_obs_np, reward, done, info = env.step(int(action.item()))
        current_pos = tuple(float(x) for x in info["pos"])
        path_length += float(np.hypot(current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1]))
        previous_pos = current_pos

        buffer.obs.append(obs.squeeze(0))
        buffer.actions.append(int(action.item()))
        buffer.logprobs.append(logprob.squeeze(0))
        buffer.values.append(value.squeeze(0))
        buffer.rewards.append(float(reward))
        buffer.dones.append(bool(done))

        rewards.append(float(reward))
        entropies.append(float(entropy.squeeze(0).item()))
        meta_list.append(
            StepMeta(
                pos=current_pos,
                action=int(action.item()),
                reward=float(reward),
                goal_dist=float(info["goal_dist"]),
                collided=bool(info["collided"]),
                done=bool(done),
                obstacle_points=tuple(tuple(p) for p in info.get("observed_obstacle_points", [])),
            )
        )
        obs_np = next_obs_np

    final_goal_dist = float(meta_list[-1].goal_dist) if meta_list else float("inf")
    min_goal_dist = float(min(m.goal_dist for m in meta_list)) if meta_list else float("inf")
    success = bool(meta_list[-1].done and final_goal_dist <= env.config.goal_tolerance) if meta_list else False
    collision_count = int(sum(int(m.collided) for m in meta_list))

    stats = EpisodeStats(
        success=success,
        steps=len(meta_list),
        total_reward=float(sum(rewards)),
        final_goal_dist=final_goal_dist,
        min_goal_dist=min_goal_dist,
        collision_count=collision_count,
        path_length=float(path_length),
        avg_entropy=float(np.mean(entropies)) if entropies else 0.0,
    )

    return {
        "buffer": buffer,
        "meta_list": meta_list,
        "stats": stats,
        "start_pose": start_pose,
        "goal": goal,
        "final_pose": meta_list[-1].pos if meta_list else start_pose[:2],
        "action_hist": build_action_hist(buffer.actions, env.num_actions),
    }


def plot_training_curves(metrics: Dict[str, List[float]], smooth_window: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_specs = [
        ("success_curve.png", "success", "Success", "Success", (-0.02, 1.02)),
        ("return_curve.png", "env_return", "Environment Return", "Return", None),
        ("episode_length_curve.png", "steps", "Episode Length", "Steps", None),
        ("goal_distance_curve.png", "final_goal_dist", "Final Goal Distance", "Distance", None),
    ]

    for filename, key, title, ylabel, ylim in plot_specs:
        y = np.asarray(metrics[key], dtype=np.float32)
        x = np.arange(1, len(y) + 1)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(x, y, alpha=0.25, linewidth=0.9, label="raw")
        ax.plot(x, moving_average(y, smooth_window), linewidth=2.2, label=f"MA-{smooth_window}")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=220, bbox_inches="tight")
        plt.close(fig)


def train(args) -> Dict[str, List[float]]:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device = {device}")

    map_yaml_path = args.map_yaml or os.environ.get(
        "TB3_MAP_YAML",
        os.path.join(PROJECT_DIR, "maps", "stage4_map_clean.yaml"),
    )
    tasks_json_path = args.tasks_json or os.environ.get(
        "TB3_TASKS_JSON",
        os.path.join(PROJECT_DIR, "tasks_stage4.json"),
    )

    env = TurtleBot3RosEnv(
        RosTurtleBot3Config(
            robot_entity_name=args.entity_name,
            map_yaml_path=map_yaml_path,
            tasks_json_path=tasks_json_path,
            seed=args.seed,
            max_steps=args.max_steps,
            linear_speed=args.linear_speed,
            angular_speed=args.angular_speed,
            action_duration=args.action_duration,
        )
    )

    model = LidarGoalActorCritic(
        in_channels=6,
        scan_bins=env.config.lidar_bins,
        num_actions=env.num_actions,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rollout = RolloutBuffer()
    metrics = {
        "success": [],
        "env_return": [],
        "steps": [],
        "final_goal_dist": [],
    }

    log_path = os.path.join(RESULTS_DIR, "ppo_episode_metrics.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)

    last_update_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    try:
        for episode in range(1, args.episodes + 1):
            record = collect_episode(env, model, device)
            stats: EpisodeStats = record["stats"]
            rollout.extend(record["buffer"])

            image_path = ""
            if args.render_every > 0 and (episode == 1 or episode % args.render_every == 0):
                image_path = os.path.join(RESULTS_DIR, "trajectory_images", f"episode_{episode:06d}.png")
                env.render_trajectory_image(record["meta_list"], image_path)

            if len(rollout) >= args.update_steps or episode == args.episodes:
                last_update_info = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    buffer=rollout,
                    clip_eps=args.clip_eps,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    max_grad_norm=args.max_grad_norm,
                )
                rollout.clear()

            metrics["success"].append(1.0 if stats.success else 0.0)
            metrics["env_return"].append(stats.total_reward)
            metrics["steps"].append(float(stats.steps))
            metrics["final_goal_dist"].append(stats.final_goal_dist)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "episode": episode,
                    "success": stats.success,
                    "env_return": stats.total_reward,
                    "steps": stats.steps,
                    "start_pose": record["start_pose"],
                    "goal": record["goal"],
                    "final_pose": record["final_pose"],
                    "final_goal_dist": stats.final_goal_dist,
                    "min_goal_dist": stats.min_goal_dist,
                    "collision_count": stats.collision_count,
                    "path_length": stats.path_length,
                    "avg_entropy": stats.avg_entropy,
                    "action_hist": record["action_hist"],
                    "policy_loss": last_update_info.get("policy_loss"),
                    "value_loss": last_update_info.get("value_loss"),
                    "update_entropy": last_update_info.get("entropy"),
                    "trajectory_image_path": image_path,
                }, ensure_ascii=False) + "\n")

            if episode == 1 or episode % args.print_every == 0:
                recent = min(args.print_every, len(metrics["success"]))
                print(
                    f"Episode {episode:04d} | "
                    f"success={np.mean(metrics['success'][-recent:]):.3f} | "
                    f"return={np.mean(metrics['env_return'][-recent:]):.3f} | "
                    f"steps={np.mean(metrics['steps'][-recent:]):.1f} | "
                    f"final_dist={stats.final_goal_dist:.3f} | "
                    f"collisions={stats.collision_count} | "
                    f"entropy={last_update_info.get('entropy', 0.0):.3f}"
                )
    finally:
        env.close()

    plot_training_curves(metrics, smooth_window=args.smooth_window)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for TurtleBot3 navigation in Gazebo stage4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--map-yaml", type=str, default="")
    parser.add_argument("--tasks-json", type=str, default="")
    parser.add_argument("--entity-name", type=str, default="burger")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--linear-speed", type=float, default=0.12)
    parser.add_argument("--angular-speed", type=float, default=0.65)
    parser.add_argument("--action-duration", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-steps", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--render-every", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
