from __future__ import annotations

import argparse
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium import spaces
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


DISEASES = ["GrayMold", "Viral", "Wilt"]
DISEASE_TO_ID = {name: idx for idx, name in enumerate(DISEASES)}
ID_TO_DISEASE = {idx: name for name, idx in DISEASE_TO_ID.items()}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    reward_mode: str
    max_steps: int
    step_penalty: float
    learning_rate: float
    gamma: float


VARIANTS = {
    "localization": VariantConfig(
        name="localization",
        reward_mode="localization",
        max_steps=8,
        step_penalty=0.005,
        learning_rate=3e-4,
        gamma=0.95,
    ),
    "balanced": VariantConfig(
        name="balanced",
        reward_mode="balanced",
        max_steps=8,
        step_penalty=0.01,
        learning_rate=2.5e-4,
        gamma=0.96,
    ),
    "efficient": VariantConfig(
        name="efficient",
        reward_mode="efficient",
        max_steps=6,
        step_penalty=0.03,
        learning_rate=2.5e-4,
        gamma=0.94,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO active visual attention experiments for tomato disease images."
    )
    parser.add_argument("--dataset", default="Tomato Disease Dataset", type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--timesteps", default=6000, type=int)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--eval-episodes", default=60, type=int)
    parser.add_argument(
        "--test-episodes",
        default=0,
        type=int,
        help="0 usa todo o conjunto de teste; valores positivos limitam a avaliacao final.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["localization", "balanced", "efficient"],
        choices=sorted(VARIANTS),
    )
    parser.add_argument("--n-envs", default=4, type=int)
    parser.add_argument("--image-size", default=96, type=int)
    parser.add_argument("--cache-side", default=384, type=int)
    parser.add_argument("--preload-images", action="store_true")
    parser.add_argument("--n-steps", default=128, type=int)
    parser.add_argument("--n-epochs", default=4, type=int)
    return parser.parse_args()


def read_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    boxes: list[list[float]] = []
    objects: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            [
                max(0.0, xmin / width),
                max(0.0, ymin / height),
                min(1.0, xmax / width),
                min(1.0, ymax / height),
            ]
        )
        objects.append(name)
    return {"width": width, "height": height, "boxes": boxes, "objects": objects}


def build_metadata(dataset_dir: Path, output_dir: Path, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for disease in DISEASES:
        image_dir = dataset_dir / "images" / disease
        ann_dir = dataset_dir / "annotations" / disease
        for xml_path in sorted(ann_dir.glob("*.xml")):
            image_path = image_dir / f"{xml_path.stem}.jpg"
            if not image_path.exists():
                matches = list(image_dir.glob(f"{xml_path.stem}.*"))
                if not matches:
                    continue
                image_path = matches[0]
            parsed = read_xml(xml_path)
            if not parsed["boxes"]:
                continue
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "xml_path": str(xml_path.resolve()),
                    "image_id": xml_path.stem,
                    "disease": disease,
                    "label": DISEASE_TO_ID[disease],
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "boxes_json": json.dumps(parsed["boxes"]),
                    "objects_json": json.dumps(parsed["objects"]),
                    "n_boxes": len(parsed["boxes"]),
                }
            )
    metadata = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(
        metadata,
        test_size=0.30,
        stratify=metadata["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )
    metadata["split"] = "unset"
    metadata.loc[train_df.index, "split"] = "train"
    metadata.loc[val_df.index, "split"] = "val"
    metadata.loc[test_df.index, "split"] = "test"
    metadata = metadata.sort_values(["split", "disease", "image_id"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def plot_dataset_summary(metadata: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=metadata, x="disease", order=DISEASES, hue="split")
    plt.title("Image distribution by class and split")
    plt.xlabel("Disease")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(fig_dir / "dataset_class_split.png", dpi=220)
    plt.close()

    objects = Counter()
    for raw in metadata["objects_json"]:
        objects.update(json.loads(raw))
    object_df = pd.DataFrame(objects.most_common(), columns=["object", "count"])
    object_df.to_csv(output_dir / "tables" / "object_counts.csv", index=False)
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=object_df, x="count", y="object", color="#4C78A8")
    plt.title("Expert-annotated objects")
    plt.xlabel("Number of boxes")
    plt.ylabel("XML label")
    plt.tight_layout()
    plt.savefig(fig_dir / "annotation_object_counts.png", dpi=220)
    plt.close()


def preload_images(metadata: pd.DataFrame, max_side: int = 384) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for image_path in sorted(metadata["image_path"].unique()):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        cache[image_path] = image
    return cache


def to_box(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array(
        [
            np.clip(cx - w / 2, 0.0, 1.0),
            np.clip(cy - h / 2, 0.0, 1.0),
            np.clip(cx + w / 2, 0.0, 1.0),
            np.clip(cy + h / 2, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def max_iou(box: np.ndarray, gt_boxes: np.ndarray) -> float:
    if gt_boxes.size == 0:
        return 0.0
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    area_b = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(
        0.0, gt_boxes[:, 3] - gt_boxes[:, 1]
    )
    union = area_a + area_b - inter + 1e-8
    return float(np.max(inter / union))


class TomatoAttentionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        variant: VariantConfig,
        seed: int = 42,
        image_size: int = 96,
        class_balanced: bool = True,
        image_cache: dict[str, np.ndarray] | None = None,
        cache_side: int = 384,
    ) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.variant = variant
        self.image_size = image_size
        self.class_balanced = class_balanced
        self.image_cache = image_cache if image_cache is not None else {}
        self.cache_side = cache_side
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.by_class = {
            label: self.df[self.df["label"] == label].index.to_list()
            for label in sorted(self.df["label"].unique())
        }
        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(40,), dtype=np.float32
        )
        self.current: pd.Series | None = None
        self.image: np.ndarray | None = None
        self.gt_boxes: np.ndarray | None = None
        self.label = 0
        self.cx = 0.5
        self.cy = 0.5
        self.w = 1.0
        self.h = 1.0
        self.steps = 0
        self.prev_iou = 0.0
        self.best_iou = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        if self.class_balanced:
            label = self.rng.choice(list(self.by_class))
            idx = self.rng.choice(self.by_class[label])
        else:
            idx = self.rng.randrange(len(self.df))
        self.current = self.df.iloc[idx]
        self.image = self._load_image(str(self.current["image_path"]))
        self.gt_boxes = np.array(json.loads(self.current["boxes_json"]), dtype=np.float32)
        self.label = int(self.current["label"])
        self.cx = 0.5
        self.cy = 0.5
        self.w = 1.0
        self.h = 1.0
        self.steps = 0
        self.prev_iou = max_iou(to_box(self.cx, self.cy, self.w, self.h), self.gt_boxes)
        self.best_iou = self.prev_iou
        return self._obs(), {}

    def _load_image(self, image_path: str) -> np.ndarray:
        cached = self.image_cache.get(image_path)
        if cached is not None:
            return cached
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(1.0, self.cache_side / max(height, width))
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        self.image_cache[image_path] = image
        return image

    def step(self, action: int):
        assert self.gt_boxes is not None
        self.steps += 1
        terminated = False
        truncated = False
        predicted: int | None = None
        correct = False

        if action <= 9:
            self._move_window(action)
        else:
            terminated = True
            predicted = action - 10
            correct = predicted == self.label

        if self.steps >= self.variant.max_steps and not terminated:
            terminated = True
            predicted = action % len(DISEASES)
            correct = predicted == self.label

        current_box = to_box(self.cx, self.cy, self.w, self.h)
        current_iou = max_iou(current_box, self.gt_boxes)
        self.best_iou = max(self.best_iou, current_iou)
        delta_iou = current_iou - self.prev_iou
        self.prev_iou = current_iou

        reward = self._reward(delta_iou, current_iou, terminated, correct)
        info = {
            "label": self.label,
            "label_name": ID_TO_DISEASE[self.label],
            "predicted": predicted,
            "predicted_name": ID_TO_DISEASE[predicted] if predicted is not None else None,
            "correct": correct,
            "current_iou": current_iou,
            "best_iou": self.best_iou,
            "steps": self.steps,
            "window_area": self.w * self.h,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _move_window(self, action: int) -> None:
        stride = 0.12 * max(self.w, self.h)
        if action == 0:
            self.cx -= stride
        elif action == 1:
            self.cx += stride
        elif action == 2:
            self.cy -= stride
        elif action == 3:
            self.cy += stride
        elif action == 4:
            self.w *= 0.78
            self.h *= 0.78
        elif action == 5:
            self.w *= 1.20
            self.h *= 1.20
        elif action == 6:
            self.w *= 1.18
        elif action == 7:
            self.w *= 0.82
        elif action == 8:
            self.h *= 1.18
        elif action == 9:
            self.h *= 0.82
        self.w = float(np.clip(self.w, 0.10, 1.0))
        self.h = float(np.clip(self.h, 0.10, 1.0))
        self.cx = float(np.clip(self.cx, self.w / 2, 1.0 - self.w / 2))
        self.cy = float(np.clip(self.cy, self.h / 2, 1.0 - self.h / 2))

    def _reward(
        self, delta_iou: float, current_iou: float, terminated: bool, correct: bool
    ) -> float:
        area = self.w * self.h
        if self.variant.reward_mode == "localization":
            reward = 3.0 * delta_iou + 0.25 * current_iou
        elif self.variant.reward_mode == "efficient":
            reward = 3.0 * delta_iou + 0.35 * current_iou - 0.08 * area
        else:
            reward = 2.5 * delta_iou + 0.45 * current_iou
        reward -= self.variant.step_penalty
        if terminated:
            reward += 1.5 if correct else -1.0
            reward += 0.7 * current_iou
            if self.variant.reward_mode == "efficient":
                reward += max(0.0, 0.25 - area)
        return reward

    def _obs(self) -> np.ndarray:
        assert self.image is not None
        crop = self._crop_attention()
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = crop.astype(np.float32) / 255.0
        mean = rgb.mean(axis=(0, 1))
        std = rgb.std(axis=(0, 1))
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist_parts = []
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [8], [0, 256]).flatten()
            hist = hist / max(1.0, hist.sum())
            hist_parts.extend((hist * 2.0 - 1.0).tolist())
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        state = np.array(
            [
                self.cx * 2 - 1,
                self.cy * 2 - 1,
                self.w * 2 - 1,
                self.h * 2 - 1,
                (self.steps / self.variant.max_steps) * 2 - 1,
                self.prev_iou * 2 - 1,
            ],
            dtype=np.float32,
        )
        visual = np.array(
            [
                *(mean * 2 - 1),
                *(std * 2 - 1),
                gray.mean() * 2 - 1,
                gray.std() * 2 - 1,
                np.clip(sobel.mean(), 0, 1) * 2 - 1,
                np.clip(sobel.std(), 0, 1) * 2 - 1,
                *hist_parts,
            ],
            dtype=np.float32,
        )
        return np.concatenate([state, visual]).astype(np.float32)

    def _crop_attention(self) -> np.ndarray:
        assert self.image is not None
        height, width = self.image.shape[:2]
        box = to_box(self.cx, self.cy, self.w, self.h)
        x1 = int(np.floor(box[0] * width))
        y1 = int(np.floor(box[1] * height))
        x2 = int(np.ceil(box[2] * width))
        y2 = int(np.ceil(box[3] * height))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return self.image[y1:y2, x1:x2]


def make_env(
    df: pd.DataFrame,
    variant: VariantConfig,
    seed: int,
    image_size: int,
    image_cache: dict[str, np.ndarray] | None,
    cache_side: int,
):
    def _factory():
        return Monitor(
            TomatoAttentionEnv(
                df,
                variant,
                seed=seed,
                image_size=image_size,
                image_cache=image_cache,
                cache_side=cache_side,
            )
        )

    return _factory


def evaluate_policy(
    model: PPO,
    df: pd.DataFrame,
    variant: VariantConfig,
    episodes: int,
    seed: int,
    image_size: int,
    image_cache: dict[str, np.ndarray] | None,
    cache_side: int,
) -> pd.DataFrame:
    env = TomatoAttentionEnv(
        df,
        variant=variant,
        seed=seed,
        image_size=image_size,
        class_balanced=False,
        image_cache=image_cache,
        cache_side=cache_side,
    )
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        last_info: dict[str, Any] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            last_info = info
        rows.append(
            {
                "episode": episode,
                "reward": total_reward,
                "label": last_info.get("label"),
                "label_name": last_info.get("label_name"),
                "predicted": last_info.get("predicted"),
                "predicted_name": last_info.get("predicted_name"),
                "correct": bool(last_info.get("correct")),
                "classified": last_info.get("predicted") is not None,
                "best_iou": last_info.get("best_iou", 0.0),
                "final_iou": last_info.get("current_iou", 0.0),
                "steps": last_info.get("steps", 0),
                "window_area": last_info.get("window_area", 0.0),
            }
        )
    return pd.DataFrame(rows)


class ResearchEvalCallback(BaseCallback):
    def __init__(
        self,
        val_df: pd.DataFrame,
        variant: VariantConfig,
        output_dir: Path,
        eval_freq: int,
        episodes: int,
        seed: int,
        image_size: int,
        image_cache: dict[str, np.ndarray] | None,
        cache_side: int,
    ) -> None:
        super().__init__()
        self.val_df = val_df
        self.variant = variant
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.image_size = image_size
        self.image_cache = image_cache
        self.cache_side = cache_side
        self.rows: list[dict[str, Any]] = []
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps
        eval_df = evaluate_policy(
            self.model,
            self.val_df,
            self.variant,
            episodes=self.episodes,
            seed=self.seed + self.n_calls,
            image_size=self.image_size,
            image_cache=self.image_cache,
            cache_side=self.cache_side,
        )
        row = summarize_eval(eval_df)
        row["timesteps"] = self.num_timesteps
        row["variant"] = self.variant.name
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output_dir / "learning_curve.csv", index=False)
        self.model.save(self.output_dir / "model_latest.zip")
        return True


def summarize_eval(eval_df: pd.DataFrame) -> dict[str, float]:
    classified = eval_df[eval_df["classified"]].copy()
    if classified.empty:
        acc = 0.0
        macro_f1 = 0.0
    else:
        acc = accuracy_score(classified["label"], classified["predicted"])
        macro_f1 = f1_score(
            classified["label"],
            classified["predicted"],
            labels=list(range(len(DISEASES))),
            average="macro",
            zero_division=0,
        )
    return {
        "mean_reward": float(eval_df["reward"].mean()),
        "accuracy_classified": float(acc),
        "macro_f1_classified": float(macro_f1),
        "classification_rate": float(eval_df["classified"].mean()),
        "mean_best_iou": float(eval_df["best_iou"].mean()),
        "mean_final_iou": float(eval_df["final_iou"].mean()),
        "mean_steps": float(eval_df["steps"].mean()),
        "mean_window_area": float(eval_df["window_area"].mean()),
    }


def plot_learning_curves(all_curves: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_reward", "Mean reward"),
        ("accuracy_classified", "Accuracy on classified episodes"),
        ("macro_f1_classified", "Macro-F1 entre episodios classificados"),
        ("mean_best_iou", "Mean best IoU"),
    ]
    for metric, ylabel in metrics:
        if metric not in all_curves:
            continue
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=all_curves, x="timesteps", y=metric, hue="variant", marker="o")
        plt.xlabel("Timesteps PPO")
        plt.ylabel(ylabel)
        plt.title(f"Learning curve - {ylabel}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"learning_curve_{metric}.png", dpi=220)
        plt.close()


def plot_final_results(summary: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    for metric, ylabel in [
        ("accuracy_classified", "Accuracy"),
        ("macro_f1_classified", "Macro-F1"),
        ("mean_best_iou", "Mean best IoU"),
        ("classification_rate", "Classification rate"),
    ]:
        plt.figure(figsize=(6.5, 4))
        sns.barplot(data=summary, x="variant", y=metric, color="#59A14F")
        plt.xlabel("PPO variant")
        plt.ylabel(ylabel)
        plt.title(f"Final comparison - {ylabel}")
        plt.ylim(0, max(1.0, float(summary[metric].max()) * 1.15))
        plt.tight_layout()
        plt.savefig(fig_dir / f"final_{metric}.png", dpi=220)
        plt.close()


def plot_confusion(eval_df: pd.DataFrame, output_path: Path, title: str) -> None:
    classified = eval_df[eval_df["classified"]].copy()
    if classified.empty:
        return
    cm = confusion_matrix(
        classified["label"], classified["predicted"], labels=list(range(len(DISEASES)))
    )
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=DISEASES,
        yticklabels=DISEASES,
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "variant",
        "mean_reward",
        "accuracy_classified",
        "macro_f1_classified",
        "classification_rate",
        "mean_best_iou",
        "mean_steps",
    ]
    latex = summary[cols].round(4).to_latex(index=False, escape=True)
    output_path.write_text(latex, encoding="utf-8")


def train_variant(
    variant: VariantConfig,
    metadata: pd.DataFrame,
    output_dir: Path,
    timesteps: int,
    eval_freq: int,
    eval_episodes: int,
    seed: int,
    n_envs: int,
    image_size: int,
    image_cache: dict[str, np.ndarray] | None,
    cache_side: int,
    n_steps: int,
    n_epochs: int,
    test_episodes: int,
) -> dict[str, Any]:
    variant_dir = output_dir / "models" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)

    env = DummyVecEnv(
        [
            make_env(
                train_df,
                variant,
                seed + i * 17,
                image_size,
                image_cache,
                cache_side,
            )
            for i in range(n_envs)
        ]
    )
    callback = ResearchEvalCallback(
        val_df=val_df,
        variant=variant,
        output_dir=variant_dir,
        eval_freq=max(1, eval_freq),
        episodes=eval_episodes,
        seed=seed,
        image_size=image_size,
        image_cache=image_cache,
        cache_side=cache_side,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        learning_rate=variant.learning_rate,
        gamma=variant.gamma,
        n_steps=n_steps,
        batch_size=min(128, max(8, n_steps * n_envs)),
        n_epochs=n_epochs,
        ent_coef=0.02,
        clip_range=0.2,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=False,
        tb_log_name=variant.name,
    )
    model.save(variant_dir / "model_final.zip")
    test_eval = evaluate_policy(
        model,
        test_df,
        variant,
        episodes=len(test_df) if test_episodes <= 0 else min(test_episodes, len(test_df)),
        seed=seed + 9000,
        image_size=image_size,
        image_cache=image_cache,
        cache_side=cache_side,
    )
    test_eval.to_csv(variant_dir / "test_episode_metrics.csv", index=False)
    summary = summarize_eval(test_eval)
    summary["variant"] = variant.name
    summary["reward_mode"] = variant.reward_mode
    summary["timesteps"] = timesteps
    summary["max_steps"] = variant.max_steps
    plot_confusion(
        test_eval,
        output_dir / "figures" / f"confusion_{variant.name}.png",
        f"Confusion matrix - {variant.name}",
    )
    return summary


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"tomato_ppo_{stamp}"
    else:
        output_dir = args.output
    output_dir = output_dir.resolve()
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    metadata = build_metadata(dataset_dir, output_dir, args.seed)
    plot_dataset_summary(metadata, output_dir)
    metadata.groupby(["split", "disease"]).size().reset_index(name="n").to_csv(
        output_dir / "tables" / "dataset_split_counts.csv", index=False
    )
    image_cache = preload_images(metadata, max_side=args.cache_side) if args.preload_images else {}

    summaries = []
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        print(f"\n=== Training PPO variant: {variant.name} ===")
        summaries.append(
            train_variant(
                variant=variant,
                metadata=metadata,
                output_dir=output_dir,
                timesteps=args.timesteps,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                n_envs=args.n_envs,
                image_size=args.image_size,
                image_cache=image_cache,
                cache_side=args.cache_side,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                test_episodes=args.test_episodes,
            )
        )

    summary_df = pd.DataFrame(summaries)
    metric_order = [
        "variant",
        "reward_mode",
        "timesteps",
        "max_steps",
        "mean_reward",
        "accuracy_classified",
        "macro_f1_classified",
        "classification_rate",
        "mean_best_iou",
        "mean_final_iou",
        "mean_steps",
        "mean_window_area",
    ]
    summary_df = summary_df[metric_order]
    summary_df.to_csv(output_dir / "tables" / "ppo_variant_comparison.csv", index=False)
    write_latex_table(summary_df, output_dir / "tables" / "ppo_variant_comparison.tex")

    curves = []
    for variant_name in args.variants:
        curve_path = output_dir / "models" / variant_name / "learning_curve.csv"
        if curve_path.exists():
            curves.append(pd.read_csv(curve_path))
    if curves:
        all_curves = pd.concat(curves, ignore_index=True)
        all_curves.to_csv(output_dir / "tables" / "learning_curves.csv", index=False)
        plot_learning_curves(all_curves, output_dir)
    plot_final_results(summary_df, output_dir)

    best = summary_df.sort_values(
        ["macro_f1_classified", "mean_best_iou", "mean_reward"], ascending=False
    ).iloc[0]
    report = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "n_images": int(len(metadata)),
        "variants": args.variants,
        "best_variant": best.to_dict(),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\n=== Best variant ===")
    print(best.to_string())
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
