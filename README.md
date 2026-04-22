# PPO for Active Visual Attention in Tomato Disease Images

This project implements a Reinforcement Learning experiment with Proximal Policy
Optimization (PPO) for the `Tomato Disease Dataset`. The task is formulated as
budgeted active visual inspection: an agent moves and resizes an attention window over
the image, receives reward from expert XML bounding-box annotations, and classifies the
disease as `GrayMold`, `Viral`, or `Wilt`.

The version adds an auxiliary visual-evidence classifier trained only
on the training split. Its class probabilities are appended to the PPO state, while PPO
still learns the active inspection policy and the attention trajectory.

## Paper Run

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\tomato_ppo_paper_run_improved" --timesteps 50000 --eval-freq 5000 --eval-episodes 100 --variants localization balanced efficient --n-envs 4 --image-size 64 --cache-side 256 --preload-images --n-steps 128 --n-epochs 4 --aux-crops-per-image 3
```

Best result from the latest run:

| Variant | Accuracy | Macro-F1 | Mean best IoU | Steps | Window area |
|---|---:|---:|---:|---:|---:|
| PPO-efficient | 0.9286 | 0.9337 | 0.5195 | 6.0 | 0.1489 |

## Quick Smoke Test

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\smoke_improved" --timesteps 32 --eval-freq 1000 --eval-episodes 3 --test-episodes 6 --variants efficient --n-envs 1 --image-size 48 --cache-side 192 --n-steps 16 --n-epochs 1 --aux-crops-per-image 2
```

## Generated Artifacts

Each run creates an output folder under `runs/` containing:

- `metadata.csv`: image paths, splits, classes, and normalized bounding boxes.
- `tables/dataset_split_counts.csv`: train/validation/test distribution.
- `tables/object_counts.csv`: XML label counts.
- `tables/auxiliary_classifier_metrics.csv`: auxiliary classifier performance.
- `tables/classification_baselines.csv`: majority-class and random baselines.
- `tables/ppo_vs_baselines.csv`: direct comparison between PPO variants and baselines.
- `tables/ppo_variant_comparison.csv`: final PPO variant comparison.
- `tables/per_class_metrics_<variant>.csv`: per-class precision, recall, and F1-score.
- `figures/*.png`: learning curves, final comparisons, dataset statistics, confusion matrices, and qualitative attention examples.
- `models/<variant>/model_final.zip`: final PPO model for each variant.
- `models/<variant>/test_episode_metrics.csv`: per-episode test metrics.
- `experiment_summary.json`: run summary and selected best variant.

## PPO Variants

- `localization`: emphasizes IoU improvement with expert bounding boxes.
- `balanced`: combines IoU, classification reward, and a small step penalty.
- `efficient`: penalizes large windows and extra steps to encourage economical visual inspection.

