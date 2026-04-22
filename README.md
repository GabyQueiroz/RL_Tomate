# PPO for Active Visual Attention in Tomato Disease Images

This project implements a Reinforcement Learning experiment with Proximal Policy
Optimization (PPO) for the `Tomato Disease Dataset`. The task is formulated as
budgeted active visual inspection: an agent moves and resizes an attention window over
the image, receives reward from expert XML bounding-box annotations, and must classify
the disease as `GrayMold`, `Viral`, or `Wilt`.

## Quick Smoke Test

Use this to verify that the code runs end to end:

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\smoke_test_en" --timesteps 8 --eval-freq 1000 --eval-episodes 1 --test-episodes 3 --variants localization --n-envs 1 --image-size 32 --cache-side 128 --n-steps 8 --n-epochs 1
```

## Main Experiment

Use this for a complete but moderate comparison:

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\tomato_ppo_budgeted_decision_en" --timesteps 512 --eval-freq 256 --eval-episodes 15 --variants localization balanced efficient --n-envs 4 --image-size 48 --cache-side 192 --n-steps 64 --n-epochs 2
```

For paper-grade results, increase the training budget:

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\tomato_ppo_paper_run" --timesteps 50000 --eval-freq 2500 --eval-episodes 100 --variants localization balanced efficient --n-envs 4 --image-size 64 --cache-side 256 --n-steps 128 --n-epochs 4
```

## Generated Artifacts

Each run creates an output folder under `runs/` containing:

- `metadata.csv`: image paths, splits, classes, and normalized bounding boxes.
- `tables/dataset_split_counts.csv`: train/validation/test distribution.
- `tables/object_counts.csv`: XML label counts.
- `tables/ppo_variant_comparison.csv`: final PPO variant comparison.
- `tables/ppo_variant_comparison.tex`: LaTeX-ready results table.
- `tables/learning_curves.csv`: validation history during training.
- `figures/*.png`: learning curves, final comparisons, dataset statistics, and
  confusion matrices.
- `models/<variant>/model_final.zip`: final PPO model for each variant.
- `models/<variant>/test_episode_metrics.csv`: per-episode test metrics.
- `experiment_summary.json`: run summary and selected best variant.

## PPO Variants

- `localization`: emphasizes IoU improvement with expert bounding boxes.
- `balanced`: combines IoU, classification reward, and a small step penalty.
- `efficient`: penalizes large windows and extra steps to encourage economical visual
  inspection.

## How to Know Whether the Algorithm Is Good

Do not judge the algorithm only by training reward. Use the held-out test metrics in
`tables/ppo_variant_comparison.csv`.

Key indicators:

- `macro_f1_classified`: the most important classification metric because the dataset
  is imbalanced.
- `accuracy_classified`: useful, but can be misleading if the model favors the largest
  class.
- `classification_rate`: should be close to `1.0`; in this protocol every episode
  should produce a diagnosis.
- `mean_best_iou`: shows whether the attention policy finds annotated diseased regions.
- `mean_steps` and `mean_window_area`: show inspection efficiency.
- Confusion matrices: reveal whether one disease class is being ignored.
- Learning curves: should improve on validation data instead of staying flat or
  becoming unstable.

Strong evidence that the model is useful:

- Macro-F1 is clearly above a majority-class baseline.
- Mean best IoU increases during training.
- The best PPO variant remains strong on the test split, not only validation.
- Confusion matrices show non-trivial performance across all three diseases.
- Results are stable across at least three random seeds.

Warning signs:

- High accuracy but low Macro-F1 means the agent is probably biased toward the majority
  class.
- Classification rate below `1.0` means the evaluation protocol is not forcing a
  diagnosis.
- Flat learning curves suggest the reward or observation representation is not strong
  enough.
- High IoU with poor Macro-F1 means the agent localizes disease regions but still
  needs a better diagnostic representation.
