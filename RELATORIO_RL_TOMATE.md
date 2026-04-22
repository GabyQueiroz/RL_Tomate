# PPO Experiment Report for Tomato Disease Images

## Experimental Idea

This experiment formulates tomato disease diagnosis as a sequential active visual
inspection problem. At each episode, a PPO agent receives an image and moves/resizes an
attention window. Expert XML bounding boxes are used as localization reward. At the end
of the inspection budget, the agent must output a diagnosis among `GrayMold`, `Viral`,
and `Wilt`.

This design evaluates four dimensions at once:

- visual localization quality through IoU;
- diagnostic performance through accuracy and Macro-F1;
- inspection efficiency through step count and attention-window area;
- learning dynamics through validation curves.

## Processed Dataset

- Valid image/XML pairs: 1,026.
- Classes: `GrayMold`, `Viral`, `Wilt`.
- Annotated bounding boxes: 3,167.
- Most frequent XML labels: `Viral_Leaf`, `Wilt_Leaf`, `Wilt_Middle`,
  `Wilt_Top`, `Wilt_Base`, `Viral_Top`, `Wilt_Stem`, `GrayMold_Leaf`,
  `GrayMold_Fruit`, `Virus_Middle`.

## PPO Variants

- `localization`: reward emphasizes IoU improvement with expert annotations.
- `balanced`: reward combines IoU, classification, and a mild step penalty.
- `efficient`: reward penalizes large windows and unnecessary steps, encouraging an
  economical inspection policy.

## Current English Run

Command:

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\tomato_ppo_budgeted_decision_en" --timesteps 512 --eval-freq 256 --eval-episodes 15 --variants localization balanced efficient --n-envs 4 --image-size 48 --cache-side 192 --n-steps 64 --n-epochs 2
```

This is a moderate run intended to validate the full pipeline. For final paper results,
repeat with at least 50,000 timesteps per PPO variant and preferably three random
seeds.

## How to Evaluate Model Quality

The main result file is:

`runs/tomato_ppo_budgeted_decision_en/tables/ppo_variant_comparison.csv`

Use the following interpretation:

- Best classification quality: highest `macro_f1_classified`.
- Best localization quality: highest `mean_best_iou`.
- Best inspection efficiency: lower `mean_steps` and lower `mean_window_area`, as long
  as Macro-F1 does not collapse.
- Best overall variant: high Macro-F1, high IoU, stable learning curve, and no class
  collapse in the confusion matrix.

For this imbalanced dataset, Macro-F1 should be prioritized over plain accuracy.
Accuracy can look acceptable even when the model mostly predicts the majority class.

## Baselines to Beat

A useful sanity baseline is the majority-class classifier. Since `Wilt` has the most
images, a naive classifier can obtain relatively high accuracy by always predicting the
majority class. Therefore, a PPO result is only convincing if:

- Macro-F1 is higher than the majority-class baseline;
- confusion matrices show meaningful predictions for `GrayMold`, `Viral`, and `Wilt`;
- IoU improves during training, showing that the agent uses the expert annotations;
- the same conclusion holds across multiple random seeds.

## Suggested Methods Text

We formulated tomato disease diagnosis as a budgeted active visual inspection problem.
Instead of directly classifying the whole image, a PPO agent learns an attention policy
that adjusts an observation window through discrete translation, zoom, and aspect-ratio
actions. Manual Pascal VOC/XML annotations were used to compute IoU-based rewards
between the current attention window and expert-labeled phenotypic regions. At the end
of a fixed inspection budget, the policy outputs a disease decision among gray mold,
viral disease, and bacterial wilt. Three reward designs were compared, prioritizing
localization, balanced localization-classification behavior, and inspection efficiency.
Performance was evaluated using accuracy, Macro-F1, classification rate, mean best IoU,
mean number of steps, mean inspected area, confusion matrices, and learning curves.
