# Updated PPO Experiment Report for Tomato Disease Images

## Current Run

The experiment was rerun with an improved active visual inspection pipeline and 50,000
PPO timesteps per variant. The updated pipeline includes an auxiliary visual-evidence
classifier trained only on the training split. Its predicted class probabilities are
added to the PPO state, while PPO remains responsible for learning the active
inspection policy.

Command used:

```powershell
python .\tomato_ppo_experiments.py --dataset ".\Tomato Disease Dataset" --output ".\runs\tomato_ppo_paper_run_improved" --timesteps 50000 --eval-freq 5000 --eval-episodes 100 --variants localization balanced efficient --n-envs 4 --image-size 64 --cache-side 256 --preload-images --n-steps 128 --n-epochs 4 --aux-crops-per-image 3
```

## Main Results

| Method | Accuracy | Macro-F1 | Mean best IoU |
|---|---:|---:|---:|
| majority_class | 0.5130 | 0.2260 | - |
| uniform_random | 0.3442 | 0.3070 | - |
| stratified_random | 0.4675 | 0.3588 | - |
| PPO-localization | 0.9026 | 0.8731 | 0.4148 |
| PPO-balanced | 0.9026 | 0.8731 | 0.4148 |
| PPO-efficient | **0.9286** | **0.9337** | **0.5195** |

The best variant was `PPO-efficient`. It achieved the highest diagnostic performance
and the best spatial alignment with expert annotations while using the smallest
attention-window area.

## PPO Variant Comparison

| Variant | Accuracy | Macro-F1 | Mean best IoU | Steps | Window area | Reward |
|---|---:|---:|---:|---:|---:|---:|
| localization | 0.9026 | 0.8731 | 0.4148 | 8.0 | 0.2044 | 3.2097 |
| balanced | 0.9026 | 0.8731 | 0.4148 | 8.0 | 0.2044 | 3.7034 |
| efficient | **0.9286** | **0.9337** | **0.5195** | **6.0** | **0.1489** | **3.7119** |

## Per-Class Metrics for PPO-efficient

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| GrayMold | 0.9048 | 1.0000 | 0.9500 | 19 |
| Viral | 0.8824 | 0.9836 | 0.9302 | 61 |
| Wilt | 0.9846 | 0.8649 | 0.9209 | 74 |

## Key Artifacts

- Article-ready methodology and results: `artigo_metodologia_resultados_atualizado.tex`
- Final run: `runs/tomato_ppo_paper_run_improved`
- PPO vs baselines: `runs/tomato_ppo_paper_run_improved/tables/ppo_vs_baselines.csv`
- PPO variant comparison: `runs/tomato_ppo_paper_run_improved/tables/ppo_variant_comparison.csv`
- Per-class metrics: `runs/tomato_ppo_paper_run_improved/tables/per_class_metrics_efficient.csv`
- Qualitative attention examples: `runs/tomato_ppo_paper_run_improved/figures/qualitative_attention_examples_efficient.png`
- Learning curves: `runs/tomato_ppo_paper_run_improved/figures/learning_curve_macro_f1_classified.png` and `runs/tomato_ppo_paper_run_improved/figures/learning_curve_mean_best_iou.png`

## Interpretation

The improved experiment changes the conclusion from a pilot validation to a stronger
result. The model now exceeds all simple baselines in Macro-F1, performs well across
all three classes, and shows improved spatial alignment with the expert annotations.
The strongest result is the combination achieved by PPO-efficient: high Macro-F1,
highest mean best IoU, fewer inspection steps, and the smallest mean attention-window
area.
