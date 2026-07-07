# Candidate Paper Experiment Results

This file records experiment numbers that may be used in the paper later.
Current entries only cover `water_plant`; append new tasks or methods as they are tested.

## Closed-Loop Success Rate

| Task | Method / Setting | Success Rate | Successes / Trials | Reported Variance | Status | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| water_plant | pi0.5 | 88.7% | TBD | ±3.1 | tested | Need confirm eval protocol / seeds. |
| water_plant | gr00t n1.5 | 72.7% | TBD | ±1.2 | tested | Need confirm eval protocol / seeds. |
| water_plant | fastwam | 75.5% | 151 / 200 | TBD | tested | Raw count from current run. |
| water_plant | failure scratch | 82.0% | 82 / 100 | TBD | tested | Raw count from current run. |
| water_plant | lora continuation | 81.5% | 163 / 200 | TBD | tested | LoRA continued training. |

## LaTeX Draft

```latex
\begin{table}[t]
\caption{Closed-loop success rate on the water_plant task.}
\label{tab:water_plant_success}
\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Success Rate} & \textbf{Successes / Trials} \\
\midrule
pi0.5             & $88.7 \pm 3.1$ & -- \\
gr00t n1.5        & $72.7 \pm 1.2$ & -- \\
fastwam           & $75.5\%$       & 151 / 200 \\
failure scratch   & $82.0\%$       & 82 / 100 \\
lora continuation & $81.5\%$       & 163 / 200 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

## Items To Confirm Before Paper Use

- Evaluation protocol: simulator / real robot, number of seeds, and whether all methods share the same test split.
- For `pi0.5` and `gr00t n1.5`, add raw successes / trials if available.
- Standardize method names before moving the table into `iclr2026_conference.tex`.
