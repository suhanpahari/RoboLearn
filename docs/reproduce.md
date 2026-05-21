# Reproducing the reports

Each figure in each report maps to exactly one command here. Runs are multi-seed and
aggregated with `rliable`. Fill this table in as experiments land.

| Report | Figure | Command |
|--------|--------|---------|
| 01 | Fetch SAC+HER learning curves | `make sweep SWEEP=her_relabel_strategy` then run the grid, then `make figures` |
| 01 | _tbd_ | _tbd_ |
| 02 | Encoder comparison on PointNav | `make sweep SWEEP=encoder_comparison` then run the grid, then `make figures` |

> Every row must be runnable from a clean clone after `make setup-gym` /
> `make setup-hab`. If it is not reproducible, it does not go in a report.
