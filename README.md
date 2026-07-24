# College Basketball Champion Prediction

Predicting the NCAA men's basketball champion from end-of-season team statistics
using XGBoost, with the preprocessing pipeline implemented in both pandas and SQL.

## Data
Bart Torvik college basketball team-season stats, 2013–2025 (`cbb.csv`).
Modeling universe restricted to NCAA tournament teams (~812 rows, 12 champions).

## Results
Champion's rank out of 68 teams, per test season (2022–2025):

| Season | XGBoost | BARTHAG baseline |
|---|---|---|
| 2022 | 1 | 3 |
| 2023 | 1 | 1 |
| 2024 | 2 | 2 |
| 2025 | 3 | 4 |

Mean rank: 1.75 (XGBoost) vs 2.50 (single power rating).

## Files
- `test.ipynb` — full analysis notebook
- `cbb.csv` — source data

## Status
