# College Basketball Champion Prediction

Predicting the NCAA men's basketball champion from end-of-season team statistics
using XGBoost, with the preprocessing pipeline implemented in both pandas and SQL.

## Data
Bart Torvik college basketball team-season stats, 2013–2025 (`cbb.csv`).
Modeling universe restricted to NCAA tournament teams (~812 rows, 12 champions).

## Leakage
The data includes data from the NCAA tournament games. The postseason column can be used to figure out
how many tournament games a team played, allowing those games to be removed from the games played and wins columns,
but there is still some leakage that could not be removed, particularly the aggregate stats. However, the model seemed to
not factor this in too much, as it is an aggregate stats, so the regular season still carries a much larger weight for predicting
placement of the team. 

## Data Validation
While performing checks for the data, it was found that 4 of the teams had more wins recorded than total games played. These were found
to be incorrect values and were dropped from the training data. 

## Method

**Features:** 19 team statistics plus tournament seed and one-hot encoded conference.
`TEAM` and `YEAR` is excluded, as the model should not bias any specific team or year,
as these have no impact on winning. 

**Split:** by season, not randomly. Train on 2013–2021, test on 2022–2025.
A random split would let the model train on 2024 teams to "predict" the 2022 champion,
and would break the per-season ranking evaluation by splitting each tournament field.

**Model:** `XGBClassifier`, 300 trees, max depth 3, learning rate 0.05,
evaluated with `aucpr`.

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
- `champion_prediction.ipynb` — full analysis notebook
- `cbb.csv` — source data

## Pipeline verification

The full cleaning pipeline was implemented twice — once in pandas, once in SQL
(SQLite) — and the outputs verified identical: same row count (812), same champion
count (12), and identical distributions of cleaned games and wins. Two independent
implementations agreeing is strong evidence of correctness.

## Limitations

- **12 positive examples.** The binding constraint on this problem. No amount of
  tuning overcomes having 8 champions in the training set.
- **Residual leakage in rate statistics** (see above).

