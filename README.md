# PL Predictor

Predicts Premier League match outcomes (home win, draw, away win) with five machine learning models, served through a FastAPI backend and a simple web page.

## What it does

Pick a home team and an away team. The site shows the win / draw / lose probabilities from each model, their average, and the expected goals for both sides.

## Data

Ten Premier League seasons, 2016/17 to 2025/26 (3,800 matches), from the open [football-datasets](https://github.com/datasets/football-datasets) repository.

## Features

Every feature is calculated only from matches played before the one being predicted, so there is no data leakage.

- **Elo rating** for each team, updated after every match (pre-match value used)
- **Form over the previous 5 matches**: points, goals for and against, shots on target for and against

## Models and results

Trained on 2016/17 to 2023/24, tested on 2024/25 and 2025/26 (759 matches the models never saw).

| Model | Accuracy | Log loss |
|---|---|---|
| Poisson goals | 50.7% | 1.007 |
| Logistic regression | 49.8% | 1.011 |
| Elo only | 50.9% | 1.012 |
| XGBoost | 50.5% | 1.020 |
| Random forest | 50.9% | 1.022 |
| Always home win (baseline) | 41.8% | 1.084 |

Tree model numbers can vary slightly between machines.

What I learned from the results:

- All models beat the baseline by about 9 points of accuracy.
- The simpler models give the best probabilities. Football results are noisy and the dataset is small, so XGBoost and random forest do not beat logistic regression or Poisson.
- Elo alone is nearly as good as Elo plus form, so better features matter more than fancier models.
- No model ever picks a draw as the most likely outcome, which is why log loss is reported alongside accuracy.

## Run it yourself

```
pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload
```

Then open `index.html` in a browser.

## Files

- `train.py` builds the features, evaluates the models and saves `models.joblib`
- `main.py` is the FastAPI backend (`/teams`, `/predict`, `/metrics`)
- `index.html` is the web page

## Possible improvements

- Rest days, head-to-head record, separate home and away form
- Comparison against bookmaker odds
- Season simulator for title and relegation probabilities

This is a learning project, not betting advice.
