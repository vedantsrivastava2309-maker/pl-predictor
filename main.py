"""
main.py - the backend. Loads the saved models and answers prediction requests.

Start it with:   python -m uvicorn main:app --reload
Then open:       http://127.0.0.1:8000
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from train import FEATURES, GOAL_FEATURES, poisson_probs

bundle = joblib.load("models.joblib")
models = bundle["models"]
team_state = bundle["team_state"]   # each team's latest Elo and last-5 form
teams = bundle["teams"]

app = FastAPI(title="PL Predictor")

# lets the website (which runs on a different address) call this backend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def as_dict(p):
    """Probabilities arrive in the order A, D, H."""
    return {"home": round(float(p[2]), 3), "draw": round(float(p[1]), 3), "away": round(float(p[0]), 3)}


@app.get("/")
def home_page():
    """The website itself."""
    return FileResponse("index.html")


@app.get("/teams")
def get_teams():
    return teams


@app.get("/metrics")
def get_metrics():
    return bundle["metrics"]


@app.get("/predict")
def predict(home: str, away: str):
    if home not in teams or away not in teams:
        raise HTTPException(status_code=404, detail="Unknown team. See /teams for valid names.")
    if home == away:
        raise HTTPException(status_code=400, detail="Pick two different teams.")

    h, a = team_state.loc[home], team_state.loc[away]

    # the same features the models were trained on, built from each team's current numbers
    row = {"EloDiff": h["Elo"] - a["Elo"]}
    for s in ["Points", "GF", "GA", "SOT", "SOTA"]:
        row["HomeForm" + s] = h["Form" + s]
        row["AwayForm" + s] = a["Form" + s]
    X = pd.DataFrame([row])

    predictions = {"Elo only": as_dict(models["Elo only"].predict_proba(X[["EloDiff"]])[0])}
    for name in ["Logistic regression", "Random forest", "XGBoost"]:
        predictions[name] = as_dict(models[name].predict_proba(X[FEATURES])[0])

    # Poisson: expected goals for each side, then probabilities from the scoreline grid
    home_row = pd.DataFrame([{"IsHome": 1, "TeamElo": h["Elo"], "OppElo": a["Elo"],
                              "TeamGF": h["FormGF"], "TeamSOT": h["FormSOT"],
                              "OppGA": a["FormGA"], "OppSOTA": a["FormSOTA"]}])
    away_row = pd.DataFrame([{"IsHome": 0, "TeamElo": a["Elo"], "OppElo": h["Elo"],
                              "TeamGF": a["FormGF"], "TeamSOT": a["FormSOT"],
                              "OppGA": h["FormGA"], "OppSOTA": h["FormSOTA"]}])
    home_xg = float(models["Poisson goals"].predict(home_row[GOAL_FEATURES])[0])
    away_xg = float(models["Poisson goals"].predict(away_row[GOAL_FEATURES])[0])
    predictions["Poisson goals"] = as_dict(poisson_probs(home_xg, away_xg))

    return {
        "home": home,
        "away": away,
        "expected_goals": {"home": round(home_xg, 2), "away": round(away_xg, 2)},
        "predictions": predictions,
    }
