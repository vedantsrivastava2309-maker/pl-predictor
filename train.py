"""
train.py - builds the features, tests the models, then saves everything the website needs.

Run it once with:   python train.py
It creates:         models.joblib
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TEST_SEASONS = ["2425", "2526"]
URL = "https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/premier-league/season-{}.csv"

K, HOME_ADV, START, PROMOTED = 20, 60, 1500, 1400
STATS = ["Points", "GF", "GA", "SOT", "SOTA"]
FEATURES = [
    "EloDiff",
    "HomeFormPoints", "HomeFormGF", "HomeFormGA", "HomeFormSOT", "HomeFormSOTA",
    "AwayFormPoints", "AwayFormGF", "AwayFormGA", "AwayFormSOT", "AwayFormSOTA",
]
GOAL_FEATURES = ["IsHome", "TeamElo", "OppElo", "TeamGF", "TeamSOT", "OppGA", "OppSOTA"]
LABELS = {"A": 0, "D": 1, "H": 2}


# ---------- 1. LOAD DATA ----------
def load_data():
    frames = []
    for season in SEASONS:
        temp = pd.read_csv(URL.format(season))
        temp["Season"] = season
        frames.append(temp)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ---------- 2. ELO RATINGS ----------
def add_elo(df):
    elo = {}
    first_season = df["Season"].iloc[0]
    home_elo, away_elo = [], []
    for row in df.itertuples():
        for team in (row.HomeTeam, row.AwayTeam):
            if team not in elo:
                elo[team] = START if row.Season == first_season else PROMOTED
        h, a = elo[row.HomeTeam], elo[row.AwayTeam]
        home_elo.append(h)   # ratings BEFORE the match
        away_elo.append(a)
        expected_home = 1 / (1 + 10 ** ((a - (h + HOME_ADV)) / 400))
        actual_home = {"H": 1, "D": 0.5, "A": 0}[row.FTR]
        change = K * (actual_home - expected_home)
        elo[row.HomeTeam] = h + change
        elo[row.AwayTeam] = a - change
    df["HomeElo"], df["AwayElo"] = home_elo, away_elo
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    return df, elo   # elo = every team's rating after the last match in the data


# ---------- 3. FORM (previous 5 matches) ----------
def add_form(df):
    df["MatchID"] = df.index
    home = pd.DataFrame({
        "MatchID": df["MatchID"], "Date": df["Date"], "Team": df["HomeTeam"],
        "Points": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        "GF": df["FTHG"], "GA": df["FTAG"], "SOT": df["HST"], "SOTA": df["AST"],
    })
    away = pd.DataFrame({
        "MatchID": df["MatchID"], "Date": df["Date"], "Team": df["AwayTeam"],
        "Points": df["FTR"].map({"H": 0, "D": 1, "A": 3}),
        "GF": df["FTAG"], "GA": df["FTHG"], "SOT": df["AST"], "SOTA": df["HST"],
    })
    team_games = pd.concat([home, away]).sort_values(["Date", "MatchID"]).reset_index(drop=True)
    for s in STATS:
        team_games["Form" + s] = (
            team_games.groupby("Team")[s]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    form_cols = ["Form" + s for s in STATS]
    home_form = team_games.merge(df[["MatchID", "HomeTeam"]], left_on=["MatchID", "Team"], right_on=["MatchID", "HomeTeam"])
    away_form = team_games.merge(df[["MatchID", "AwayTeam"]], left_on=["MatchID", "Team"], right_on=["MatchID", "AwayTeam"])
    df = df.merge(home_form[["MatchID"] + form_cols].rename(columns={c: "Home" + c for c in form_cols}), on="MatchID")
    df = df.merge(away_form[["MatchID"] + form_cols].rename(columns={c: "Away" + c for c in form_cols}), on="MatchID")
    return df, team_games


# ---------- 4. MODELS ----------
def goal_rows(d):
    """One row per team per match, for the Poisson goals model."""
    home = pd.DataFrame({
        "IsHome": 1, "TeamElo": d["HomeElo"], "OppElo": d["AwayElo"],
        "TeamGF": d["HomeFormGF"], "TeamSOT": d["HomeFormSOT"],
        "OppGA": d["AwayFormGA"], "OppSOTA": d["AwayFormSOTA"], "Goals": d["FTHG"],
    })
    away = pd.DataFrame({
        "IsHome": 0, "TeamElo": d["AwayElo"], "OppElo": d["HomeElo"],
        "TeamGF": d["AwayFormGF"], "TeamSOT": d["AwayFormSOT"],
        "OppGA": d["HomeFormGA"], "OppSOTA": d["HomeFormSOTA"], "Goals": d["FTAG"],
    })
    return home, away


def fit_models(d):
    """Train all five models on the matches in d. Every model outputs probabilities in the order A, D, H."""
    y = d["FTR"].map(LABELS)
    models = {}
    models["Elo only"] = LogisticRegression(max_iter=1000).fit(d[["EloDiff"]], y)
    models["Logistic regression"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[FEATURES], y)
    models["Random forest"] = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=42).fit(d[FEATURES], y)
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42).fit(d[FEATURES], y)
    home, away = goal_rows(d)
    goals = pd.concat([home, away])
    models["Poisson goals"] = make_pipeline(
        StandardScaler(), PoissonRegressor(alpha=0.001, max_iter=1000)).fit(goals[GOAL_FEATURES], goals["Goals"])
    return models


def poisson_probs(lam_home, lam_away, max_goals=10):
    """Expected goals for each side -> [P(away win), P(draw), P(home win)]."""
    from scipy.stats import poisson
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    p_home, p_draw, p_away = np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()
    total = p_home + p_draw + p_away
    return [p_away / total, p_draw / total, p_home / total]


def predict_all(models, d):
    """Probabilities (A, D, H) from every model for the matches in d."""
    out = {}
    out["Elo only"] = models["Elo only"].predict_proba(d[["EloDiff"]])
    for name in ["Logistic regression", "Random forest", "XGBoost"]:
        out[name] = models[name].predict_proba(d[FEATURES])
    home, away = goal_rows(d)
    home_xg = models["Poisson goals"].predict(home[GOAL_FEATURES])
    away_xg = models["Poisson goals"].predict(away[GOAL_FEATURES])
    out["Poisson goals"] = np.array([poisson_probs(h, a) for h, a in zip(home_xg, away_xg)])
    return out


# ---------- 5. RUN EVERYTHING ----------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    df, elo = add_elo(df)
    df, team_games = add_form(df)
    data = df.dropna(subset=FEATURES)
    print("Matches:", len(data))

    # test: train on the older seasons, score on the last two
    train = data[~data["Season"].isin(TEST_SEASONS)]
    test = data[data["Season"].isin(TEST_SEASONS)]
    y_test = test["FTR"].map(LABELS)
    probs = predict_all(fit_models(train), test)

    freq = train["FTR"].map(LABELS).value_counts(normalize=True).sort_index().values
    metrics = [{
        "model": "Always home win",
        "accuracy": round(float((y_test == 2).mean()), 3),
        "log_loss": round(float(log_loss(y_test, [freq] * len(y_test), labels=[0, 1, 2])), 3),
    }]
    for name, p in probs.items():
        metrics.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, p.argmax(axis=1))), 3),
            "log_loss": round(float(log_loss(y_test, p, labels=[0, 1, 2])), 3),
        })
    print(pd.DataFrame(metrics).sort_values("log_loss").to_string(index=False))

    # final models: retrain on ALL ten seasons, so the website uses every match available
    final_models = fit_models(data)

    # each team's current strength: latest Elo + average of their last 5 matches
    latest_form = team_games.groupby("Team")[STATS].apply(lambda g: g.tail(5).mean())
    latest_form.columns = ["Form" + s for s in STATS]
    team_state = latest_form.copy()
    team_state["Elo"] = pd.Series(elo)
    teams = sorted(set(df[df["Season"] == SEASONS[-1]]["HomeTeam"]))   # the 20 teams of the latest season
    team_state = team_state.loc[teams]

    joblib.dump({
        "models": final_models,
        "team_state": team_state,
        "teams": teams,
        "metrics": metrics,
    }, "models.joblib")
    print("\nSaved models.joblib with", len(teams), "teams")
