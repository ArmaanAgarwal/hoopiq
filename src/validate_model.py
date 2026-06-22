import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

def load_split():
    df = pd.read_csv("data/games_model_ready.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    feature_cols = [c for c in df.columns if (c.startswith("HOME_") or c.startswith("AWAY_"))
                    and c not in ["HOME_TEAM", "AWAY_TEAM", "HOME_WIN"]]

    train = df[df["GAME_DATE"] < "2022-10-01"]
    test = df[df["GAME_DATE"] >= "2022-10-01"]

    return (train[feature_cols], train["HOME_WIN"],
            test[feature_cols], test["HOME_WIN"], feature_cols)

def make_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols = load_split()

    # === TEST 1: REAL LABELS ===
    print("=== TEST 1: Real vs Shuffled Labels ===")
    model = make_model()
    model.fit(X_train, y_train)
    real_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model on REAL labels:     {real_acc:.4f}")

    # Shuffle the training labels - destroy the real pattern
    np.random.seed(42)
    y_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
    model_fake = make_model()
    model_fake.fit(X_train.reset_index(drop=True), y_shuffled)
    fake_acc = accuracy_score(y_test, model_fake.predict(X_test))
    print(f"Model on SHUFFLED labels: {fake_acc:.4f}")
    print(f"--> If real >> shuffled, the model learned genuine signal.\n")

    # === TEST 2: Simple rules ===
    print("=== TEST 2: vs Simple Rules ===")
    df = pd.read_csv("data/games_model_ready.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    test = df[df["GAME_DATE"] >= "2022-10-01"]
    always_home = accuracy_score(test["HOME_WIN"], np.ones(len(test)))
    better_l10 = accuracy_score(test["HOME_WIN"], (test["HOME_L10_MARGIN"] > test["AWAY_L10_MARGIN"]).astype(int))
    print(f"Always pick home:          {always_home:.4f}")
    print(f"Pick better L10 margin:    {better_l10:.4f}")
    print(f"Your model:                {real_acc:.4f}\n")

    # === TEST 3: Calibration ===
    print("=== TEST 3: Calibration ===")
    probs = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    print("Predicted % | Actual %")
    for p, a in zip(mean_pred, frac_pos):
        print(f"   {p*100:5.1f}%   |  {a*100:5.1f}%")
    print("--> If these two columns are close, the probabilities are honest.")