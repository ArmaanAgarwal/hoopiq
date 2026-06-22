import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib

def load_data():
    df = pd.read_csv("data/games_model_ready.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df

def split_data(df, split_date="2022-10-01"):
    feature_cols = [c for c in df.columns if c.startswith("HOME_") or c.startswith("AWAY_")]
    feature_cols = [c for c in feature_cols if c not in ["HOME_TEAM", "AWAY_TEAM", "HOME_WIN"]]
    
    train = df[df["GAME_DATE"] < split_date]
    test = df[df["GAME_DATE"] >= split_date]
    
    X_train = train[feature_cols]
    y_train = train["HOME_WIN"]
    X_test = test[feature_cols]
    y_test = test["HOME_WIN"]
    
    print(f"Training games: {len(X_train)} (before {split_date})")
    print(f"Testing games: {len(X_test)} (after {split_date})")
    print(f"Features used: {len(feature_cols)}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    
    # Baseline: always pick home team
    home_baseline = accuracy_score(y_test, np.ones(len(y_test)))
    
    print("\n=== Model Performance ===")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Log Loss:       {ll:.4f}")
    print(f"AUC:            {auc:.4f}")
    print(f"Home baseline:  {home_baseline:.4f}")
    print(f"Improvement:    {(acc - home_baseline)*100:+.2f} percentage points")
    
    return model

if __name__ == "__main__":
    df = load_data()
    X_train, y_train, X_test, y_test, feature_cols = split_data(df)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n=== Top 10 Most Important Features ===")
    print(importance.head(10).to_string(index=False))
    
    joblib.dump(model, "models/baseline_model.pkl")
    print("\nModel saved to models/baseline_model.pkl")

