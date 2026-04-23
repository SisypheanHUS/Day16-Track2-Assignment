#!/usr/bin/env python3
"""
LightGBM Benchmark — Credit Card Fraud Detection
Used as CPU fallback when GPU quota is unavailable (Lab 16, Part 7).

Usage:
    python3 benchmark.py

Expects creditcard.csv in the current directory (~/ml-benchmark/).
Outputs metrics to stdout and saves benchmark_result.json.
"""

import json
import time
import platform
import os
import sys

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def main():
    csv_path = "creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found in {os.getcwd()}")
        print("Download it first:  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p .")
        sys.exit(1)

    print("=" * 60)
    print("  LightGBM Benchmark — Credit Card Fraud Detection")
    print("=" * 60)
    print(f"Platform : {platform.platform()}")
    print(f"CPU      : {platform.processor() or 'N/A'}")
    print(f"Python   : {platform.python_version()}")
    print(f"LightGBM : {lgb.__version__}")
    print()

    # --- Load data ---
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    load_time = time.perf_counter() - t0
    print(f"[1/4] Data loaded: {df.shape[0]:,} rows x {df.shape[1]} cols  ({load_time:.3f}s)")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train ---
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True,
        "seed": 42,
    }

    print("[2/4] Training LightGBM (early stopping, max 1000 rounds) ...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
    ]

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - t0
    print(f"       Training done in {train_time:.3f}s  (best iteration: {model.best_iteration})")

    # --- Evaluate ---
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"[3/4] Evaluation:")
    print(f"       AUC-ROC   : {auc:.6f}")
    print(f"       Accuracy  : {acc:.6f}")
    print(f"       F1-Score  : {f1:.6f}")
    print(f"       Precision : {prec:.6f}")
    print(f"       Recall    : {rec:.6f}")

    # --- Inference latency ---
    single_row = X_test.iloc[[0]]
    batch_1k = X_test.iloc[:1000]

    n_warmup = 10
    for _ in range(n_warmup):
        model.predict(single_row, num_iteration=model.best_iteration)

    n_iters = 1000
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model.predict(single_row, num_iteration=model.best_iteration)
    single_latency_ms = (time.perf_counter() - t0) / n_iters * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        model.predict(batch_1k, num_iteration=model.best_iteration)
    batch_throughput_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"[4/4] Inference:")
    print(f"       Single row latency  : {single_latency_ms:.4f} ms")
    print(f"       1000-row batch      : {batch_throughput_ms:.4f} ms")

    # --- Save results ---
    result = {
        "system": {
            "platform": platform.platform(),
            "cpu": platform.processor() or "N/A",
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__,
        },
        "data": {
            "dataset": "Credit Card Fraud Detection (Kaggle)",
            "total_rows": int(df.shape[0]),
            "features": int(df.shape[1] - 1),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "training": {
            "load_time_sec": round(load_time, 4),
            "training_time_sec": round(train_time, 4),
            "best_iteration": model.best_iteration,
            "num_leaves": params["num_leaves"],
            "learning_rate": params["learning_rate"],
        },
        "metrics": {
            "auc_roc": round(auc, 6),
            "accuracy": round(acc, 6),
            "f1_score": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
        },
        "inference": {
            "single_row_latency_ms": round(single_latency_ms, 4),
            "batch_1000_rows_ms": round(batch_throughput_ms, 4),
        },
    }

    out_path = "benchmark_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Results saved to {out_path}")

    # --- Summary table ---
    print()
    print("=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Result':>20}")
    print(f"  {'-'*35} {'-'*20}")
    print(f"  {'Data load time':<35} {load_time:>17.3f} s")
    print(f"  {'Training time':<35} {train_time:>17.3f} s")
    print(f"  {'Best iteration':<35} {model.best_iteration:>20}")
    print(f"  {'AUC-ROC':<35} {auc:>20.6f}")
    print(f"  {'Accuracy':<35} {acc:>20.6f}")
    print(f"  {'F1-Score':<35} {f1:>20.6f}")
    print(f"  {'Precision':<35} {prec:>20.6f}")
    print(f"  {'Recall':<35} {rec:>20.6f}")
    print(f"  {'Inference latency (1 row)':<35} {single_latency_ms:>16.4f} ms")
    print(f"  {'Inference throughput (1000 rows)':<35} {batch_throughput_ms:>16.4f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
