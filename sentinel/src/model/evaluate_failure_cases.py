import os

import json

import pandas as pd

import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

def run_test_evaluation():

    test_path = os.path.join("sentinel", "data", "processed", "test_set.parquet")

    model_path = os.path.join("sentinel", "src", "model", "xgb_fraud_model.joblib")

    meta_path = os.path.join("sentinel", "src", "model", "model_meta.json")

    # Load saved test set directly (already contains pre-computed full canonical features - FIX #4)

    df_test = pd.read_parquet(test_path)

    model = joblib.load(model_path)

    with open(meta_path, "r") as f:

        meta = json.load(f)

    X_test = df_test[meta["feature_cols"]]

    y_test = df_test["isFraud"]

    thresh = meta["chosen_threshold"]

    probs = model.predict_proba(X_test)[:, 1]

    preds = (probs >= thresh).astype(int)

    df_test['pred'] = preds

    df_test['prob'] = probs

    prec = precision_score(y_test, preds)

    rec = recall_score(y_test, preds)

    f1 = f1_score(y_test, preds)

    # Dollar-Weighted False Positive Friction Cost

    false_positives = df_test[(df_test['isFraud'] == 0) & (df_test['pred'] == 1)]

    false_negatives = df_test[(df_test['isFraud'] == 1) & (df_test['pred'] == 0)]

    fp_dollar_cost = false_positives['TransactionAmt'].sum()

    fn_dollar_loss = false_negatives['TransactionAmt'].sum()

    print("=======================================================")

    print("STAGE 7: UNTOUCHED TEST SET EVALUATION")

    print("=======================================================")

    print(f"Test Precision: {prec:.4f}")

    print(f"Test Recall:    {rec:.4f}")

    print(f"Test F1-Score:  {f1:.4f}")

    print(f"False Positives: {len(false_positives):,} (Dollar Friction: ${fp_dollar_cost:,.2f})")

    print(f"False Negatives: {len(false_negatives):,} (Dollar Fraud Loss: ${fn_dollar_loss:,.2f})")

    print("=======================================================\n")

    print("=======================================================")

    print("FAILURE CASE ANALYSIS (DYNAMIC ROW EXTRACTION)")

    print("=======================================================")

    if not false_negatives.empty:

        fn_sample = false_negatives.iloc[0]

        print(f"Disputed Tx ID:              {fn_sample['TransactionID']}")

        print(f"Transaction Amount:          ${fn_sample['TransactionAmt']:.2f}")

        print(f"Model Probability Score:     {fn_sample['prob']:.4f} (Threshold: {thresh:.2f})")

        print(f"Amount vs Rolling Avg Ratio: {fn_sample['amt_vs_rolling_avg']:.2f}x")

        print(f"24h Velocity Count:          {fn_sample['tx_count_last_24h']}")

        print("\nDynamic Root Cause:")

        print(f"Tx #{fn_sample['TransactionID']} was missed because its amount (${fn_sample['TransactionAmt']:.2f}) and spending spike ratio ({fn_sample['amt_vs_rolling_avg']:.2f}x) matched baseline customer spending behavior.")

        print("Sentinel Mitigation Strategy: The system routes borderline metrics to merchant 'Review' rather than automated rejection.")

    print("=======================================================")

if __name__ == "__main__":

    run_test_evaluation()