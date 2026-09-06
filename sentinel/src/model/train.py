import os 
 
import json 
 
import joblib 
 
import pandas as pd 
 
import numpy as np 
 
from sklearn.model_selection import train_test_split 
 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
 
from xgboost import XGBClassifier 
 
from imblearn.over_sampling import SMOTE 
 
FEATURE_COLS = [ 
 
    'TransactionAmt', 'card1', 'card2', 'card3', 'card5',  
 
    'addr1', 'addr2', 'dist1', 'C1', 'C2', 'C5', 'C13' 
 
] 
 
def train_and_evaluate(): 
 
    data_path = os.path.join("sentinel", "data", "processed", "cleaned_transactions.parquet") 
 
    test_out_path = os.path.join("sentinel", "data", "processed", "test_set.parquet") 
 
    if not os.path.exists(data_path): 
 
        raise FileNotFoundError("Cleaned dataset missing. Run Stage 1 clean_data.py first.") 
 
    df = pd.read_parquet(data_path) 
 
    X = df[FEATURE_COLS] 
 
    y = df['isFraud'] 
 
    # Step 1: Stratified Train/Test Split (80/20) BEFORE tuning or resampling 
 
    df_train, df_test = train_test_split( 
 
        df, test_size=0.20, random_state=42, stratify=y 
 
    ) 
 
    # SAVE UNTOUCHED TEST SET TO DISK IMMEDIATELY FOR STAGE 7 
 
    df_test.to_parquet(test_out_path, index=False) 
 
    print(f"[INFO] Untouched test set saved to disk: {test_out_path} ({len(df_test):,} rows)") 
 
    X_train = df_train[FEATURE_COLS] 
 
    y_train = df_train['isFraud'] 
 
    X_test = df_test[FEATURE_COLS] 
 
    y_test = df_test['isFraud'] 
 
    # Step 2: Split Training set into Train Fold and Validation Fold 
 
    X_tr, X_val, y_tr, y_val = train_test_split( 
 
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train 
 
    ) 
 
    print(f"[INFO] Train fold: {len(X_tr):,}, Val fold: {len(X_val):,}, Test set: {len(X_test):,}") 
 
    # Model A: Baseline Class-Weighted XGBoost (No SMOTE) 
 
    scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr) 
 
    model_base = XGBClassifier( 
 
        n_estimators=100, max_depth=6, learning_rate=0.1, 
 
        scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss' 
 
    ) 
 
    model_base.fit(X_tr, y_tr) 
 
    val_probs_base = model_base.predict_proba(X_val)[:, 1] 
 
    val_f1_base = f1_score(y_val, (val_probs_base >= 0.5).astype(int)) 
 
    # Model B: SMOTE on Train Fold Only 
 
    smote = SMOTE(random_state=42) 
 
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr) 
 
    model_smote = XGBClassifier( 
 
        n_estimators=100, max_depth=6, learning_rate=0.1, 
 
        random_state=42, eval_metric='logloss' 
 
    ) 
 
    model_smote.fit(X_tr_sm, y_tr_sm) 
 
    val_probs_smote = model_smote.predict_proba(X_val)[:, 1] 
 
    val_f1_smote = f1_score(y_val, (val_probs_smote >= 0.5).astype(int)) 
 
    print("\n[MODEL SELECTION COMPARISON ON VALIDATION FOLD]") 
 
    print(f"  - Class-Weighted XGBoost Val F1 (@0.50): {val_f1_base:.4f}") 
 
    print(f"  - SMOTE + XGBoost Val F1:         {val_f1_smote:.4f}") 
 
    chosen_val_probs = val_probs_base 
 
    use_smote = val_f1_smote > val_f1_base 
 
    if use_smote: 
 
        print("[DECISION] SMOTE approach achieved higher validation F1.") 
 
        chosen_val_probs = val_probs_smote 
 
    else: 
 
        print("[DECISION] Class-Weighted XGBoost baseline achieved superior validation F1.") 
 
    # Step 3: Select Decision Threshold on VALIDATION PROBABILITIES ONLY 
 
    print("\n=======================================================") 
 
    print("VALIDATION THRESHOLD TRADE-OFF TABLE") 
 
    print("=======================================================") 
 
    print(f"{'Cutoff':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'% Fraud Caught':<15} | {'% False Positives':<18}") 
 
    print("-" * 75) 
 
    best_thresh = 0.70 
 
    best_val_f1 = 0.0 
 
    thresholds = [0.30, 0.50, 0.70, 0.85, 0.90] 
 
    for th in thresholds: 
 
        preds = (chosen_val_probs >= th).astype(int) 
 
        prec = precision_score(y_val, preds, zero_division=0) 
 
        rec = recall_score(y_val, preds, zero_division=0) 
 
        f1 = f1_score(y_val, preds, zero_division=0) 
 
        fraud_caught = rec * 100 
 
        fp_rate = ((preds == 1) & (y_val == 0)).sum() / (y_val == 0).sum() * 100 
 
        print(f"{th:<8.2f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {fraud_caught:<15.1f}% | {fp_rate:<18.2f}%") 
 
        if f1 > best_val_f1: 
 
            best_val_f1 = f1 
 
            best_thresh = th 
 
    print(f"\n[SELECTED THRESHOLD] Chosen Cutoff: {best_thresh:.2f} (Val F1: {best_val_f1:.4f})") 
 
    # Step 4: Retrain Final Model on Full Training Set (FIX #1) 
 
    print("\n[INFO] Retraining final model on full training data using chosen approach...") 
 
    if use_smote: 
 
        X_train_final, y_train_final = SMOTE(random_state=42).fit_resample(X_train, y_train) 
 
        print(f"[FIT CONFIRMATION] Retraining XGBoost on SMOTE resampled training data ({len(X_train_final):,} rows)...") 
 
        final_model = XGBClassifier( 
 
            n_estimators=150, max_depth=6, learning_rate=0.1, 
 
            random_state=42, eval_metric='logloss' 
 
        ) 
 
        final_model.fit(X_train_final, y_train_final) 
 
    else: 
 
        full_scale_pos = (len(y_train) - sum(y_train)) / sum(y_train) 
 
        print(f"[FIT CONFIRMATION] Retraining XGBoost on class-weighted raw training data ({len(X_train):,} rows)...") 
 
        final_model = XGBClassifier( 
 
            n_estimators=150, max_depth=6, learning_rate=0.1, 
 
            scale_pos_weight=full_scale_pos, random_state=42, eval_metric='logloss' 
 
        ) 
 
        final_model.fit(X_train, y_train) 
 
    # Step 5: Final Evaluation on Saved Untouched Test Set 
 
    test_probs = final_model.predict_proba(X_test)[:, 1] 
 
    test_preds = (test_probs >= best_thresh).astype(int) 
 
    final_prec = precision_score(y_test, test_preds) 
 
    final_rec = recall_score(y_test, test_preds) 
 
    final_f1 = f1_score(y_test, test_preds) 
 
    final_auc = roc_auc_score(y_test, test_probs) 
 
    print("\n=======================================================") 
 
    print("FINAL UNTOUCHED TEST SET PERFORMANCE (VERIFIED)") 
 
    print("=======================================================") 
 
    print(f"ROC-AUC:   {final_auc:.4f}") 
 
    print(f"Precision: {final_prec:.4f}") 
 
    print(f"Recall:    {final_rec:.4f}") 
 
    print(f"F1-Score:  {final_f1:.4f}") 
 
    print("=======================================================\n") 
 
    # Save Model Artifacts 
 
    model_dir = os.path.join("sentinel", "src", "model") 
 
    os.makedirs(model_dir, exist_ok=True) 
 
    joblib.dump(final_model, os.path.join(model_dir, "xgb_fraud_model.joblib")) 
 
    meta = { 
 
        "chosen_threshold": float(best_thresh), 
 
        "test_precision": float(final_prec), 
 
        "test_recall": float(final_rec), 
 
        "test_f1": float(final_f1), 
 
        "test_auc": float(final_auc), 
 
        "feature_cols": FEATURE_COLS 
 
    } 
 
    with open(os.path.join(model_dir, "model_meta.json"), "w") as f: 
 
        json.dump(meta, f, indent=2) 
 
    print(f"[SUCCESS] Model artifact saved to {os.path.join(model_dir, 'xgb_fraud_model.joblib')}") 
 
if __name__ == "__main__": 
 
    train_and_evaluate()