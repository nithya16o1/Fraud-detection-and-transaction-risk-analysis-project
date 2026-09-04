import os

import pandas as pd
import numpy as np


SELECTED_COLUMNS = [
    'TransactionID',
    'isFraud',
    'TransactionDT',
    'TransactionAmt',
    'card1',
    'card2',
    'card3',
    'card4',
    'card5',
    'card6',
    'addr1',
    'addr2',
    'dist1',
    'C1',
    'C2',
    'C5',
    'C13',
    'P_emaildomain',
    'R_emaildomain'
]


def analyze_customer_identifiers(df: pd.DataFrame) -> str:
    """
    Computes unique-value counts and 3+ transaction entity counts
    for candidate identifiers and dynamically selects the identifier
    with superior multi-transaction coverage.
    """

    print("\n=======================================================")
    print("STAGE 1: ENTITY IDENTIFIER CARDINALITY & PROXY ANALYSIS")
    print("=======================================================")

    # Candidate A: card1 alone
    card1_series = df['card1'].fillna(0).astype(int).astype(str)

    card1_unique = card1_series.nunique()
    card1_counts = card1_series.value_counts()
    card1_3plus = (card1_counts >= 3).sum()

    # Candidate B: card1 + addr1
    addr1_series = df['addr1'].fillna(0).astype(int).astype(str)

    comb_series = card1_series + "_" + addr1_series

    comb_unique = comb_series.nunique()
    comb_counts = comb_series.value_counts()
    comb_3plus = (comb_counts >= 3).sum()

    print("Candidate A [card1 alone]:")
    print(f"  - Unique Entities: {card1_unique:,}")
    print(
        f"  - Entities with >= 3 Transactions: "
        f"{card1_3plus:,} "
        f"({card1_3plus / card1_unique * 100:.1f}%)"
    )

    print("\nCandidate B [card1 + addr1]:")
    print(f"  - Unique Entities: {comb_unique:,}")
    print(
        f"  - Entities with >= 3 Transactions: "
        f"{comb_3plus:,} "
        f"({comb_3plus / comb_unique * 100:.1f}%)"
    )

    # Dynamic choice
    if comb_3plus >= card1_3plus:
        chosen_proxy = "card1_addr1"

        print(
            f"\n[DECISION] 'card1 + addr1' dynamically selected "
            f"based on higher 3+ transaction coverage "
            f"({comb_3plus:,} vs {card1_3plus:,})."
        )
    else:
        chosen_proxy = "card1_only"

        print(
            f"\n[DECISION] 'card1 alone' dynamically selected "
            f"based on higher 3+ transaction coverage "
            f"({card1_3plus:,} vs {comb_3plus:,})."
        )

    print("=======================================================\n")

    return chosen_proxy


def compute_full_timeline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes canonical timeline features across the full dataset.
    """

    print("[INFO] Pre-computing canonical timeline features on full dataset...")

    df = df.copy()

    df = df.sort_values(
        by=['entity_id', 'TransactionDT', 'TransactionID']
    ).reset_index(drop=True)

    # 1. Time since last transaction
    df['prev_TransactionDT'] = (
        df.groupby('entity_id')['TransactionDT'].shift(1)
    )

    df['time_since_last_tx'] = (
        df['TransactionDT'] -
        df['prev_TransactionDT']
    ).fillna(-1.0)

    # 2. Amount vs rolling average of prior transactions
    df['prior_sum'] = (
        df.groupby('entity_id')['TransactionAmt']
        .shift(1)
        .groupby(df['entity_id'])
        .cumsum()
    )

    df['prior_count'] = (
        df.groupby('entity_id').cumcount()
    )

    df['amt_vs_rolling_avg'] = np.where(
        df['prior_count'] > 0,
        df['TransactionAmt'] /
        (df['prior_sum'] / df['prior_count']),
        1.0
    )

    # 3. Transaction count during previous 24 hours
    counts_24h = np.zeros(len(df), dtype=int)

    group_indices = (
        df.groupby('entity_id', sort=False).indices
    )

    dts = df['TransactionDT'].values

    for entity, idxs in group_indices.items():

        g_dts = dts[idxs]

        window_starts = np.searchsorted(
            g_dts,
            g_dts - 86400,
            side='left'
        )

        counts_24h[idxs] = (
            np.arange(len(idxs)) - window_starts
        )

    df['tx_count_last_24h'] = counts_24h

    # Remove temporary columns
    df = df.drop(
        columns=[
            'prev_TransactionDT',
            'prior_sum',
            'prior_count'
        ]
    )

    return df


def load_and_clean_data(
    raw_csv_path: str,
    output_parquet_path: str
):

    print(
        f"[INFO] Reading raw transaction table: "
        f"{raw_csv_path}"
    )

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            f"Missing {raw_csv_path}. "
            f"Download from Kaggle IEEE-CIS competition."
        )

    df = pd.read_csv(
        raw_csv_path,
        usecols=lambda c: c in SELECTED_COLUMNS
    )

    print(f"[INFO] Loaded shape: {df.shape}")

    # Analyze and select customer entity proxy
    chosen_proxy = analyze_customer_identifiers(df)

    card1_clean = (
        df['card1']
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    addr1_clean = (
        df['addr1']
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    if chosen_proxy == "card1_addr1":
        df['entity_id'] = (
            card1_clean + "_" + addr1_clean
        )
    else:
        df['entity_id'] = card1_clean

    # Sort by transaction time
    df = df.sort_values(
        by=['TransactionDT', 'TransactionID']
    ).reset_index(drop=True)

    # Missing value imputation
    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for c in num_cols:
        df[c] = df[c].fillna(
            df[c].median()
        )

    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')

    df['isFraud'] = df['isFraud'].astype(int)

    # Compute timeline features
    df = compute_full_timeline_features(df)

    # Create output directory
    os.makedirs(
        os.path.dirname(output_parquet_path),
        exist_ok=True
    )

    # Save processed dataset
    df.to_parquet(
        output_parquet_path,
        index=False
    )

    print(
        f"[SUCCESS] Saved clean dataset with "
        f"canonical timeline features to "
        f"{output_parquet_path}"
    )


if __name__ == "__main__":

    raw_path = os.path.join(
        "sentinel",
        "data",
        "raw",
        "train_transaction.csv"
    )

    out_path = os.path.join(
        "sentinel",
        "data",
        "processed",
        "cleaned_transactions.parquet"
    )

    load_and_clean_data(
        raw_path,
        out_path
    )