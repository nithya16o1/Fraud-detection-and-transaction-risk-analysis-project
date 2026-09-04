import pandas as pd

import numpy as np

def get_card_timeline(df: pd.DataFrame, target_tx_id: int, max_history: int = 10):

    """
    Returns ordered transaction history for the target transaction's entity up to target_tx_id,

    using pre-computed canonical full-dataset timeline features (FIX #4).

    """

    target_row = df[df['TransactionID'] == target_tx_id]

    if target_row.empty:

        raise ValueError(f"Transaction ID {target_tx_id} not found.")

    entity = target_row['entity_id'].values[0]

    target_dt = target_row['TransactionDT'].values[0]

    history = df[(df['entity_id'] == entity) & (df['TransactionDT'] <= target_dt)].copy()

    history = history.sort_values(by='TransactionDT').tail(max_history)

    stats = {

        "transaction_id": int(target_tx_id),

        "amount": float(target_row['TransactionAmt'].values[0]),

        "tx_count_last_24h": int(target_row['tx_count_last_24h'].values[0]),

        "amt_vs_rolling_avg": round(float(target_row['amt_vs_rolling_avg'].values[0]), 2),

        "time_since_last_tx": float(target_row['time_since_last_tx'].values[0])

    }

    return history, stats