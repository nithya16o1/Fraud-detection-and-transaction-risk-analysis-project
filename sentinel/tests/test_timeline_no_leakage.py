import pytest

import pandas as pd

from sentinel.src.data.clean_data import compute_full_timeline_features

def test_no_temporal_leakage():

    test_df = pd.DataFrame({

        'TransactionID': [101, 102, 103],

        'entity_id': ['card_X', 'card_X', 'card_X'],

        'TransactionDT': [1000, 2000, 3000],

        'TransactionAmt': [100.0, 500.0, 1000.0]

    })

    res = compute_full_timeline_features(test_df)

    # Tx 1: first tx -> ratio 1.0, 24h count 0

    assert res.loc[0, 'amt_vs_rolling_avg'] == 1.0

    assert res.loc[0, 'tx_count_last_24h'] == 0

    # Tx 2: current=500, prior=[100] (avg=100) -> ratio = 5.0

    assert res.loc[1, 'amt_vs_rolling_avg'] == 5.0

    # Tx 3: current=1000, prior=[100, 500] (avg=300) -> ratio = 1000/300 = 3.33

    assert round(res.loc[2, 'amt_vs_rolling_avg'], 2) == 3.33