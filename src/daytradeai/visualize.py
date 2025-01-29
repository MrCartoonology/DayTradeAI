import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

import daytradeai.preprocess as preprocess


def hist_pdiff_1d(df: pd.DataFrame, tickers: List[str]) -> None:
    """Plot histogram and summariy statitics of all daily price changes - current day.

    Returns all the daily price changes in single array
    """
    pdiffs_cols = preprocess.get_feature_columns(df, suffix="_pdiff_1d", tickers=tickers)
    pdiffs = np.concatenate([df[col].dropna().values for col in pdiffs_cols])  # type: ignore
    plt.hist(
        pdiffs,
        bins=400,
        label=f"mu={pdiffs.mean():.3f}\nmedian={np.median(pdiffs):.3f}\nsigma={pdiffs.std():.3f}\nn={len(pdiffs)}",
    )
    plt.xlabel("Daily Price Change (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Price Changes")
    plt.legend()
    return pdiffs
