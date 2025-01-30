import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List

import daytradeai.preprocess as preprocess


def hist_pdiff_1d(df: pd.DataFrame, tickers: List[str]) -> None:
    """Plot histogram and summariy statitics of all daily price changes - current day.

    Returns all the daily price changes in single array
    """
    pdiffs_cols = [preprocess.get_feat_name(col=stock, feat="pdiff", anchor=0, lag=1) for stock in tickers]
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


def hist_pdiff_yesterday_vs_today(df: pd.DataFrame, tickers: List[str]) -> Dict[str, np.ndarray]:
    today_cols = [preprocess.get_feat_name(col=stock, feat="pdiff", anchor=0, lag=1) for stock in tickers]
    yesterday_cols = [preprocess.get_feat_name(col=stock, feat="pdiff", anchor=1, lag=1) for stock in tickers]
    df2 = df[today_cols + yesterday_cols].dropna()

    pdiffs = np.concatenate([df2[col].values for col in today_cols])  # type: ignore
    a, b = np.quantile(pdiffs, [0.1, 0.9])

    plt.figure(figsize=(11, 4))
    result = dict()
    for name, thresh in dict(low=a, high=b).items():
        yesterday_pdiffs = []
        for today_col, yesterday_col in zip(today_cols, yesterday_cols):
            if name == 'low':
                idx = df2[today_col] <= thresh
            else:
                idx = df2[today_col] >= thresh
            yesterday_pdiffs.append(df2[yesterday_col][idx].values)
        ypdiffs = np.concatenate(yesterday_pdiffs)
        plt.hist(
            ypdiffs,
            bins=400,
            alpha=0.3,
            label=f"{name} mu={ypdiffs.mean():.3f}\nmedian={np.median(ypdiffs):.3f}\nsigma={ypdiffs.std():.3f}\nn={len(ypdiffs)}",
        )
        result[name] = ypdiffs

    plt.xlabel("Daily Price Change (%)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Yesterdays Daily Price Changes corresponding to Today's Low and High Price Changes")
    plt.legend()
    plt.show()

    return result
