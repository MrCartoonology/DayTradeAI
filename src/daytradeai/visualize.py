import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

import daytradeai.preprocess as preprocess
import daytradeai.policies as policies
import daytradeai.evaluate as evaluate


def hist_pdiff_1d(df: pd.DataFrame, tickers: List[str]) -> None:
    """Plot histogram and summariy statitics of all daily price changes - current day.

    Returns all the daily price changes in single array
    """
    pdiffs_cols = [
        preprocess.get_feat_name(col=stock, feat="pdiff", anchor=0, lag=1)
        for stock in tickers
    ]
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


def hist_pdiff_yesterday_vs_today(
    df: pd.DataFrame, tickers: List[str]
) -> Dict[str, np.ndarray]:
    today_cols = [
        preprocess.get_feat_name(col=stock, feat="pdiff", anchor=0, lag=1)
        for stock in tickers
    ]
    yesterday_cols = [
        preprocess.get_feat_name(col=stock, feat="pdiff", anchor=1, lag=1)
        for stock in tickers
    ]
    df2 = df[today_cols + yesterday_cols].dropna()

    pdiffs = np.concatenate([df2[col].values for col in today_cols])  # type: ignore
    a, b = np.quantile(pdiffs, [0.1, 0.9])

    plt.figure(figsize=(11, 4))
    result = dict()
    for name, thresh in dict(low=a, high=b).items():
        yesterday_pdiffs = []
        for today_col, yesterday_col in zip(today_cols, yesterday_cols):
            if name == "low":
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
    plt.title(
        "Histogram of Yesterdays Daily Price Changes corresponding to Today's Low and High Price Changes"
    )
    plt.legend()
    plt.show()

    return result


def control_random_daily(
    df: pd.DataFrame,
    p_cfg: Dict[str, Any],
    stocks: List[str],
    T: int = 250,
    num_rand: int = 10,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    start_iloc = -T
    end_iloc = -1

    ctrl_policy = policies.ControlPolicy(index_name=p_cfg["index_name"])
    random_policy = policies.RandomPolicy(stocks=stocks)

    plt.figure(figsize=figsize)
    control_vals, _ = evaluate.get_asset_values_and_stocks(
        df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=ctrl_policy
    )
    plt.plot(control_vals, "r", label="control")

    for ii in range(num_rand):
        random_vals, _ = evaluate.get_asset_values_and_stocks(
            df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=random_policy
        )

        label = "random" if ii == 0 else None
        plt.plot(random_vals, "g", label=label, alpha=0.2)
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("value")
    plt.title("Day by Day Growth of Control (Averaged) vs Random Stock from Averaged")
    first_day = df.index[start_iloc].strftime("%Y-%m-%d")
    last_day = df.index[end_iloc].strftime("%Y-%m-%d")

    plt.xlabel(f"trading day - between {first_day} and {last_day}")
    plt.ylabel("value")
    plt.title("Compare Control (Averaged) to Best YTD performance Stock (from averaged)")


def hist_random_vs_control(
    df: pd.DataFrame,
    p_cfg: Dict[str, Any],
    stocks: List[str],
    T: int = 250,
    num_rand: int = 1000,
    figsize: Tuple[int, int] = (10, 5),
) -> None:

    start_iloc = -T
    end_iloc = -1

    random_policy = policies.RandomPolicy(stocks=stocks)
    ctrl_policy = policies.ControlPolicy(index_name=p_cfg["index_name"])
    ctrl_final = evaluate.get_asset_final_value(
        df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=ctrl_policy
    )

    random_finals = [
        evaluate.get_asset_final_value(
            df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=random_policy
        )
        for _ in range(num_rand)
    ]
    random_normalized = np.array(random_finals) / ctrl_final

    plt.figure(figsize=(10, 5))
    plt.hist(random_normalized, bins=100)
    plt.axvline(1.0, color="red", label="control")
    plt.title("Random model vs Control")
    plt.xlabel("Final value / Control final value")


def control_max_YTD_daily(
    df: pd.DataFrame, p_cfg: Dict[str, Any], stocks: List[str], T: int = 240
) -> None:
    start_iloc = -T
    end_iloc = -1

    ctrl_policy = policies.ControlPolicy(index_name=p_cfg["index_name"])
    max_year_return_policy = policies.MaxFeatPolicy(
        df=df, stocks=stocks, feat="pdiff", anchor=0, lag=240
    )

    plt.figure(figsize=(10, 5))
    control_vals, _ = evaluate.get_asset_values_and_stocks(
        df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=ctrl_policy
    )
    plt.plot(control_vals, "r", label="control")

    max_feat_vals, max_feat_stocks = evaluate.get_asset_values_and_stocks(
        df=df, start_iloc=start_iloc, end_iloc=end_iloc, policy=max_year_return_policy
    )
    plt.plot(max_feat_vals, "g")

    day = np.arange(T)
    for stock in set(max_feat_stocks):
        stock_idx = np.array([el == stock for el in max_feat_stocks])

        plt.plot(day[stock_idx], np.array(max_feat_vals[1:])[stock_idx], "*", label=stock)
    plt.legend()
    first_day = df.index[start_iloc].strftime("%Y-%m-%d")
    last_day = df.index[end_iloc].strftime("%Y-%m-%d")

    plt.xlabel(f"trading day - between {first_day} and {last_day}")
    plt.ylabel("value")
    plt.title("Compare Control (Averaged) to Best YTD performance Stock (from averaged)")
