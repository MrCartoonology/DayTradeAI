import copy as cp
from logging import getLogger, basicConfig, INFO
from typing import Any, Dict, List
import pandas as pd
from daytradeai.stocks import get_tickers


basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


def preprocess_data(
    df: pd.DataFrame, data_cfg: Dict[str, Any], preprocess_cfg: Dict[str, Any]
) -> pd.DataFrame:
    logger.info("Preprocessing data...")

    df = df[preprocess_cfg["price"]]
    df = df.sort_index()
    tickers = get_tickers(group=data_cfg["stocks"])
    df = add_cash_fund(df)
    tickers_plus_cash = tickers + ["cash"]

    for lag_feat in preprocess_cfg["lag_feats"]:
        df = add_lag_feat(
            df, tickers_plus_cash, lag_feat, preprocess_cfg["trading_day_lags"]
        )
    df = label_beat_index_1d(df, tickers_plus_cash, preprocess_cfg)
    return df


def add_cash_fund(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding index fund")
    df["cash"] = 1.0
    return df


def add_lag_feat(df, tickers, feat, trading_day_lags):
    logger.info(f"Adding {feat} lag features")
    for lag in trading_day_lags:
        for col in tickers:
            cur = df[col]
            past = cur.shift(lag)
            if feat == "lag":
                df[f"{col}_{feat}_{lag}d"] = past
            elif feat == "diff":
                df[f"{col}_{feat}_{lag}d"] = cur - past
            elif feat == "pdiff":
                df[f"{col}_{feat}_{lag}d"] = 100.0 * (cur - past) / past
    return df


def label_beat_index_1d(
    df: pd.DataFrame, stocks: List[str], preprocess_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    add column label_{ticker} that is 1 if the stock outperforms the index by pdiff. The index is a equally weighted
    investimement in stocks. The labels should be 50/50 for success/failure, overall.
    """
    logger.info("Adding label")
    if "pdiff" not in preprocess_cfg["lag_feats"]:
        raise ValueError("Label requires pdiff")

    # index performance
    for stock in stocks:
        # get next days performance of each stock in index
        df[f"label_{stock}_pdiff_1f"] = df[f"{stock}_pdiff_1d"].shift(-1)
    # assume index is equally weighted -
    # note this is note how DIJA or S&P500 is calculated (DIJA is price weighted, with divisor, S&P500 is market cap weighted)
    index_performance = df[[f"label_{stock}_pdiff_1f" for stock in stocks]].mean(axis=1)

    # create labels for each stock, 0/1 for
    for stock in stocks:
        df[f"label_{stock}"] = (
            df[f"label_{stock}_pdiff_1f"] > index_performance
        ).astype(int)

    return df


def save_preprocessed(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    pass
