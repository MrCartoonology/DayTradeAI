from logging import getLogger, basicConfig, INFO
import os
from typing import Any, Dict, List
import pandas as pd
from daytradeai.stocks import get_tickers


basicConfig(
    level=INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"  # remove seconds and milliseconds
)
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
            df=df, tickers=tickers_plus_cash, feat=lag_feat, anchor_days=preprocess_cfg["anchor_days"], lag_days=preprocess_cfg["lag_days"]
        )
    df = label_beat_index_1d(df, tickers_plus_cash, preprocess_cfg)
    return df


def add_cash_fund(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding index fund")
    df["cash"] = 1.0
    return df


def add_lag_feat(df, tickers, feat, anchor_days, lag_days):
    logger.info(f"Adding {feat} lag features")
    for anchor in anchor_days:
        for lag in lag_days:
            for col in tickers:
                cur = df[col]
                past = cur.shift(lag)
                feature_name = f"{col}_{feat}_{anchor}d_{lag}d"
                if feat == "lag":
                    df[feature_name] = past
                elif feat == "diff":
                    df[feature_name] = cur - past
                elif feat == "pdiff":
                    df[feature_name] = 100.0 * (cur - past) / past
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
        df[f"label_{stock}_pdiff_1f"] = df[f"{stock}_pdiff_0d_1d"].shift(-1)
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
    latest_day = df.index.max().strftime("%Y-%m-%d")
    os.makedirs(cfg['data_dir'], exist_ok=True)
    output = os.path.join(cfg['data_dir'], latest_day + '.parquet')
    if os.path.exists(output):
        logger.info(f"Overwriting preprocessed data: {output}")
    else:
        logger.info(f"Saving preprocessed data to {output}")
    df.to_parquet(output)


def get_feature_columns(df: pd.DataFrame, suffix: str, tickers: List[str]) -> List[str]:
    cols = [col for col in df.columns if col.endswith(suffix) and col.split('_')[0] in tickers]
    assert len(cols) > 0, f'No columns ending with {suffix} and starting with strings in tickers found, tickers={tickers[0:3] } ... {tickers[-3:]}'
    return cols

