from logging import getLogger, basicConfig, INFO
from typing import Any, Dict
import pandas as pd


basicConfig(level=INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)


def preprocess_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    logger.info('Preprocessing data...')
    df = df[cfg['price']]
    df = df.sort_index()
    df = add_index_fund(df)
    df = add_cash_fund(df)
    for lag_feat in cfg['lag_feats']:
        df = add_lag_feat(df, lag_feat, cfg['trading_day_lags'])
    return df


def add_index_fund(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Adding index fund')
    df['index'] = df.sum(axis=1)
    return df


def add_cash_fund(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Adding index fund')
    df['cash'] = 1.0
    return df


def add_lag_feat(df, feat, trading_day_lags):
    logger.info(f'Adding {feat} lag features')
    for lag in trading_day_lags:
        for col in df.columns:
            cur = df[col]
            past = cur.shift(lag)
            if feat == 'lag':
                df[f'{col}_{feat}_{lag}d'] = past
            elif feat == 'diff':
                df[f'{col}_{feat}_{lag}d'] = cur - past
            elif feat == 'pdiff':
                df[f'{col}_{feat}_{lag}'] = 100.0 * (cur - past ) / past
    return df


    today_pdiffs = 100.0 * (df.shift(1) - df) / df
    yesterday_pdiffs = 100.0 * (df.shift(2) - df.shift(1)) / df.shift(1)
    df_pdiff0 = df.join(today_pdiffs, on=None, how='left', lsuffix='', rsuffix='_today_pdiff_1d')
    return df_pdiff0.join(yesterday_pdiffs, on=None, how='left', lsuffix='', rsuffix='_yesterday_pdiff_1d').dropna()

