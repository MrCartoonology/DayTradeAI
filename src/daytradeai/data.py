from typing import Dict, Optional
import os
import glob
from logging import getLogger, basicConfig, INFO

import pandas as pd
import yfinance as yf

from daytradeai.stocks import get_tickers


basicConfig(level=INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)


def get_stock_download_dir(cfg: Dict[str, str]) -> str:
    loc = os.path.join(cfg['data_dir'], cfg['stocks'])
    os.makedirs(loc, exist_ok=True)
    return loc


def get_downloaded_data(cfg: Dict[str, str]) -> Optional[pd.DataFrame]:
    stock_download_dir = get_stock_download_dir(cfg=cfg)
    timestamped_files = glob.glob(os.path.join(stock_download_dir, "*.parquet"))

    if timestamped_files:
        logger.info(f"Reading {len(timestamped_files)} files from {stock_download_dir}")
        return pd.concat([pd.read_parquet(file) for file in timestamped_files], axis=1)
    logger.warning(f"No files found in {stock_download_dir}")
    return None


def get_new_data(cfg: Dict[str, str], df_current: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    tickers = yf.Tickers(get_tickers(cfg['stocks'], num_tickers=cfg['num_tickers']))
    if df_current is None:
        logger.info("Fetching new data from scratch")
        df_new = tickers.history(period=cfg['period'], interval=cfg['interval'])
        df_new = df_new.dropna()
    else:
        last_date = df_current.index.max()
        start = last_date + pd.Timedelta(days=1)
        if start > pd.Timestamp.now().normalize():
            logger.info("No new data to fetch")
            return None
        logger.info(f"Fetching new data starting from {start}") 
        df_new = tickers.history(start=start, interval=cfg['interval'])
        df_new = df_new.dropna()
        if df_new.empty:
            logger.warning("No new data found")
            return None
    return df_new


def save_downloaded_data(df: Optional[pd.DataFrame], cfg: Dict[str, str]) -> None:
    if df is not None and not df.empty:
        max_date = df.index.max().strftime('%Y-%m-%d')
        stock_download_dir = get_stock_download_dir(cfg=cfg)
        file_path = os.path.join(stock_download_dir, f"{max_date}.parquet")
        logger.info(f"Saving data to {file_path}")
        df.to_parquet(file_path)
    else:
        logger.warning("No data to save")
