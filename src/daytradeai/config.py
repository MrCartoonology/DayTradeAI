from typing import Any, Dict


cfg_data: Dict[str, Any] = dict(
    stocks="dowjones",
    period="5y",
    interval="1d",
    data_dir="/Users/davidschneider/data/daytradeai/prd/yfinance_downloads",
    num_tickers=-1,
)

cfg_data_dbg = cfg_data.copy()
cfg_data_dbg.update(
    period="1mo",
    data_dir="/Users/davidschneider/data/daytradeai/dbg/yfinance_downloads",
    num_tickers=2,
)


cfg_preprocess: Dict[str, Any] = dict(
    price="Open",
    anchor_and_lags={
        0: [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180, 210, 240], 
        1: [1]
    },
    lag_feats=["diff", "pdiff", "lag"],  # only need lag for a deep neural network?
    data_dir="/Users/davidschneider/data/daytradeai/prd/preprocessed",
)

cfg = dict(data=cfg_data, preprocess=cfg_preprocess)

cfg_dbg = cfg.copy()
cfg_dbg["data"] = cfg_data_dbg
