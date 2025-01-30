import numpy as np
from typing import List

import pandas as pd
import daytradeai.preprocess as preprocess


class Policy:
    def __init__(self):
        pass

    def get_stock(self, iloc: int) -> str:
        raise NotImplementedError


class ControlPolicy(Policy):
    def __init__(self, index_name: str):
        super().__init__()
        self.index_name = index_name

    def get_stock(self, iloc: int) -> str:
        return self.index_name


class RandomPolicy(Policy):
    def __init__(self, stocks: List[str]):
        super().__init__()
        self.stocks = stocks

    def get_stock(self, iloc: int) -> str:
        return np.random.choice(self.stocks)


class MaxFeatPolicy(Policy):
    def __init__(
        self, df: pd.DataFrame, stocks: List[str], feat: str, anchor: int, lag: int
    ):
        super().__init__()
        self.df = df
        self.stocks = stocks
        self.feat = feat
        self.anchor = anchor
        self.lag = lag

    def get_stock(self, iloc: int) -> str:
        feat_cols = [
            preprocess.get_feat_name(
                col=stock, feat=self.feat, anchor=self.anchor, lag=self.lag
            )
            for stock in self.stocks
        ]
        stock_idx = np.argmax(self.df[[col for col in feat_cols]].iloc[iloc])
        return self.stocks[stock_idx]
