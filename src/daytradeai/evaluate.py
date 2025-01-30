import pandas as pd
from typing import List, Tuple

import daytradeai.policies as policies


def get_index_perf(df: pd.DataFrame, stocks: List[str]) -> pd.Series:
    """index performance is from equally weighted investment in each stock, for
    one day.

    Args:
        df (pd.DataFrame): contains 1d pdiff performance for each stock
        stocks (List[str]): stocks to average

    Returns:
        pd.Series: _description_
    """
    cols = [f"label_{stock}_pdiff_1f" for stock in stocks]
    return df[cols].mean(axis=1)


def add_index_performance(
    df: pd.DataFrame, stocks: List[str], index_name: str
) -> pd.DataFrame:
    """adds the column label_{index_name}_pdiff_1f to the dataframe with
    the index performance.

    Args:
        df (pd.DataFrame): contains stock performance
        stocks (List[str]): stocks to average
        index_name (str): 'stock name' fo average

    Returns:
        pd.DataFrame: modifies df, adds column
    """
    df[f"label_{index_name}_pdiff_1f"] = get_index_perf(df=df, stocks=stocks)
    return df


def get_next_value_and_stock(
    df: pd.DataFrame, iloc: int, policy: policies.Policy, v: float
) -> Tuple[float, str]:
    """gets stock pick for iloc and new value after investing in that stock for a day.

    Args:
        df (pd.DataFrame): dataframe with stock performance
        iloc (int): index/day
        policy (policies.Policy): picks stock for day
        v (float): initial value

    Returns:
        Tuple[float, str]: new value, and stock picked
    """
    stock = policy.get_stock(iloc=iloc)
    pdiff = df[f"label_{stock}_pdiff_1f"].iloc[iloc]
    ratio = 1 + pdiff / 100.0
    return v * ratio, stock


def get_asset_values_and_stocks(
    df: pd.DataFrame,
    start_iloc: int,
    end_iloc: int,
    policy: policies.Policy,
    v0: float = 1.0,
) -> Tuple[List[float], List[str]]:
    """starting with v0 prior to start_day, asks policy for the next stock to buy each day until end_day.
    Returns list of value changes and stocks picked.

    Args:
        df: (pd.DataFrame): dataframe with stock performance
        start_iloc, end_iloc: these are integer location values into df for the days to use.
        policy (policies.Policy): returns stock pick for that iloc
        v0 (float, optional): initial value. Defaults to 1.0.

    Returns:
        Tuple[List[float], List[str]]: list of values and stocks. There will be one less value in stocks
        than values
    """
    vals = [v0]
    stocks = []

    for iloc in range(start_iloc, end_iloc + 1):
        v, stock = get_next_value_and_stock(df=df, iloc=iloc, policy=policy, v=vals[-1])
        vals.append(v)
        stocks.append(stock)

    return vals, stocks


def get_asset_final_value(
    df: pd.DataFrame,
    start_iloc: int,
    end_iloc: int,
    policy: policies.Policy,
    val: float = 1.0,
) -> float:
    for iloc in range(start_iloc, end_iloc + 1):
        val, _ = get_next_value_and_stock(df=df, iloc=iloc, policy=policy, v=val)
    return val
