import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hist_pdiff_1d(df: pd.DataFrame) -> None:
    """Plot histogram and summariy statitics of all daily price changes - current day.
    
    Returns all the daily price changes in single array
    """
    pdiffs_cols = [col for col in df.columns if col.endswith('_today_pdiff_1d')]
    assert len(pdiffs_cols) > 0, 'No columns ending with "_today_pdiff_1d" found'
    pdiffs = np.concatenate([df[col].values for col in pdiffs_cols])
    plt.hist(pdiffs, bins=400, label=f'mu={pdiffs.mean():.3f}\nmedian={np.median(pdiffs):.3f}\nsigma={pdiffs.std():.3f}\nn={len(pdiffs)}')
    plt.xlabel('Daily Price Change (%)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Daily Price Changes')
    plt.legend()
    return pdiffs


def correlate_yesterday_pdiff_with_today(df):
    """take top 10% and bottom 10% of daily price change and plot histogram
    and summary statistics of yesterday's daily price change for each group.
    Return concatenated array of yesterday's daily price change for high and low groups,
    as well as all of todays and yesterdays price changes.
    """
    pdiffs = dict()
    for suffix in ['_today_pdiff_1d', '_yesterday_pdiff_1d']:
        cols = [col for col in df.columns if col.endswith(suffix)]
        assert len(cols) > 0, f'No columns ending with "{suffix}" found'
        pdiffs[suffix] = np.concatenate([df[col].values for col in cols])
    for x, y in pdiffs.items():
        print(x, len(y))
    quantiles = dict(lower=np.percentile(pdiffs['_today_pdiff_1d'], 10), upper=np.percentile(pdiffs['_today_pdiff_1d'], 90))
    print(quantiles)
    idxes = dict(low=pdiffs['_today_pdiff_1d'] < quantiles['lower'], high=pdiffs['_today_pdiff_1d'] > quantiles['upper'])

    for label, idx in idxes.items():
        v = pdiffs['_yesterday_pdiff_1d'][idx]
        mu = np.mean(v) 
        std = np.std(v)
        plt.hist(v, bins=500, label=f'{label} mean={mu:.3f}\nmedian={np.median(v):.3f}\nstd={std:.3f}\nn={len(v)})')
        pdiffs[label] = v

    plt.xlabel('Daily Price Change (%)')
    plt.ylabel('Frequency')
    plt.title('Histogram of yesterday Daily Price Changes for high/low of today')
    plt.legend()
    return pdiffs

