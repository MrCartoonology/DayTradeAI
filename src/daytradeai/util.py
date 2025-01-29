from scipy.stats import ks_2samp, ttest_ind
from scipy.stats import mannwhitneyu


def get_num_periods(interval: str) -> float:
    int2periods = {
        "1d": 252,
        "1w": 52,
        "1m": 12,
        "2d": 252 / 2,
        "3d": 252 / 3,
    }
    assert (
        interval in int2periods
    ), f"Invalid interval. Must be one of {list(int2periods.keys())}."
    return int2periods[interval]


def get_interval_rate(annual: float, interval: str) -> float:
    n_periods = get_num_periods(interval=interval)
    return (1 + annual) ** (1 / n_periods) - 1


def goodnes_fit_tests(sample, population):
    res = dict()
    for stat_name, fn in dict(
        ks_2samp=ks_2samp,
        ttest_ind=ttest_ind,  # just means
        mannwhitneyu=mannwhitneyu,  # continuous, not neccessarily normal
    ).items():
        stat, pval = fn(sample, population)
        res[stat_name] = dict(pval=pval, stat=stat)
    return res
