import scipy.stats as stats
from sklearn.model_selection import train_test_split
import numpy as np


def estimate_p_crit(model, sample, test, iters=10):
    pv = []
    for _ in range(iters):
        sample1, sample2 = train_test_split(sample)
        score1, score1 = model.predict(sample1), model.predict(sample2)
        pv.append(test(score1, score1).pvalue)

    return np.mean(pv)


def get_criterion(criterion_name):
    if criterion_name == 'u-test':
        return stats.mannwhitneyu
    elif criterion_name == 't-test':
        return stats.ttest_ind
    else:
        raise ValueError(f"Wrong criterion name {criterion_name}")
