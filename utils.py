import scipy.stats as stats


def estimate_p_crit(model, sample):
    # TODO
    return 0.1


def get_criterion(criterion_name):
    if criterion_name == 'u-test':
        return stats.mannwhitneyu
    elif criterion_name == 't-test':
        return stats.ttest_ind
    else:
        raise ValueError(f"Wrong criterion name {criterion_name}")
