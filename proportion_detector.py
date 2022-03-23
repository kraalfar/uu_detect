from utils import estimate_p_crit, get_criterion


def high_proportion_detector(model,
                             pos_sample,
                             unl_sample,
                             p_crit=0.1,
                             criterion='u-test'):
    """
    :param model: model with predict method. Predict function should take sample from data and return anomaly scores
                    for each sample as numpy array
    :param pos_sample: sample from positive distribution
    :param unl_sample: sample from unlabeled distribution
    :param p_crit: threshold for p_value. If None will be estimated from automatically. If model is randomly initialized
                    we suggest setting p_crit=None. Same goes for .
    :param criterion: What statistical test will be used for testing. We advise to use default values

    :return: returns the probability that proportion of positive data in unlabeled is high (> 0.9) with threshold value
    """

    if p_crit is None:
        p_crit = estimate_p_crit(model, pos_sample)

    pos_score = model.predict(pos_sample)
    unl_score = model.predict(unl_sample)

    test = get_criterion(criterion)
    pv = test(pos_score, unl_score).pvalue

    return pv, p_crit
