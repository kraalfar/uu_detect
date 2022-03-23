from utils import estimate_p_crit, get_criterion


def unreliable_data_detector(model,
                             sample1,
                             sample2,
                             p_crit=None,
                             criterion='u-test'):
    """
        :param model: model with predict method. Predict function should take sample from data and return anomaly scores
                        for each sample as numpy array
        :param sample1: sample from positive distributino for high alpha test and from unlabeled train distribution for
                        negative distribution shift test.
        :param sample2: sample from unlabeled distributino for high alpha test and from unlabeled test distribution for
                        negative distribution shift test
        :param p_crit: threshold for p_value. If None will be estimated from automatically. For negative shift we
                       suggest to set p_crit to None. If model is randomly initialized we suggest setting p_crit=None.
        :param criterion: What statistical test will be used for testing. We advise to use default values

        :return: returns the probability that proportion of positive data in unlabeled is high (> 0.9) with threshold value
        """

    test = get_criterion(criterion)
    if p_crit is None:
        p_crit = estimate_p_crit(model, sample1, test)

    pos_score = model.predict(sample1)
    unl_score = model.predict(sample2)


    pv = test(pos_score, unl_score).pvalue

    return pv, p_crit
