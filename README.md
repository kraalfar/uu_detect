# Detecting unreliable unlabeled data

This repository accompanies paper ["Improving State-of-the-Art in One-Class Classification by Leveraging Unlabeled Data"](https://arxiv.org/abs/2203.07206) and contains implementations of algorithms from section 5.8 from it.


## High alpha identification

* To test if proportion of positive data in unlabeled is high, one need samples from each distribution and a model that outputs probability of sample being positive (i.e. OC or PU model)
* Running ```unreliable_data_detector(model, positive_sample, negative_sample, 0.1)``` will perform steps 2-4 of the algorithm and output two values: pv and p_crit
* If pv > p_crit then it is likely that proportion of positive data is high

## Negative shift identification

* To test if negative shift has occurred, one need samples from two distributions and a model that outputs probability of sample being positive (PU model)
* Running ```unreliable_data_detector(model, sample1, sample2, None)``` will perform steps 2-4 of the algorithm and output two values: pv and p_crit
* If pv < p_crit then it is likely that negative shift occurred
## Example

You can run an example by running following prompt
```python run.py```