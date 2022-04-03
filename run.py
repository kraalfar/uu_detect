import sys

sys.path.insert(1, 'utils')
sys.path.insert(1, 'example')
sys.path.insert(1, 'example/model')

from example import dataset, nnpu
from uud_detector import *

pos_cls = [0, 1, 8, 9]


def run():
    train_holder = dataset.get_data(True, None, None, False)
    test_holder = dataset.get_data(False, None, None, False)

    train_bin = train_holder.pos_neg_split(pos_cls)
    test_bin = test_holder.pos_neg_split(pos_cls)

    for alpha in [0.05, 0.5, 0.95]:
        train_data, _ = train_bin.get_dataset(alpha, svm_labels=False, size_lab=2500, size_unl=5000)
        test_data, _ = test_bin.get_dataset(alpha, c=0.5, svm_labels=False)

        f = nnpu.nnPU()
        f.fit(train_data)

        res = unreliable_data_detector(f, test_data.lab_data(lab=1), test_data.lab_data(lab=0), 0.1)

        print(f"positive proportion={alpha:.2f}. pvalue={res[0]:.4f}", flush=True)


if __name__ == '__main__':
    run()
