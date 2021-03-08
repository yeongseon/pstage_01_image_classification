import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluation(gt_path, pred_path):
    """
    Args:
        gt_path (string) : root directory of ground truth file
        pred_path (string) : root directory of prediction file (output of inference.py)
    """
    num_classes = getattr(import_module("dataset"), args.dataset).num_classes
    gt = pd.read_csv(os.path.join(gt_path, f'gt.csv'))
    pred = pd.read_csv(os.path.join(pred_path, f'output.csv'))

    cls_report = classification_report(gt.values, pred.values, labels=np.arange(num_classes), output_dict=True)
    acc = cls_report['accuracy']
    f1 = np.mean([cls_report[str(i)]['f1-score'] for i in range(num_classes)])

    results = {'accuracy': acc, 'f1': f1}

    print(results)
    result_str = f'{results["accuracy"] * 100:.2f}%'

    return result_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MaskMultiClassDataset', help='dataset type (default: MaskMultiClassDataset)')
    args = parser.parse_args()

    gt_path = os.environ.get('SM_GROUND_TRUTH_DIR')
    pred_path = os.environ.get('SM_OUTPUT_DATA_DIR')

    result_str = evaluation(gt_path, pred_path)
    print(f'Final Score: {result_str}')