import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluation(gt_dir, pred_dir):
    """
    Args:
        gt_dir (string) : root directory of ground truth file
        pred_dir (string) : root directory of prediction file (output of inference.py)
    """
    num_classes = getattr(import_module("dataset"), args.dataset).num_classes
    results = {}

    gt = pd.read_csv(os.path.join(gt_dir, 'gt.csv'))
    pred = pd.read_csv(os.path.join(pred_dir, 'output.csv'))
    cls_report = classification_report(gt.ans.values, pred.ans.values, labels=np.arange(num_classes), output_dict=True)
    acc = round(cls_report['accuracy'] * 100, 2)
    f1 = np.mean([cls_report[str(i)]['f1-score'] for i in range(num_classes)])
    f1 = round(f1, 2)

    results['accuracy'] = {
        'value': acc,
        'rank': True,
        'decs': True,
    }
    results['f1'] = {
        'value': f1,
        'rank': False,
        'decs': True,
    }

    return json.dumps(results)


#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--dataset', type=str, default='MaskMultiClassDataset', help='dataset type (default: MaskMultiClassDataset)')
#    args = parser.parse_args()
#
#    gt_dir = os.environ.get('SM_GROUND_TRUTH_DIR')
#    pred_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
#
#    result_str = evaluation(gt_dir, pred_dir)
#    print(f'Final Score: {result_str}')
