import argparse
import json
from pathlib import Path
import numpy as np


def check(res: dict, gt: dict) -> None:
    """[summary]

    Parameters
    ----------
    res : dict
        [description]
    gt : dict
        [description]
    """

    points = 0
    count = 0

    for key, val in res.items():
        val_gt = gt.get(key, -1)

        if val_gt == -1:
            print(f'No find {key} in ground truth - skipping')
            continue

        if len(val) != len(val_gt):
            count += len(val_gt)
            print(f'{key}: {val} - should be {val_gt}')
            continue

        point = np.sum(np.array(list(val_gt)) == np.array(list(val)))
        points += point
        count += len(val_gt)

        if point != len(val_gt):
            print(f'{key}: {val} - should be {val_gt}')

    try:
        print(
            f'Accurancy: {points/count:2f} ({points} good chars per {count} total)')
    except ZeroDivisionError:
        print(f'Accurancy: {0.00} ({points} good chars per {count} total)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str)
    parser.add_argument('ground_truth_file', type=str)
    args = parser.parse_args()

    results_file = Path(args.results_file)
    ground_truth_file = Path(args.ground_truth_file)

    with results_file.open('r') as f:
        res = json.load(f)

    with ground_truth_file.open('r') as f:
        gt = json.load(f)

    check(res, gt)


if __name__ == "__main__":
    main()