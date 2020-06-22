import argparse
import json
from pathlib import Path
import numpy as np


def check(res: dict, gt: dict, show: bool) -> None:
    """Check if numbers for plate are correct.

    Parameters
    ----------
    res : dict
        Car plates numbers recognized by decoder.
    gt : dict
        True car plates numbers.
    show : bool
        Print incorrect plate's numbers.
    """

    points = 0
    count = 0
    readed = len(res.items())-1

    for key, val in res.items():
        val_gt = gt.get(key, -1)

        if val_gt == -1:
            if show == True:
                print(f'No find {key} in ground truth - skipping')
            readed -= 1
            continue

        if val == '???????':
            readed -= 1

        if len(val) != len(val_gt):
            count += len(val_gt)
            if show == True:
                print(f'{key}: {val} - should be {val_gt}')
            continue

        point = np.sum(np.array(list(val_gt)) == np.array(list(val)))
        points += point
        count += len(val_gt)

        if point != len(val_gt) and show == True:
            print(f'{key}: {val} - should be {val_gt}')

    try:
        print(
            f'Accurancy: {points/count:2f} ({points} good chars per {count} total)')
        print(
            f'Find acc: {readed/(len(gt.items())-1):2f} ({readed} readed plates for {(len(gt.items())-1)} total)'
        )
        print(
            f'OCR acc: {points/(readed*7):2f} ({points} readed chars for {readed*7} total)'
        )
    except ZeroDivisionError:
        print(f'Accurancy: {0.00} ({points} good chars per {count} total)')


def main(results_file: str, ground_truth_file: str, show: bool=False):
    """Main function of result calculator.

    Parameters
    ----------
    results_file : str
        JSON file with car plates decoder results.
    ground_truth_file : str
        JSON file with true results of car plates.
    show : bool
        Print incorrect plate's numbers.
    """

    with results_file.open('r') as f:
        res = json.load(f)

    with ground_truth_file.open('r') as f:
        gt = json.load(f)

    check(res, gt, show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str)
    parser.add_argument('ground_truth_file', type=str)
    args = parser.parse_args()

    results_file = Path(args.results_file)
    ground_truth_file = Path(args.ground_truth_file)

    main(results_file, ground_truth_file, True)