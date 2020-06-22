import argparse
import json
from pathlib import Path
import time

import cv2

from processing.car_plate import CarPlate
import evaluate.calculate as calc

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument('images_dir', type=str)
    #parser.add_argument('results_file', type=str)
    parser.add_argument('--images_dir', type=str, default='train/')
    parser.add_argument('--results_file', type=str, default='results.json')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}

    CP = CarPlate()

    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = CP.process(image)

    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)

    stop = time.time()
    full = stop - start
    print(f'Full time: {full}s\nTime per image: {full/len(images_paths)}s')


if __name__ == '__main__':
    main()

    calc.main(Path('results.json'), Path('evaluate/train.json'))
