import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

import pca

TIME_TEMPLATE = '%Y%m%d%H%M%S'


def main(args):
    result_path = Path(args.result_path)
    timestamp = datetime.now().strftime(TIME_TEMPLATE)
    result_path = result_path/timestamp
    if not result_path.exists():
        try:
            result_path.mkdir(parents=True)
        except Exception as err:
            print(err)

    # loading data
    df1 = pd.read_csv('../data1.csv')
    df2 = pd.read_csv('../data2.csv')
    df3 = pd.read_csv('../data3.csv')

    # df to nd ndarray
    data1 = df1.values
    data2 = df2.values
    data3 = df3.values

    # data1.csv
    pca1 = pca.PCA()
    pca1.fit(data1)

    # data2.csv
    pca2 = pca.PCA()
    pca2.fit(data2)

    # data3.csv
    pca3 = pca.PCA()
    pca3.fit(data3)


if __name__ == "__main__":
    description = 'Example: python main.py ./result'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('result_path', default='./result', help='path to save the result')

    args = parser.parse_args()

    main(args)
