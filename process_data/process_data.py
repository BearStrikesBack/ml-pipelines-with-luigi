import re
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

import click

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
@click.option('--test-size')
def process_data(in_csv, out_dir, test_size):
    """
    Clean a csv file and split it into train/test sets.
    For simplicity, column with description and some other columns are dropped.
    Additional transformations like word2vec, OHE, encoders etc. can be further applied.

    Parameters
    ----------
    in_csv: str
        path to previously downloaded csv file.
    out_dir: str
        directory where files should be saved to.
    test_size: str
        size of the test dataset.

    Returns
    -------
    None
    """
    log = logging.getLogger('process-data')
    log.info('Processing dataset')
    log.info(f'Load file from {in_csv}')

    # read csv file and split into train/test sets
    data = pd.read_csv(in_csv)

    # get year from title, clean possible errors
    year = []

    for i in range(data.shape[0]):
        if len(re.findall(r'.*([1-3][0-9]{3})', data['title'][i])) == 0:
            year.append(np.nan)
        else:
            year.append(int(re.findall(r'.*([1-3][0-9]{3})', data['title'][i])[0]))

    data['year'] = year
    data.loc[(data['year'] < 1900)&(data['year'] > 2021), 'year'] = np.nan

    # apply log transformation to price
    data['price'] = np.log(data['price'])

    # make split
    train, test = train_test_split(data, test_size=float(test_size), \
                                    stratify=data['points'], random_state=42)

    log.info(f'Save train/test datasets to {out_dir}')

    # save datasets
    train.to_csv(Path(out_dir) / 'train.csv', index=False)
    test.to_csv(Path(out_dir) / 'test.csv', index=False)

    # save .SUCCESS flag
    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    process_data()
