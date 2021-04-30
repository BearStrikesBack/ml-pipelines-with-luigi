from pathlib import Path
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import click
import logging
import pickle
import joblib

import pandas as pd
import numpy as np
import lightgbm as lgb


logging.basicConfig(level=logging.INFO)


def lr_model(data, out_dir):
    """
    Train linear model and save to pickle.
    
    Parameters
    ----------
    data: pd.DataFrame
        data for model to be trained on.
    out_dir: str
        directory where models should be saved to.

    Returns
    -------
    None
    """
    # prepare data for LR model
    no_nans = data.copy()
    no_nans.loc[no_nans['price'].isnull(), 'price'] = no_nans['price'].mean()
    X_train = no_nans['price'].values.reshape(-1, 1)
    y_train = no_nans['points'].values.reshape(-1, 1)

    # train LR
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    # save model to pickle
    pickle.dump(reg, open(Path(out_dir) / 'linear.pkl', 'wb'))


def gmb_model(data, out_dir):
    """
    Train gbm model and save to pickle.

    Parameters
    ----------
    data: pd.DataFrame
        data for model to be trained on.
    out_dir: str
        directory where models should be saved to.

    Returns
    -------
    None
    """
    # cast cat dtype to columns of interest
    cat_features = ['country', 'province', 'taster_name', \
                    'region_1', 'region_2', 'variety']

    for col in cat_features:
        data[col] = data[col].astype("category")
    
    coltouse = ['country', 'price', 'province', 'taster_name', \
                'region_1', 'region_2', 'variety', 'year']
    
    # get val set for early stop
    X_train, X_val, y_train, y_val = train_test_split(data[coltouse], \
                                                      data[['points']], \
                                                      test_size=0.2, \
                                                      random_state=42)
    
    # set params
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': 0.04,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # prepare data
    train_data = lgb.Dataset(X_train, 
                             y_train, 
                             categorical_feature=cat_features, 
                             free_raw_data=False)

    valid_data = lgb.Dataset(X_val, 
                             y_val, 
                             categorical_feature=cat_features, 
                             free_raw_data=False)

    # train model
    gbm = lgb.train(params, 
                    train_data, 
                    valid_sets=[valid_data], 
                    verbose_eval=False, 
                    categorical_feature=cat_features, 
                    num_boost_round=1200,
                    early_stopping_rounds=100)

    # save model 
    joblib.dump(gbm, Path(out_dir) / 'gbm.pkl')


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def train_models(in_csv, out_dir):
    """
    Train models on dataset.
    Save pickled models to out_dir.

    Parameters
    ----------
    in_csv: str
        path to train dataset.
    out_dir: str
        directory where models should be saved to.

    Returns
    -------
    None
    """
    log = logging.getLogger('train-models')
    log.info('Training models')
    log.info(f'Load data from {in_csv}')

    # read data
    data = pd.read_csv(in_csv)

    log.info('Train simple linear model')
    log.info(f'Save linear model to {out_dir}linear.pkl')

    # train linear model
    lr_model(data, out_dir)

    log.info('Train simple lightgbm model')
    log.info(f'Save lightgbm model to {out_dir}gbm.pkl')

    # train gbm model
    gmb_model(data, out_dir)

    # saving .SUCCESS flag
    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    train_models()
