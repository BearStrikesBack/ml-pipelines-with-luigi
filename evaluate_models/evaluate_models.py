import pickle
import json
from pathlib import Path
import logging
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

import click

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
logging.basicConfig(level=logging.INFO)


def eval_lr(data, in_dir, scores):
    """
    Evaluate linear model.

    Parameters
    ----------
    data: pd.DataFrame
        data for model to be tested on.
    in_dir: str
        directory where pretrained models are saved.
    scores: dict
        dictionary to save scores of the model.

    Returns
    -------
    mae: float
        mean absolute error score.
    mse: float
        mean squared error score.
    residuals: array-like of shape (n_samples, 1)
        array of residuals for model predictions.
    """
    # prepare data
    no_nans = data.copy()
    no_nans.loc[no_nans['price'].isnull(), 'price'] = no_nans['price'].mean()
    X_test = no_nans['price'].values.reshape(-1, 1)
    y_test = no_nans['points'].values.reshape(-1, 1)

    # load model and make predictions
    lr = pickle.load(open(Path(in_dir) / 'linear.pkl', 'rb'))
    y_pred = lr.predict(X_test)

    # evaluate predictions on test set
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    residuals = y_test - y_pred

    # update dict with scores
    scores['score'].extend([np.round(mae, 2), np.round(mse, 2)])
    scores['metrics'].extend(['mae', 'mse'])
    scores['model'].extend(['linear', 'linear'])

    return mae, mse, residuals


def eval_lgb(data, in_dir, scores, out_dir):
    """
    Evaluate gbm model.

    Parameters
    ----------
    data: pd.DataFrame
        data for model to be tested on.
    in_dir: str
        directory where pretrained models are saved.
    scores: dict
        dictionary to save scores of the model.
    out_dir: str
        directory where results should be saved to.

    Returns
    -------
    mae: float
        mean absolute error score.
    mse: float
        mean squared error score.
    residuals: array-like of shape (n_samples, 1)
        array of residuals for model predictions.
    """
    # cast cat columns, so lbg can handle cat encoding
    coltocat = ['country', 'province', 'taster_name', \
                'region_1', 'region_2', 'variety']

    for col in coltocat:
        data[col] = data[col].astype("category")

    # prepare data
    coltouse = ['country', 'price', 'province', 'taster_name', \
                'region_1', 'region_2', 'variety', 'year']

    X_test = data[coltouse]
    y_test = data['points']

    # load model and make predictions
    gbm = joblib.load(Path(in_dir) / 'gbm.pkl')
    y_pred = gbm.predict(X_test)

    # evaluate predictions on test set
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    residuals = y_test.values.reshape(-1, 1) - y_pred.reshape(-1, 1)

    # update dict with scores
    scores['score'].extend([np.round(mae, 2), np.round(mse, 2)])
    scores['metrics'].extend(['mae', 'mse'])
    scores['model'].extend(['lightgbm', 'lightgbm'])

    # plot feature importance and save png
    feature_imp = pd.DataFrame({'Value':gbm.feature_importance(), 'Feature':X_test.columns})

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), \
                palette="gray")
    plt.title('LGBM Feature Importance')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'gbm_feat_importance.png', dpi=200)

    return mae, mse, residuals


def draw_residuals(residuals_gbm, residuals_lr, out_dir):
    """
    Draw residuals of both models to compare.

    Parameters
    ----------
    residuals_gbm: array-like of shape (n_samples, 1)
        residuals from gbm model predictions.
    residuals_lr: array-like of shape (n_samples, 1)
        residuals from lr model predictions.
    out_dir: str
        directory where results should be saved to.

    Returns
    -------
    None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    bins = 30

    ax1.hist(residuals_gbm.reshape(-1, 1), bins=bins, color='b', density=True)
    ax1.set_xlabel("Residuals")
    ax1.set_ylabel("Freq")
    ax1.set_title('Residuals distribution for lgb')
    ax1.grid(axis='y')
    ax1.set_xlim(-10, 10)

    ax2.hist(residuals_lr.reshape(-1, 1), bins=bins, color='r', density=True)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Freq")
    ax2.set_title('Residuals distribution for lr')
    ax2.grid(axis='y')
    ax2.set_xlim(-10, 10)

    ax3.hist(residuals_gbm.reshape(-1, 1), bins=bins, color='b', density=True)
    ax3.hist(residuals_lr.reshape(-1, 1), bins=bins, color='r', alpha=0.6, density=True)
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Freq")
    ax3.set_title('Residuals distribution for both models')
    ax3.legend(['Lgb model', 'Linear model'])
    ax3.grid(axis='y')
    ax3.set_xlim(-10, 10)
    plt.tight_layout()

    plt.savefig(Path(out_dir) / 'residuals.png', dpi=200)


def draw_scores(scores, out_dir):
    """
    Draw metric perfomance for both models.

    Parameters
    ----------
    scores: dict
        dictionary to save scores of the model.
    out_dir: str
        directory where results should be saved to.

    Returns
    -------
    None
    """
    df = pd.DataFrame(scores)

    plt.figure(figsize=(6, 6))
    sns.barplot(x="metrics", y="score", hue="model", data=df, palette=['r', 'b'], saturation=1)
    plt.title('Scores of models')
    plt.legend(loc=2)
    plt.tight_layout()

    plt.savefig(Path(out_dir) / 'scores.png', dpi=200)


@click.command()
@click.option('--in-csv')
@click.option('--in-dir')
@click.option('--out-dir')
def evaluate_models(in_csv, in_dir, out_dir):
    """
    Evaluate models on holdout dataset.

    Parameters
    ----------
    in_csv: str
        path to test dataset
    in_dir: str
        directory where pretrained models are saved.
    out_dir: str
        directory where results should be saved to.

    Returns
    -------
    None
    """
    log = logging.getLogger('evaluate-models')
    log.info('Evaluating models')
    log.info(f'Load data from {in_csv}')

    # read data and prepare dict to save scores
    data = pd.read_csv(in_csv)
    scores = {'score':[], 'metrics':[], 'model':[]}

    log.info('Evaluate linear model')
    log.info(f'Load linear model from {in_dir}linear.pkl')

    # evaluate lr model
    mae_lr, mse_lr, residuals_lr = eval_lr(data, in_dir, scores)

    log.info(f'Linear model MAE is {mae_lr:.2f}, MSE is {mse_lr:.2f}')
    log.info('Evaluate lightgbm model')
    log.info(f'Load lightgbm model from {in_dir}gbm.pkl')

    # evaluate gbm model
    mae_gbm, mse_gbm, residuals_gbm = eval_lgb(data, in_dir, scores, out_dir)

    log.info(f'Lightgbm model MAE is {mae_gbm:.2f}, MSE is {mse_gbm:.2f}')

    # plot some charts
    draw_residuals(residuals_gbm, residuals_lr, out_dir)
    draw_scores(scores, out_dir)

    # save dict with scores as json
    with open(Path(out_dir) / 'scores.json', 'w') as file:
        json.dump(scores, file)

    # saving .SUCCESS flag
    flag = Path(out_dir) / '.SUCCESS'
    flag.touch()


if __name__ == '__main__':
    evaluate_models()
