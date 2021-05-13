#' % Wine rating prediction using docker and luigi
#' % bearstrikesback
#' % 30.04.2021

#' # Introduction

#' This report was generated as the final output of the project pipeline.
#' This report consists of two parts, and it represents my view on the problem.
#' Part number 1 is the utility part. In this part I calculate some metrics,
#' plot some charts or import them from the output of another tasks of the pipeline.
#' Part number 2 is the presentation part. I present key points of my approach in this part.

#' # Part 1

#+ echo=False
# import packages
from pathlib import Path
from matplotlib import image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# read data
data = pd.read_csv('/usr/share/data/raw/wine_dataset.csv')

# get mean and median price per point
avg_price = data.groupby('points')['price'].mean().values
median_price = data.groupby('points')['price'].median().values
idx = data.groupby('points')['price'].mean().index.values

# plot mean and median price per point
plt.figure(figsize=(6, 4))
plt.scatter(idx[3:-5], avg_price[3:-5])
plt.scatter(idx[3:-5], median_price[3:-5], c='red')
plt.xlabel("Points")
plt.ylabel("Median Price")
plt.title(f"Correlation for means is: {np.round(pd.Series(idx[3:-5]).corr(pd.Series(avg_price[3:-5])), 2)}\n\
Correlation for medians is: {np.round(pd.Series(idx[3:-5]).corr(pd.Series(median_price[3:-5])), 2)}")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(idx[3:-5], np.log(avg_price[3:-5]))
plt.scatter(idx[3:-5], np.log(median_price[3:-5]), c='red')
plt.xlabel("Points")
plt.ylabel("Log of Median Price")
plt.title(f"Correlation for log(means) is: {np.round(pd.Series(idx[3:-5]).corr(pd.Series(np.log(avg_price[3:-5]))), 2)}\n\
Correlation for log(medians) is: {np.round(pd.Series(idx[3:-5]).corr(pd.Series(np.log(median_price[3:-5]))), 2)}")
plt.show()

# load and plot charts from previous task
PATH = '/usr/share/data/results/'
fe = image.imread(PATH + 'gbm_feat_importance.png')
res = image.imread(PATH + 'residuals.png')
scores = image.imread(PATH + 'scores.png')

plt.figure(figsize=(12, 10))
plt.imshow(fe)
plt.grid()
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 10))
plt.imshow(res)
plt.grid()
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 10))
plt.imshow(scores)
plt.grid()
plt.axis('off')
plt.show()

#' # Part 2
#' Key points from the case are:
#' 
#' * Wine sellers don't make money from the rating of the bottle. They make money from selling wine. It is a big question whether the wine seller will 
#' benefit from the rating of the wine bottle. We don't know the margin of one bottle sold as well as the type of the wine seller (i.e. whether it is a wholesaler 
#' aimed at the volume of bottles sold or a premium seller with high margin on one bottle and rare sales). Therefore, it is a bit difficult to say, whether ML is needed
#' here or not.
#' * If we speak about predicting rating of the wine bottle, then `np.log(price)` can be used as a good naive predictor. Mean and median prices grouped by each unique point of rating
#' are almost perfectly correlated with the rating itself. The higher the price of the wine bottle is, the higher the rating is. Simple linear regression (LR) on `np.log(price)` provides a very robust
#' baseline estimator with MSE of 6.0 and MAE of 1.95. Not to mention the fact that almost 40% of unique values of rating are outliers with less than 1% of total observations for each such point. We can expect that
#' once we have a larger dataset, the LR estimator will be even more accurate.
#' * We can go deeper and use gradient boosting, for example. It supports the original hypothesis that price is a good predictor of the rating. Price is the most important feature in the LBGM model. Additional features allow a model to capture
#' some non-linearity in the data and provide a better estimator than a simple LR model. The LGBM with built-in handling of categorical features and some common sense feature engineering (i.e. extracting the year of the wine from the description) provides a
#' model with MSE of 5.0 and MAE of 1.85. Not to mention the fact residuals of LGBM model are more bell shaped with more errors around 0 and fewer outliers.
#' * It definitely is not the best possible estimator. Some feature engineering as well as some parameter tuning can be further applied. However, it is outside of this project.
