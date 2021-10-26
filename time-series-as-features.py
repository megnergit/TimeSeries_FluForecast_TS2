# |------------------------------------------------------------------
# | # Flu Prediction  - Time Series Analysis TS2
# |------------------------------------------------------------------
# |
# | ## 1. Introduction
# |
# | This is a notebook to practice the routine procedures
# | commonly used in the time sequence analysis.
# | This notebook is based on the Kaggle [Time Series Analysis](https://www.kaggle.com/learn/time-series)
# | offered by Ryan Holbrook.

# | A time series consists of __trend__, __seasonality__, __cycles__,
# | and __peculiarity__. For each features, we have a procedure to deal with,
# |
# | - For __trend__ : Analytical fitting of the baselines (linear, polynomial, etc)
# | - For __seasonality__ : Fourier decomposition.
# | - For __cycle__ : Lags.
# | - For __peculiarity__ : Categorical features.
# |
# | In this notebook we will learn how to use lagged features to make predictions.

# | We also learn some terminology.
# | - __Forecast horizon__ : a part of the time-sequence where we do forecast.
# | - __Forecast origin__ : a point of the time-sequence where the training data ends.
# | - __Lead time__ : a part of the time-sequence after the forecast origin, but before
# | the forecast horizon starts.


# | When we have a forecast horizon longer than one unit time, the prediction requires,
# | - __Multioutput model__.

# | We have a couple of strategies how to create a multiple output.
# | - __Direct strategy__ :  Create one model for each day in the horizon,
# | and perform the prediction directly.
# | One needs so many models as the forecasting points in the forecast horizon.
# | - __Recursive strategy__ :  First, train a model to predict the first
# | day in the horizon. Only the given training data is used for the training.
# | Use that same model to predict the second day in the horizon, but
# | now we have one new input from the day before (=the forecast on the first
# | day in the horizon).
# | - __DirRec strategy__ : Combination of the  above two. Create a
# | model to forecast on the first day in the horizon. Use that new information
# | as a ground truth, and create the second model to forecast on the second day.
# | One needs so many models as the forecasting points in the forecast horizon.

# | ## 2. Task
# | From the historical record of the visits to the doctor's office in the past,
# | we will forecast the numbers of  such visits  in the future.

# | ## 3. Data
# | 1. The historical record of the visits to the doctor's office
# |     over a week, starting from 2009 and ending in 2016.
# |
# | 2. The data above comes with the Google search records related to
# |    a flu. The keywords and the number of searches are tabulated for
# |    each week.

# | ## 4. Notebook
# -------------------------------------------------------
# | Import packages.

from pathlib import Path
import os
import pandas as pd
import numpy as np

from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm
from xgboost import XGBRegressor

from IPython.display import display
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import kaleido
from kaggle_tsa.ktsa import *

# -------------------------------------------------------
# | Set up directories.

CWD = Path('/Users/meg/git7/flu/')
DATA_DIR = Path('../input/ts-course-data/')
KAGGLE_DIR = Path('ryanholbrook/ts-course-data/')
IMAGE_DIR = Path('./images')
HTML_DIR = Path('./html')

os.chdir(CWD)
set_cwd(CWD)

# -------------------------------------------------------
# | If the data is not downloaded yet, do so now.

set_data_dir(KAGGLE_DIR, CWD)
show_whole_dataframe(True)

# -------------------------------------------------------
# | Read the data, first as it is.

flu = pd.read_csv(DATA_DIR/'flu-trends.csv')
print(flu.info())
display(flu.head(3))
# -------------------------------------------------------
# | First, let us only deal with `FluVisits`.
# | We would laso like to parse `Week`. We tried `parse_dates=["Week"]`,
# | and `infer_datetime_format=True`, but neither worked out.
# | There might be a simpler way, but this is all what I can
# | think of.

flu = pd.read_csv(DATA_DIR/'flu-trends.csv',
                  dtype={'FluVisits': 'float64'},
                  usecols=['Week', 'FluVisits'])

flu = split_week(flu, append=False)
is_index_continuous(flu, freq='W-MON')

# -------------------------------------------------------
# | Write a function to check the continuity of the index.
# | Raise flag if there are missing/skipped date in the sequence.
# | => Done.
# -------------------------------------------------------
# | Let us take a look at data again.

trace = go.Scatter(x=flu.index,
                   y=flu['FluVisits'])

data = [trace]
layout = go.Layout(height=512,
                   font=dict(size=16),
                   showlegend=False)

fig = go.Figure(data=data, layout=layout)
fig_wrap(fig, IMAGE_DIR/'fig1.png')

# -------------------------------------------------------
# | Create lag plots.

y = flu['FluVisits'].copy()
n_lag = 12
fig, corr = create_lag_plot(y, n_lag=n_lag, n_cols=3)
fig_wrap(fig, IMAGE_DIR/'fig2.png')

_ = [print(f'Lag {i:-2}: {y.autocorr(lag=i):5.3f}')
     for i in range(1, n_lag+1)]

# -------------------------------------------------------
# | Create PACF (Partial Autocorrelation Function) plot.

fig = create_pacf_plot(y, n_lag=n_lag)
fig_wrap(fig, IMAGE_DIR/'fig3.png')

# -------------------------------------------------------
# | Lag 1, 2, 3, and 4 have significant correlation
# | with the target.

# -------------------------------------------------------
# | Construct the time dummy.
# | __Note this trick!__ One can use dictionary `{}`
# | in `pd.concat` to specify the column names and the values
# | in the same time.
y = flu['FluVisits'].copy()
n_lag = 4
X = pd.concat({f'y_lag_{i}': y.shift(i) for i in range(1, n_lag+1)},
              axis=1).fillna(0.0)

# -------------------------------------------------------
# | Start the machine learning part.
# | Note __`shuffle=False`__, so that the test data comes
# | after the training data in the time sequence.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# -------------------------------------------------------
# | Let us see the results.

fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Year]', '[Cases]',
                                    'Flu-visit predictions in 2015-2016 (Lags only)'))

fig_wrap(fig, IMAGE_DIR/'fig4.png')

# -------------------------------------------------------
# | Error evaluation.

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
print(f'Train RMSE : \033[96m{train_rmse:6.2f}\033[0m')

test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test RMSE : \033[96m{test_rmse:6.2f}\033[0m')

# -------------------------------------------------------
# | The test error being smaller than the training error indicates
# | the model is still under fitting. There is a room to improve the
# | forecast.
# |
# | We will see if we can make the error smaller if we include
# | the Google search keywords. First take a look at the data.
# |

search = pd.read_csv(DATA_DIR/'flu-trends.csv')

# | Which words are most often Google-searched?

search.mean(axis=0).sort_values()

# -------------------------------------------------------
# | Let us take the keywords that contains a string 'flu'.

s_cols_week = search.columns[search.columns.str.contains('flu|Week')]
s_cols = search.columns[search.columns.str.contains('flu')]

search = pd.read_csv(DATA_DIR/'flu-trends.csv',
                     dtype={s: 'float64' for s in s_cols},
                     usecols=s_cols_week)

search = split_week(search, append=False)
is_index_continuous(search, freq='W-MON')

# -------------------------------------------------------
# | Create lagged time features.

y = flu['FluVisits'].copy()
n_lag = 4
X_lag = pd.concat({f'y_lag_{i}': y.shift(i) for i in range(1, n_lag+1)},
                  axis=1).fillna(0.0)
X_lag.isna().sum().sum()

# -------------------------------------------------------
# | Create lagged search-words features.

y_search = search[s_cols]
y_search.isna().sum()

X_search = pd.concat({f'y_lag_{i}': y_search.shift(i)
                      for i in range(1, n_lag+1)}, axis=1).fillna(0.0)

X_search.isna().sum().sum()

X = pd.concat([X_lag, X_search], axis=1)
X.isna().sum().sum()

# -------------------------------------------------------
# | Train, predict, and show.

y, X = y.align(X, join='inner', axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)


fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Year]', '[Cases]',
                                    'Flu-visit predictions in 2015-2016 (Google Trends)'))

fig_wrap(fig, IMAGE_DIR/'fig5.png')
# -------------------------------------------------------
# | Error evaluation.

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
print(f'Train RMSE : \033[96m{train_rmse:6.2f}\033[0m')

test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test RMSE : \033[96m{test_rmse:6.2f}\033[0m')

#  -------------------------------------------------------
# | Now the model is not under-fitting, but the forecast is not
# | as good as before. Need trials and errors to select
# | the search words.
# |
# | Overall, the reproduction of the `FluVisits` is excellent.
# |
# | * This is in a sense very much expected though,
# | from the high correlation between `FluVisits` and the data  on the one day before.
# | One can just tell some similar value with yesterday, and
# | the forecast will be quite good.

# -------------------------------------------------------
# | Now we will perform forecasting under more realistic conditions,
# | with a forecast horizon longer than one unit.
# |
# | We first try a simple __Multioutput model__.
# | To output the forecast on the multiple days, what we have to do is only one thing.
# |
# | 1. Make target (=`y`) concatenated for multiple steps in column-wise,
# | ```
# | y_step_1  y_step_2 y_step_3 ...
# | ```

# | Here we perform forecast on a 8-weeks horizon.
# |

y = flu['FluVisits'].copy()
n_step = 8
y = pd.concat({f'y_step_{i}': y.shift(-i) for i in range(0, n_step)},
              axis=1).fillna(0.0)
y.head(3)

# -------------------------------------------------------
# | Do not forget this.

y, X = y.align(X, join='inner', axis=0)
# -------------------------------------------------------
# | Do the training.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Year]', '[Cases]',
                                    'Flu-visit predictions in 2015-2016 (Multi-Output)'))

fig_wrap(fig, IMAGE_DIR/'fig6.png')

# -------------------------------------------------------
# | Use `XGBoost` to perform Direct strategy.

model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Year]', '[Cases]',
                                    'Flu-visit predictions in 2015-2016 (XGBRegressor)'))

fig_wrap(fig, IMAGE_DIR/'fig7.png')

# -------------------------------------------------------
# | Use `XGBoost` to perform DirRec strategy

model = RegressorChain(XGBRegressor())
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Year]', '[Cases]',
                                    'Flu-visit predictions in 2015-2016 (XGBRegressor)'))

fig_wrap(fig, IMAGE_DIR/'fig8.png')

# -------------------------------------------------------
# | END
