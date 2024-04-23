# %% [markdown]
# # Forecasting Time Series data
# 
# * Idea and some code taken from, and also OPSD time series data set explained here: https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
# * Some code and approaches from: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# * Nice tutorial: https://www.tensorflow.org/beta/tutorials/text/time_series

# %%
%matplotlib inline

import matplotlib.pyplot as plt
# plt.xkcd()
# plt.style.use('ggplot')
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20, 8)

# %%
import pandas as pd
import numpy as np

# %%
# for local
# url = 'opsd_germany_daily.csv'

# for colab
url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'

time_series_df = pd.read_csv(url,
                             sep=',',
                             index_col=0, # you can use the date as the index for pandas
                             parse_dates=[0]) # where is the time stamp?

# %%
cols_plot = ['Consumption', 'Solar', 'Wind']

axes = time_series_df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')

# %%
axes = time_series_df.plot(marker='.', alpha=0.5, linestyle='None')
plt.ylabel('Daily Totals (GWh)');

# %% [markdown]
# ## Can we predict each day's consumption from its past?
# 
# ### We train on the years 2006 - 2016 and validate on 2017

# %%
consumption = time_series_df['Consumption'].to_numpy()
consumption.shape

# %%
plt.title('Power Consumption from 2006 to 2017')
plt.plot(consumption);

# %% [markdown]
# ## Statistical Methods directly geared towards forecasting
# 
# _Statistical Methods are often favorable: https://twitter.com/togelius/status/1173272424177119233_
# 
# * https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
# * https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b#targetText=Time%20series%20forecasting%20is%20the,forecasting%20retail%20sales%20time%20series.
# * https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
# * https://research.fb.com/prophet-forecasting-at-scale/
# 
# https://www.statsmodels.org
# * https://www.statsmodels.org/stable/examples/index.html#stats
# * https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html
# * https://www.statsmodels.org/stable/tsa.html
# 
# 

# %%
import statsmodels.api as sm

decomposition = sm.tsa.seasonal_decompose(time_series_df['2017']['Consumption'], model='additive')
decomposition.plot();

# %%
# derived from here: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# %% [markdown]
# ### How much of the past would we like for each individual prediction?

# %%
#@title Prediction from n past days

# https://colab.research.google.com/notebooks/forms.ipynb

n_steps_in = 30 #@param {type:"slider", min:1, max:100, step:1}
n_steps_out = 1

# %%
X, Y = split_sequence(consumption, n_steps_in, n_steps_out)
X.shape, Y.shape

# %%
X[0], Y[0]

# %%
# reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
X.shape

# %%
# we do a special test / train split,
# we see how well we can predict 2017 as test/validation


X_train = X[:-365]
Y_train = Y[:-365]

X_test = X[-365:]
Y_test = Y[-365:]

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

# %% [markdown]
# ## Baslines

# %% [markdown]
# ### R2 Metric: MSE and MAE are not speaking
# 
# * R^2 score, the closer to 1 the better
# * loosely speaking: how much better is this than predicting the constant mean
# * 0 would mean just as good
# * 1 is perfect
# * neg. would mean even worse
# * it can become arbitrarily worse
# 
# https://en.wikipedia.org/wiki/Coefficient_of_determination
# 

# %% [markdown]
# ### Rolling mean baseline

# %%
def rolling_mean_baseline(history):
  return np.mean(history, axis=1)

# %%
Y_pred_baseline_rolling = rolling_mean_baseline(X)
Y_pred_baseline_rolling.shape

# %%
# this gets really crowded, using a stride, makes it more readable
stride = 5


plt.plot(consumption[n_steps_in::stride], alpha=0.5, color='g')
plt.plot(Y_pred_baseline_rolling[::stride], color='r', ls='dashed')

plt.title('Prediction using the rolling mean');


# %%
from sklearn.metrics import r2_score

r2_score(Y, Y_pred_baseline_rolling)

# %% [markdown]
# ### Baseline: Previous Value

# %%
def previous_value_baseline(history):
  return history[:, -1]

# %%
Y_pred_baseline_previous_value = previous_value_baseline(X)
Y_pred_baseline_previous_value.shape

# %%
# this gets really crowded, using a stride, makes it more readable
stride = 5


plt.plot(consumption[n_steps_in::stride], alpha=0.5, color='g')
plt.plot(Y_pred_baseline_previous_value[::stride], color='r', ls='dashed')
plt.title('Prediction using the previous value');

# %% [markdown]
# ### This looks good at first sight, but wait for the R2 score
# 
# It is (obviously) always off by one

# %%
from sklearn.metrics import r2_score

r2_score(Y, Y_pred_baseline_previous_value)

# %%
# this gets really crowded, using a stride, makes it more readable
stride = 1
# too crwoded, zoom in
window_start = 300
window_end = 600

plt.plot(consumption[n_steps_in+window_start:n_steps_in+window_end:stride], alpha=0.5, color='g')
plt.plot(Y_pred_baseline_previous_value[window_start:window_end:stride], color='r', ls='dashed')

plt.title('Prediction using the previous value, zooming in reveals its issue');

# %% [markdown]
# ## Prediction using RNNs and TensorFlow
# 
# ### Can we beat an r2 score around .20?

# %%
# Gives us a well defined version of tensorflow

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

# %%
import tensorflow as tf
print(tf.__version__)

# %% [markdown]
# ### Just for the reference, what GPU are we running on?

# %%
# What kind of GPU are we running on
!nvidia-smi

# %%
# adapted from https://stackoverflow.com/a/42351397/1756489 and ported to TF 2
# https://keras.io/metrics/#custom-metrics

# only works on tensors while training, use sklearn version when using on numpy arrays

def r2_metric(y_true, y_pred):
  total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
  unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
  R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

  return R_squared

# %% [markdown]
# ### Training our model

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model

model = Sequential()

# two layer model, known to work well
# model.add(GRU(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# model.add(GRU(100, activation='relu'))

# trains faster, but also works
# model.add(GRU(256, activation='relu', input_shape=(n_steps_in, n_features)))

# one layer SimpleRNN seems to be enough for this data set
model.add(SimpleRNN(256, activation='relu', input_shape=(n_steps_in, n_features)))

# horrible results
# model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
# model.add(LSTM(64, activation='relu'))

# optional regularization
# model.add(BatchNormalization())
# model.add(Dropout(0.2))


# combines final outputs from RNN into continous output
model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse', metrics=[r2_metric])

model.summary()

# %%
%%time

batch_size = 32
epochs=25

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, Y_test),
                    verbose=1)

# %%
plt.yscale('log')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Log Loss over Epochs')

plt.legend(['loss', 'validation loss']);

# %%
plt.yscale('log')
plt.plot(history.history['r2_metric'])
plt.plot(history.history['val_r2_metric'])
plt.title('R^2 over Epochs')

plt.legend(['r2', 'validation r2']);

# %%
model.evaluate(X, Y, batch_size=batch_size, verbose=0)

# %% [markdown]
# ### R2 for all data

# %%
from sklearn.metrics import r2_score

Y_pred = model.predict(X)
r2_score(Y, Y_pred)

# %% [markdown]
# ### R2 for training data

# %%
Y_train_pred = model.predict(X_train)
r2_score(Y_train, Y_train_pred)

# %% [markdown]
# ### R2 for validation data

# %%
Y_test_pred = model.predict(X_test)
r2_score(Y_test, Y_test_pred)

# %% [markdown]
# ## Let's plot predicted and true over each other

# %%
# this gets really crowded, using a stride, makes it more readable
stride = 10

# for the first 90 days we have no prediction, so get rid of them
plt.plot(consumption[n_steps_in::stride], alpha=0.5, color='g')
plt.plot(Y_pred[::stride], color='r', ls='dashed')

plt.title('All data, train and test combined, stride 10');


# %%
stride = 1


plt.plot(consumption[-2 * 365:-365:stride], alpha=0.5, color='g')
plt.plot(Y_train_pred[-365::stride], color='r', ls='dashed')

plt.title('Trained data, 2016 only, no stride');


# %%
stride = 1


plt.plot(consumption[-365::stride], alpha=0.5, color='g')
plt.plot(Y_test_pred[::stride], color='r', ls='dashed')

plt.title('Test data, 2017, no stride');


# %% [markdown]
# ## Observation / Wrap-Up
# 
# ### overall a pretty good result
# * r2 score is pretty promising
# * defintely improved over baseline
# * no domain knowledge necessary
# 
# ### no dramatic difference between training and test data
# * neither in metrices
# * nor in visual inspection
# 
# ### peaks and valleys are underestimated very often
# * but often the most interesting part
# * this seems to be a common problem in time series prediction
# * it gets better we train the model for longer
# * also when we increase its capacity
# * on the other hand this might counter regularization
# 

# %% [markdown]
# ## Next Steps
#   
# ### make use of seasonal decomposition
# * predict each of the components
#   * trend should be pretty easy
#   * residual shows clear spices on national holidays
#   * remove them
#   * the results might be noise only
#   * but maybe it is not, train a model on it to check
#   
# ### have a more powerful statistical baseline
# * Use more powerful statistical models
#   * https://www.statsmodels.org/stable/tsa.html
#   * http://www.statsmodels.org/dev/tsa.html
#   * http://www.statsmodels.org/dev/vector_ar.html
# * MSBVAR (Markov-Switching, Bayesian, Vector Autoregression Models) might beat our model, but
#   * needs a lot of modelling
#   * does not seem readily available in the Python world (in R it would be)
#   
# 

# %%



