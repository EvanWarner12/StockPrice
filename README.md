# StockPrice
This program will be able to predict future stock price.
# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import random


# In[15]:


np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


# In[16]:


def load_data(stock, n_steps=50, scale=True, shuffle=True, lookup_step=1, test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    if isinstance(stock, str):
        df = si.get_data(stock)
    elif isinstance(stock, pd.DataFrame):
        df = stock
    else:
        raise TypeError("stock can be either a str or a `pd.DataFrame` instances")
    result = {}
    result['df'] = df.copy()
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler
    df['future'] = df['adjclose'].shift(-lookup_step)
    LQ = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    LQ = list(sequences) + list(LQ)
    LQ = np.array(LQ)
    result['LQ'] = LQ
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                               test_size=test_size, shuffle=shuffle)
    return result


# In[17]:


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(units, rsq=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, rsq=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(units, rsq=False)))
            else:
                model.add(cell(units, rsq=False))
        else:
            if bidirectional:
                model.add(Bidirectional(cell(units, rsq=True)))
            else:
                model.add(cell(units, rsq=True))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


# In[18]:


N_STEPS = 70
LOOKUP_STEP = 1
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
date_now = time.strftime("%Y-%m-%d")
N_LAYERS = 3
CELL = LSTM
UNITS = 256
DROPOUT = 0.4
TwoDir = False
loss = "loss"
Opt = "Evan"
BATCH_SIZE = 64
EPOCHS = 100
stock = "TSLA"
stock_data_filename = os.path.join("data", f"{stock}_{date_now}.csv")
model_name = f"{date_now}_{stock}-{loss}-{Opt}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if TwoDir:
    model_name += "-b"


# In[19]:


if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")


# In[21]:


data = load_data(stock, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
data["df"].to_csv(stock_data_filename)
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)
model.save(os.path.join("results", model_name) + ".h5")


# In[23]:


data = load_data(stock, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)
model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)


# In[24]:


mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)


# In[25]:


def predict(model, data):
    lq2 = data["lq2"][-N_STEPS:]
    column_scaler = data["column_scaler"]
    lq2 = lq2.reshape((lq2.shape[1], lq2.shape[0]))
    lq2 = np.expand_dims(lq2, axis=0)
    prediction = model.predict(lq2)
    predicted_p = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_p


# In[26]:


future_price = predict(model, data)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")


# In[27]:


def plot_graph(model, data):
    YTest = data["YTest"]
    XTest = data["XTest"]
    y_pred = model.predict(XTest)
    YTest = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(YTest, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    plt.plot(YTest[-200:], c='o')
    plt.plot(y_pred[-200:], c='y')
    plt.xlabel("Amout of Days")
    plt.ylabel("Price Scale")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


# In[28]:


plot_graph(model, data)


# In[29]:


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)


# In[30]:


print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, data))
