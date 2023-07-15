import random
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint



SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future', axis=1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buy = []
    sell = []

    for seq, target in sequential_data:
        if target == 0:
            sell.append([seq, target])
        elif target ==1:
            buy.append([seq,target])

    random.shuffle(buy)
    random.shuffle(sell)

    lower = min(len(buy), len(sell))

    buy = buy[:lower]
    sell = sell[:lower]

    sequential_data = buy + sell

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq,target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y



main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for ratio in ratios:
    dataset = f"C:/Users/USER/Desktop/Resources from IM/RNNCrypto_data/crypto_data/crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

times = sorted(main_df.index.values)
last_5pc = times[-int(0.05 * len(times))]

validation_main_df = main_df[(main_df.index >= last_5pc)]
main_df = main_df[(main_df.index < last_5pc)]


train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train_data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont Buy: {train_y.count(0)}, buy: {train_y.count(1)}")
print(f"Validation Dont Buy: {validation_y.count(0)}, buy: {validation_y.count(1)}")

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
filepath = "RNN_Final-{epoch:02d}-{val_acc: .3f}" # unique file name that will include the epoch and the validation accuracy for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor= 'val_acc', verbose= 1, save_best_only=True, mode='max')) # Saves only the best ones

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x,validation_y),
                    callbacks=[tensorboard, checkpoint])





























