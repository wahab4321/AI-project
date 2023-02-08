import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna as opt
import random
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import MetaTrader5 as mt5

from IQFeed_API import *


# ATR function
def wwma(values, n):
        return values.ewm(alpha=1/n, adjust=False).mean()
def atr(df, n=14):
            data = df.copy()
            high = data['High']
            low = data['Low']
            close = data['Close']
            data['tr0'] = abs(high - low)
            data['tr1'] = abs(high - close.shift())
            data['tr2'] = abs(low - close.shift())
            tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
            atr = wwma(tr, n)
            return atr

def Mt5_Download(InpSymbol, timeframe, start_pos, count):
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    rates = mt5.copy_rates_from_pos(InpSymbol, timeframe, start_pos, count)
    df = pd.DataFrame(rates)
    data = pd.DataFrame()
    
    #df = self.Download(x.name, mt5.TIMEFRAME_M1, 0, row)
    try:
        data["Date"]  = df["time"].apply(lambda x: dt.datetime.fromtimestamp(x))
    except KeyError:
        print("Error : ", mt5.last_error())
        return
    data['Open'] = df["open"]
    data['High'] = df["high"]
    data['Low'] = df["low"]
    data['Close'] = df["close"]

    return data



def ModelANN(InpSymbol="U30USD.HKT"):
    


    corr = pd.read_csv(f"Data/{InpSymbol}_Correlation.csv", decimal=",", sep=";")
    print(corr)

    correlated_symbols = corr["Symbol"].values[:30]
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    mt5_symbols = [x.name for x in mt5.symbols_get()]
    big_data = pd.DataFrame()
    data = Mt5_Download(InpSymbol, mt5.TIMEFRAME_M1, 0, 28800)

    x = np.array(data["Close"].to_list())
    peaks, _ = find_peaks(x, height=0)
    Npeaks, _ = find_peaks(-x)
    data["TP"] = abs(atr(data,300).shift(-14).pct_change()*100)
    data["SL"] = abs(atr(data,28).shift(-28).pct_change()*100)
    
    data = data.iloc[:-28, :]
    #data.fillna(random.random(), inplace=True)
    data.dropna(axis=0, inplace=True)       
    Y = data.iloc[1:,-2:].values
    big_data = data
    print(Y)
    

    for i, symbol in enumerate(correlated_symbols):
        #rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 10000, 10000)
        print(data)
        if symbol in mt5_symbols:
            data =  Mt5_Download(symbol, mt5.TIMEFRAME_M1, 0, 28800)
            big_data[f"{symbol}_pct_change"] = data["Close"].pct_change()*1000 
        else:
            data = Request_OHLCV(symbol, 60, 28800) 
            delta_data = Request_last_trades(symbol, 28800)
            big_data[f"{symbol}_pct_change"] = data["Close"].pct_change()*1000                   
            big_data[f"{symbol}_Volume"] = delta_data["Last Size"]
        print('Symbol >> ' + symbol)
        
        



    big_data.to_csv('BigData.csv', sep=";", decimal=",")
    big_data.fillna(random.random(), inplace=True)



    print(big_data)


    X = big_data.iloc[1:,1:].values 
    y_test_dates = big_data.iloc[1:, :1]
    train = 0.6
    val = 0.2
    test = 0.2

    X_train = X[:int(X.shape[0] * train)]



    X_val = X[int(X.shape[0] * train):int(X.shape[0] * (train + val))]
    X_test = X[int(X.shape[0] * (train + val)):]

    Y_train = Y[:int(Y.shape[0] * train)]
    Y_val = Y[int(Y.shape[0] * train):int(Y.shape[0] * (train + val))]
    Y_test = Y[int(Y.shape[0] * (train + val)):]

    
    

    idx = np.where(X_train=='')
    X_train[idx] = 0
    idx = np.where(X_val=='')
    X_val[idx] = 0
    idx = np.where(X_test=='')
    X_test[idx] = 0
    
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    def get_model_result(trial, eval=False):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='ANNmodel.h5',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True))

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(X_train.shape[1], activation='sigmoid', input_shape=(X_train.shape[1],)))
        for dense_layers in range(5):
            model.add(tf.keras.layers.Dense(trial.suggest_int(f'dense_{dense_layers}', X_train.shape[1], X_train.shape[1]*2), activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=30, batch_size=32, callbacks=[model_checkpoint_callback, early_stopping_callback])
        
        if eval:
            return model

        return model.history.history['val_loss'][-1]

    study = opt.create_study(direction='minimize')
    study.optimize(get_model_result, n_trials=5)
    if True:
        ann = get_model_result(study.best_trial, eval=True)
        import statistics
        prediction = ann.predict(X_test)
        
        prediction[prediction == 1] = [random.uniform(0, 1) for _ in range(np.count_nonzero(prediction == 1))]
        prediction[prediction == 0] = [random.uniform(0, 1) for _ in range(np.count_nonzero(prediction == 0))]
        df_predictions = pd.DataFrame(prediction, columns=["TP", "SL"])
        
        y_test_dates.index =pd.to_datetime(y_test_dates['Date'],format='%Y-%m-%d %H:%M:%S')
        
        y_test_dates = y_test_dates.resample('1D').mean()
        y_test_dates["Date"] = y_test_dates.index 
        print("Number of days >>>> ", len(y_test_dates), len(prediction))
        
        print(y_test_dates) 
        if __name__ == "__main__":
            plt.plot(prediction)
            print(prediction)
            plt.show()
            print(data)

            plt.plot(x)
            plt.plot(peaks, [x[i] for i in peaks], "x") 
            plt.plot(Npeaks, [x[i] for i in Npeaks], "x")
            plt.show()
        else:
            return y_test_dates
    else:
        return False


if __name__=="__main__":
    ModelANN()