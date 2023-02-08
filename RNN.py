import numpy as np
import optuna as opt
import pandas as pd
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import MetaTrader5 as mt5
import random
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
    rates = mt5.copy_rates_from_pos(InpSymbol, timeframe, start_pos, count)
    df = pd.DataFrame(rates)
    data = pd.DataFrame()
    #df = self.Download(x.name, mt5.TIMEFRAME_M1, 0, row)
    data["Date"]  = df["time"].apply(lambda x: dt.datetime.fromtimestamp(x))
    data['Open'] = df["open"]
    data['High'] = df["high"]
    data['Low'] = df["low"]
    data['Close'] = df["close"]

    return data



def ModelRNN(InpSymbol="U30USD.HKT"):
    


    corr = pd.read_csv(f"Data/{InpSymbol}_Correlation.csv", decimal=",", sep=";")
    print(corr)

    correlated_symbols = corr["Symbol"].values[:30]

    mt5.initialize()
    mt5_symbols = [x.name for x in mt5.symbols_get()]
    big_data = pd.DataFrame()
    data = Mt5_Download(InpSymbol, mt5.TIMEFRAME_M1, 0, 14400)

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
            data =  Mt5_Download(symbol, mt5.TIMEFRAME_M1, 0, 14400)
            big_data[f"{symbol}_pct_change"] = data["Close"].pct_change()*1000 
        else:
            data = Request_OHLCV(symbol, 60, 14400) 
            delta_data = Request_last_trades(symbol, 14400)
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
    y_test = Y[int(Y.shape[0] * (train + val)):]


    idx = np.where(X_train=='')
    X_train[idx] = 0
    idx = np.where(X_val=='')
    X_val[idx] = 0
    idx = np.where(X_test=='')
    X_test[idx] = 0
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = Y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = Y_val.astype(np.float32)
    # Reshape the input data to be 3D
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Create target function which will be optimized by Optuna find the best hyperparameters
    def get_model_result(trial, eval=False):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='model.h5',
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
        model.add(tf.keras.layers.SimpleRNN(trial.suggest_int('lstm', 3, 32), input_shape=(None, 1)))
        for dense_layers in range(3):
            model.add(tf.keras.layers.Dense(trial.suggest_int(f'dense_{dense_layers}', 2, 32), activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=30, batch_size=32, callbacks=[model_checkpoint_callback, early_stopping_callback])
        
        if eval:
            return model

        return model.history.history['val_loss'][-1]

    # Optimize the model
    study = opt.create_study(direction='minimize')
    study.optimize(get_model_result, n_trials=5)
    if True:
        model = get_model_result(study.best_trial, eval=True)
        import statistics
        prediction = model.predict(X_test)
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
    
if __name__ == "__main__":
    ModelRNN()