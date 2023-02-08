import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import time
from dateutil.relativedelta import  relativedelta
import seaborn as sns
import os
from pathlib import Path
from pandas.plotting import register_matplotlib_converters
from scipy.stats import percentileofscore as score
import datetime as dt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from Toaster import QToaster

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import shutil


def ClearDir(folder):
    for filename in os.listdir(folder):
        file_dir = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_dir) or os.path.islink(file_dir):
                os.unlink(file_dir)
            elif os.path.isdir(file_dir):
                shutil.rmtree(file_dir)
        except Exception as e:
            print('Failed to delete')


def ShowToast(msg : str):
    app1 = QApplication(sys.argv)
    corner = Qt.Corner(3)
    QToaster.showMessage(
                None, msg, corner=corner, desktop=True)
    app1.exec()

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=UnicodeWarning)
warnings.filterwarnings("ignore", category=BytesWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

register_matplotlib_converters()

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed, error code : ", mt5.last_error())
    exit()

terminal_info = mt5.terminal_info()
print(terminal_info)
#ShowToast(f"Connected to MT5 with account Name : {terminal_info.name}" )

def Loop(exp):
    done = False
    res =None
    while not done:
        try:
            res = exp
            done=True
        except:
            pass

    return res

def get_symbol_rates(symbol, pos):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1,0, pos)
    df = pd.DataFrame(rates) 
    return df



#######################################
##         Correlation Script        ##
#######################################
def Update_Correlation(row=1000,gl_dict=None):
    method = "open" # You can choose close as well
    big_df = pd.DataFrame()
    min_rows = 999999999999999999
    for i, symbol in enumerate(symbols_list):
        df = pd.read_csv("Data/"+symbol+".csv", decimal=",", sep=";").iloc[row-1000:row,:]
        if df.size >0:
            df[symbol] = df[method]
            big_df = pd.concat([big_df, df[symbol]],axis=1)
            if df.shape[0]<min_rows:
                min_rows = df.shape[0]

    big_df = big_df.corr("pearson")
    # print(big_df)
    # print("HeatMap Correlation :",big_df)
    symbols_list_df = pd.DataFrame()
    symbols_list_df['Symbol'] = symbols_list
    symbols_list_df = symbols_list_df.T
    # print(symbols_list)
    df = symbols_list_df.to_numpy(dtype=np.str_,copy=True)

    df = np.array(symbols_list, dtype=np.str_, copy=True)

    saved = False
    while not saved:
        try:
            # df.tofile(mt5.terminal_info().commondata_path.replace("\\", "/")+f"/Files/Data/symbols_list.txt")
            with open(mt5.terminal_info().commondata_path.replace("\\", "/")+f"/Files/Data/symbols_list.txt", 'w') as file:
                file.write('\n'.join(symbols_list))
            saved = True
            # print("Symbols list saved")
        except :
            saved = False

    
    start = time.time()
    chart_symbol_df = None
    for symbol in symbols_list:
        try:
            symbol_corr_df = pd.DataFrame(big_df[symbol])
            symbol_corr_df['Symbol'] = symbol_corr_df.index
            symbol_corr_df.index = range(symbol_corr_df.shape[0])
            symbol_corr_df = symbol_corr_df.dropna(axis=0)
            symbol_corr_df = symbol_corr_df.sort_values(symbol, ascending=False)

            Correlation_matrix = pd.concat([symbol_corr_df.head(75),symbol_corr_df.tail(75)], axis=0)
            Correlation_matrix['Direction'] = Correlation_matrix[symbol] / Correlation_matrix[symbol].abs()
            Correlation_matrix[symbol] = Correlation_matrix[symbol].abs()
            Correlation_matrix = Correlation_matrix.sort_values(symbol, ascending=False)
            Correlation_matrix.index = range(Correlation_matrix.shape[0])
            
            if gl_dict:
                Correlation_matrix['O-C %'] = Correlation_matrix['Symbol'].apply(lambda x: gl_dict[x][0] if x in  gl_dict else 0)
                Correlation_matrix['H-L %'] = Correlation_matrix['Symbol'].apply(lambda x: gl_dict[x][1] if x in  gl_dict else 0)
                Correlation_matrix['Chng Direction'] =  Correlation_matrix['O-C %'] /  Correlation_matrix['O-C %'].abs() 
                Correlation_matrix['O-C %'] =  Correlation_matrix['O-C %'].abs()

            # print(Correlation_matrix)
            Correlation_matrix = Correlation_matrix.sort_values([symbol, "O-C %", "H-L %"], ascending=False)
            
            if symbol == chart_symbol:
                chart_symbol_df = Correlation_matrix["Symbol"]

            # print(Correlation_matrix)
            #print(len(symbols_list))
            Correlation_matrix['Symbol'] = [symbols_list.index(x) for x in list(Correlation_matrix['Symbol'])]
            # print(Correlation_matrix)
            
            Correlation_matrix = Correlation_matrix.to_numpy(dtype=np.float64)

            saved = False

            while not saved:
                try:
                    Correlation_matrix.tofile(mt5.terminal_info().commondata_path.replace("\\", "/")+f"/Files/Data/{symbol}_Correlation.bin")
                    
                    saved = True
                    
                except :
                    saved = False


            # print(symbol, df[symbol].idxmax(), df[symbol].max())
            # print(symbol, df[symbol].idxmin(), df[symbol].min())
        except:
            pass
    return big_df, chart_symbol_df

    # plt.figure(figsize=(11,8))
    # sns.heatmap(big_df, cmap="Greens")
    # plt.show()

#######################################
##         Calendar Script           ##
#######################################
def Update_Calendar(row=1000,path=terminal_info.data_path.replace("\\", "/")+"/MQL5/Files"):
    list_files = os.listdir(path+'/Data')
    done = False
    while not done:
        try:
            df = pd.read_csv(path+"/Data/Countries.csv", encoding="utf-16", sep='\t')
            done = True
        except :
            pass
        

    list_countries = list(df['name'])
    list_countries_codes = list(df['id'])
    list_countries_codes[list_countries_codes.index(0)] = 1000
    currency_dict = {key:value for key, value in zip(list(df['name']), list(df['currency']))}
    currency_list = list(dict.fromkeys(list(df['currency'])))
    events_dict = {}
    big_df = pd.DataFrame()
    for symbol in currency_list:
        done = False
        while not done:
            try:
                df = pd.read_csv(path+"/Data/Events_"+symbol+".csv", encoding="utf-16", sep="\t")
                done = True
            except :
             pass
        big_df = pd.concat([big_df, df],axis=0)
        # list_events_ids = list(df['id'])
        # list_event_names = list(df['name'])
        # list_event_importance = list(df['importance'])
        # symbol_dict = {key:(name, importance) for key, (name, importance) in zip(list_events_ids, zip(list_event_names, list_event_importance))}
        # events_dict.update(symbol_dict)
    big_df.index = range(big_df.shape[0])

    def Insert_symbols(row):
        try:
            event_id = row['event_id']
            try:
                country = list_countries[list_countries_codes.index(int(str(event_id)[:4]))]
            except:
                try:
                    country = list_countries[list_countries_codes.index(int(str(event_id)[:3]))]
                except:
                    country = list_countries[list_countries_codes.index(int(str(event_id)[:2]))]

            row["Country"] = country
            row['Currency'] = currency_dict[country]
            row['Event_name'] = list(big_df['name'][ big_df['id'] == int(event_id)])[0] #events_dict[event_id][0]
            row['Event_importance'] = list(big_df['importance'][ big_df['id'] == int(event_id)])[0]#events_dict[event_id][1]
        except :
            pass
        return row

    success = False
    while not success:
        try:
            df = pd.read_csv(path+'/Data/All_Events_Values.csv',encoding="utf-16", sep="\t")
            success = True
        except:
            success = False
        
    df.drop(df.index[-1], inplace=True)
    df['Country'] = range(df.shape[0])
    df['Currency'] = range(df.shape[0])
    df['Event_name'] = range(df.shape[0])
    df['Event_importance'] = range(df.shape[0])

    df = df.apply(lambda row: Insert_symbols(row),axis=1)
    return df


#######################################
##     Gainers-Losers Script         ##
#######################################

def Update_GL(row=1000):
    date_from = datetime.now()-relativedelta(days=20)
    date_to = datetime.now()
    big_df = pd.DataFrame()
    s = 0
    for i, symbol in enumerate(symbols_list):
        # df = pd.DataFrame()      
        # rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, date_from, date_to)
        # df = pd.DataFrame(rates)
        df = pd.read_csv("Data/"+symbol+".csv", decimal=",", sep=";").iloc[row-1000:row,:]
        df["O-C %"] = (df['close'] - df['open']) * 100 / df['open']
        df["H-L %"] = (df['high'] - df['low']) * 100 / df['low']
        df = df.iloc[-1:, :]
        df['Symbol'] = symbol
        df = df[['Symbol','O-C %', 'H-L %']]
        big_df =  pd.concat([big_df, df], axis=0)
        big_df.index = range(big_df.shape[0])
        s+=1
    gl_dict = dict(zip(big_df.Symbol, zip(big_df['O-C %'], big_df['H-L %'])))

    return gl_dict


#######################################
##  ATR / STD + AI Script            ##
#######################################

timeframe = "trade"

# ml classification method:
def computeClassification(actual):
    if (actual > 0):
        return 1
    else:
        return -1

def Update_AI(corr_df, symbols,row=1000, Tester=False):
    if True:
        frames_prices = []
        for i, symbol in enumerate(symbols):
            # rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, date_from, date_to)
            # df = pd.DataFrame(rates)
            df = pd.read_csv("Data/"+symbol+".csv", decimal=",", sep=";").iloc[row-1000:row,:]
            df['pct_change'] = df['open'].pct_change()
            # print("Data Received for:", symbol)
            price = df['open']
            # print(price )
            frames_prices.append(price)

        # print("ALL PRICES:")
        frames_prices = pd.DataFrame(frames_prices)
        frames_prices = frames_prices.T

        frames_prices.columns = symbols

        # print(frames_prices)
        ####################################################################################################################
        dc = []

        for i, col in enumerate(list(dict.fromkeys(list(frames_prices.columns)))):
            prices = frames_prices.iloc[:,i:i+1]
            prices = prices.loc[:,~prices.columns.duplicated()].copy()
            # print(prices)
            dc.append(float(prices.pct_change().sum()))

        momentum = pd.DataFrame()
        momentum['symbol'] = list(dict.fromkeys(list(frames_prices.columns)))
        momentum['day_change'] = dc
        # print(dc)
        momentum['momentum'] = range(momentum.shape[0])
        for i in range(len(momentum)):
            # EXAMPLE USAGE OF SCIPY MOMENTUM:
            # A percentileofscore of, for example, 80% means that 80% of the scores in a are below the given score.
            # In the case of gaps or ties, the exact definition depends on the optional keyword, kind.
            # So when you supply the value 73.94, there are 5 elements of df that fall below that score, and 5/6
            # gives you your 83.3333% result.
            momentum.loc[i, 'momentum'] = score(momentum['day_change'].to_list(), momentum.loc[i, 'day_change']) / 100

        momentum['momentum'] = momentum['momentum'].astype(float)
        # print("Momentum Scores:")
        # print(momentum)

        # sort the stocks based on the momentum value we calculated and
        # find the top n names having the highest momentum:
        # use n largest function to find top n names:
        top_picks = momentum.nlargest(19, 'momentum')['symbol'].reset_index().drop('index', axis=1)  # take top 18 names
        # print("Top Momentum Plays for the Day:")
        top_picks = top_picks.head(len(top_picks) - 1)
        # print(top_picks)

        # send to a list:
        top_picks_list = top_picks.values.tolist()
        top_picks_list = [chart_symbol] + [val for sublist in top_picks_list for val in sublist] 
        # print("TOP MOMENTUM PAIRS TO TRADE FROM:")
        # print(top_picks_list)
        time.sleep(2)


        # ATR function
        def wwma(values, n):
            return values.ewm(alpha=1/n, adjust=False).mean()

        def atr(df, n=14):
            data = df.copy()
            high = data['high']
            low = data['low']
            close = data['close']
            data['tr0'] = abs(high - low)
            data['tr1'] = abs(high - close.shift())
            data['tr2'] = abs(low - close.shift())
            tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
            atr = wwma(tr, n)
            return atr


        ################################################################################################################################################
        # ENTER TRADE LOGIC

        average_score = 0
        average_prediction = 0

        chart_symbol_df = None
        chart_symbol_latest_prediction = None

        for symbol in top_picks_list:

            # df = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1000)
            # df = pd.DataFrame(df)
            df = pd.read_csv("Data/"+symbol+".csv", decimal=",", sep=";").iloc[row-1000:row,:]
            df['pct_change'] = df['open'].pct_change()
            # print("Data Received for:", symbol)
            

            bb_period = 75
            # df['sma'] = df['open'].rolling(window=bb_period).mean()
            if momentum_indicator == "atr":
                df['atr'] = atr(df, bb_period)

            elif momentum_indicator == "std_deviation":
                df[momentum_indicator] = df['open'].rolling(bb_period).std()

            df[f'upper_{momentum_indicator}'] = df['open'] + (df[momentum_indicator] * num_of_std_deviations)
            df[f'lower_{momentum_indicator}'] = df['open'] - (df[momentum_indicator] * num_of_std_deviations)
            df[f'average_{momentum_indicator}'] = (df[f'upper_{momentum_indicator}'] + df[f'lower_{momentum_indicator}']) / 2
            df[f'upper_{momentum_indicator}_target_profit'] = df['open'] + (df[momentum_indicator] * num_of_std_deviations_profit)
            df[f'lower_{momentum_indicator}_target_profit'] = df['open'] - (df[momentum_indicator] * num_of_std_deviations_profit)
            
            ############################################################################################################
            # ADD ML LOGIC:
            df['returns'] = np.log(df['open'] / df['open'].shift(1))
            df['returns'].fillna(0)
            df['returns_1'] = df['returns'].fillna(0)
            df['returns_2'] = df['returns_1'].replace([np.inf, -np.inf], np.nan)
            df['returns_final'] = df['returns_2'].fillna(0)
            # # print(df['returns_final'])  # we apply the defined classifier above, being computeClassification, to determine whether this will be 1 or -1 based on % move up or % move down on the daily return

            # Compute the last column (Y) -1 = down, 1 = up by applying the defined classifier above to the 'returns_final' dataframe
            df.iloc[:, len(df.columns) - 1] = df.iloc[:, len(df.columns) - 1].apply(computeClassification)
            # print(symbol)
            if symbol == chart_symbol:
                chart_symbol_df = df
            # Now that we have a complete dataset with a predictable value, the last colum “Return” which is either -1 or 1, create the train and test dataset.
            # convert float to int so you can slice the dataframe
            testData = df[-int((len(df) * 0.20)):]  # 2nd half is forward tested on
            trainData = df[:-int((len(df) * 0.80))]  # 1st half is trained on

            # replace all inf with nan
            testData_1 = testData.replace([np.inf, -np.inf], np.nan)
            trainData_1 = trainData.replace([np.inf, -np.inf], np.nan)
            # replace all nans with 0
            testData_2 = testData_1.fillna(0)
            trainData_2 = trainData_1.fillna(0)

            data_X_train = trainData_2.iloc[:, 0:len(trainData_2.columns) - 1]
            # print(data_X_train)
            # Y is the 1 or -1 value to be predicted (as we added this for the last column above using the apply.(computeClassification) function
            data_Y_train = trainData_2.iloc[:, len(trainData_2.columns) - 1]
            
            # Same thing for the test dataset
            data_X_test = testData_2.iloc[:, 0:len(testData_2.columns) - 1]
            data_Y_test = testData_2.iloc[:, len(testData_2.columns) - 1]


            # logistic regression with adaboost and bagging classifier
            logisticregression = LogisticRegression()
            ada = AdaBoostClassifier(base_estimator=logisticregression, n_estimators=100, learning_rate=0.5,random_state=42)  # learning rate is a regularization parameter (avoid overfitting), used to minimize loss function, increasing test accuracy.
            clf = BaggingClassifier(base_estimator=ada)  # learning rate is the contribution of each model to the weights and defaults to 1. Reducing this rate means the weights will be increased or decreased to a small degree, forcing the model to train slower (but sometimes resulting in better performance).
            # n estimators id maximum number of estimators (or models) at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early. It is the maximum number of models to iteratively train. Defaults to 50.
            # random state default = None.  If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
            successful_fits = 0
            try:
                clf.fit(data_X_train, data_Y_train)
                predictions = clf.predict(
                data_X_test)  # predictions is an array containing the predicted values (-1 or 1) for the features in data_X_test
                predictions_dataframe = pd.DataFrame(predictions)  # convert array to dataframe

                from sklearn.metrics import accuracy_score

                y_predictions = clf.predict(data_X_test)  # predict y based on x_test
                
                print(symbol, y_predictions)
                #print("Accuracy Score Employing Machine Learning: " + str(accuracy_score(data_Y_test, y_predictions)))

                accuracy_score1 = accuracy_score(data_Y_test, y_predictions)
                latest_prediction = predictions_dataframe.values[-1:]
                if symbol == chart_symbol:
                    chart_symbol_latest_prediction = latest_prediction
                latest_prediction = int(latest_prediction)
                # print("Latest Prediction for:", symbol)
                # print(latest_prediction)
                # print("Test Accuracy Score for:", symbol)
                # print(accuracy_score1)
                average_prediction += accuracy_score1 * latest_prediction * float(corr_df[chart_symbol].iloc[symbols_list.index(symbol)])
                average_score += accuracy_score1
                successful_fits += 1
                ############################################################################################################
            except :
                pass
                # print(f"{symbol} has low quality data")
                #ShowToast(f"{symbol} has low quality data.")

            

        average_prediction /= successful_fits
        average_score /= successful_fits


        # price last
        df = chart_symbol_df
        symbol = chart_symbol
        latest_prediction = chart_symbol_latest_prediction 
        average_score = average_score
        
        
        print(symbol, average_prediction, average_score)

        last_price = df['open'].values[-1:]
        last_price = round(float(last_price), 4)
        #print("Last Price for:", symbol)
        #print(last_price)
        # lower_std_deviation last
        lower_std_deviation_last = df[f'lower_{momentum_indicator}'].values[-1:]
        lower_std_deviation_last = round(float(lower_std_deviation_last), 4)
        #print(f"Lower {momentum_indicator} Latest for:", symbol)
        #print(lower_std_deviation_last)
        # upper_std_deviation last
        upper_std_deviation_last = df[f'upper_{momentum_indicator}'].values[-1:]
        upper_std_deviation_last = round(float(upper_std_deviation_last), 5)
        #print(f"Upper {momentum} Latest for:", symbol)
        #print(upper_std_deviation_last)
        # target profit upside 2 standard deviations
        target_profit_long = df[f'upper_{momentum_indicator}_target_profit'].values[-1:]
        target_profit_long = round(float(target_profit_long), 4)
        #print("Upper Profit Target:", symbol)
        #print(target_profit_long)
        # target profit downside 2 standard deviations
        target_profit_short = df[f'lower_{momentum_indicator}_target_profit'].values[-1:]
        target_profit_short = round(float(target_profit_short), 4)
        #print("Lower Profit Target:", symbol)
        #print(target_profit_short)
        # average_std_deviation last
        avg_std_deviation_last = df[f'average_{momentum_indicator}'].values[-1:]
        avg_std_deviation_last = round(float(avg_std_deviation_last), 4)
        #print(f"Average {momentum_indicator} Latest for:", symbol)
        #print(avg_std_deviation_last)
        ####################################################################################################################
        # calculate target level for exit, based on mean reversion to average mean of deviation move
        # distance between last open price and avg_std_deviation last
        target_profit_distance = abs(avg_std_deviation_last - last_price)
        target_profit_distance = round(float(target_profit_distance), 4)
        target_profit_distance = target_profit_distance * 10000
        target_profit_distance = int(target_profit_distance)
        #print("Target Profit Distance for:", symbol)
        #print(target_profit_distance)
        ####################################################################################################################
        # Get open positions:
        positions = 0 # default to 0
        try:
            # get open positions
            positions=mt5.orders_get(symbol)
            print("Open Positions : ",positions)
            #print("Open Positions for:", symbol)
            #print(len(positions))
        except:
            pass
        ####################################################################################################################
        # LONG LOGIC TO ENTER:
        if (not Tester and latest_prediction == 1 and accuracy_score1>=0.5):
            # prepare the buy request structure
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                #print(symbol, "not found, can not call order_check()")
                mt5.shutdown()
                quit()
            point = mt5.symbol_info(symbol).point
            price = upper_std_deviation_last
            deviation = 20
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY_STOP,
                "price": price,
                "sl": lower_std_deviation_last,
                "tp": target_profit_long,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            # send a trading request
            result = mt5.order_send(request)
            # check the execution result
            #print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price, deviation));
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                #print("2. order_send failed, retcode={}".format(result.retcode))
                # request the result as a dictionary and display it element by element
                result_dict = result._asdict()
                
                #ShowToast(f"Trade Failed, Error {result.retcode} : {result_dict['comment']} ")
                for field in result_dict.keys():
                    pass
                    #print("   {}={}".format(field, result_dict[field]))
        ####################################################################################################################
        ####################################################################################################################
        # SHORT LOGIC TO ENTER:
        if (not Tester and latest_prediction == -1 and accuracy_score1>=0.5):
            # prepare the buy request structure
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                #print(symbol, "not found, can not call order_check()")
                mt5.shutdown()
                quit()
            point = mt5.symbol_info(symbol).point
            price = lower_std_deviation_last
            deviation = 20
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL_STOP,
                "price": price,
                "sl": upper_std_deviation_last,
                "tp": target_profit_short,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            # send a trading request
            result = mt5.order_send(request)
            # check the execution result
            #print(
                # "1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price, deviation));
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                #print("2. order_send failed, retcode={}".format(result.retcode))
                # request the result as a dictionary and display it element by element
                result_dict = result._asdict()
                #ShowToast(f"Trade Failed, Error {result.retcode} : {result_dict['comment']} ")
                for field in result_dict.keys():
                    pass
                    #print("   {}={}".format(field, result_dict[field]))
        ####################################################################################################################
    else:
        pass

    #print(str(datetime.now()) + "	 " + timeframe + " Update Function Completed.\n")

    return symbol, average_prediction, average_score




########################################################################################################################
# INPUTS:

momentum_indicator = "atr" # Can use "std_deviation" or "atr" as well

num_of_std_deviations = 1
num_of_std_deviations_profit = 2 # profit threshold change to 2 or 3
max_positions = 5 # enter max number of positions allowed per currency pair
lot = 0.5 # amount of quantity or size to enter in per trade
chart_symbol = "EURUSD.i" # Symbol on your MT5 chart
data_source = '12Data' # "12Data" or "MT5"
apikey="f9b2c703f5554f9288e21cfca2bf9dc8" # 12Data Api Key

########################################################################################################################




# Banned Symbols 
banned_symbols = ["FB", "META", "ICPUSD"]

# List of all symbols
symbols_list = []
for symbol in mt5.symbols_get():
    if symbol.name not in banned_symbols:
        symbols_list.append(symbol.name)


def Run(chart_symbol, Tester=False):
    loop = True
    while loop:
        current_time = datetime.now()
        if True:
            ShowToast("Synchronizing Data ...")
            gl_dict = Update_GL()
            ShowToast("Gainers-Loosers Updated!")
            corr_df, chart_symbol_corr = Update_Correlation(gl_dict=gl_dict)
            corr_df = corr_df.replace(1, 0)
            ShowToast("Correlation Updated!")
            cal_df = Update_Calendar()
            ShowToast("Calendar Updated!")
            # print(chart_symbol_corr)
            symbol, prediction, score = Update_AI(corr_df=corr_df, symbols=list(chart_symbol_corr), Tester=Tester)
            ShowToast("AI Updated!")
        
        loop = not Tester
        
    return symbol, prediction, score



def AutoTrader(progress_callback=""):
    
    ClearDir('Data')
    for symbol in symbols_list:
        df = get_symbol_rates(symbol,1000)
        df.to_csv("Data/"+symbol+".csv", decimal=",", sep=";")


    Run(chart_symbol)

if sys.argv[0]=="run" or __name__=="__main__":
    AutoTrader()