import sys
import os
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import MetaTrader5 as mt5
import uuid
from ANN import ModelANN
from CNN import ModelCNN
from RNN import ModelRNN
import tensorflow as tf

from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from IQFeed_API import *

import psutil

import requests
from multiprocessing import Pool

from Toaster import QToaster

if mt5.initialize():
    print(mt5.terminal_info())

try:
    os.mkdir(mt5.terminal_info().commondata_path.replace("\\", "/")+"/Data")
except:
    pass

def ShowToast(msg : str):
    corner = Qt.Corner(3)
    QToaster.showMessage(
                None, msg, corner=corner, desktop=True)



class WorkerSignals(QObject):
	"""
	Define signals available from thread while running or finished

		progress:
			precentage progress value int : 0-100
	"""

	progress = pyqtSignal(str, int)
	finished = pyqtSignal(str)
	started = pyqtSignal(bool)
	list_response = pyqtSignal(list)

class Worker(QRunnable):

    def __init__(self,function,args=(),desc=""):
        super().__init__()
        self.desc = desc
        self.job_id = uuid.uuid4().hex
        self.signals  =WorkerSignals()
        self.function = function
        self.args = args

    def run(self):
            self.signals.started.emit(True)
            self.result = self.function(*self.args,progress = self.signals.progress,job_id=self.job_id)
            #self.signals.list_response.emit(self.result)
            self.signals.finished.emit(self.job_id)



    


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spash Screen Example')
        self.setFixedSize(1100, 500)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.counter = 0
        self.n = 300 # total instance

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.loading)
        self.timer.start(5)
        

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QFrame()
        layout.addWidget(self.frame)


        self.labelTitle = QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

    
        # Optional, resize window to image size
        


        # center labels
        self.labelTitle.resize(self.width() - 10, 150)
        self.labelTitle.move(0, 40) # x, y
        self.labelTitle.setText('SONIC')
        self.labelTitle.setAlignment(Qt.AlignCenter)

        self.labelDescription = QLabel(self.frame)
        self.labelDescription.resize(self.width() - 10, 50)
        self.labelDescription.move(0, self.labelTitle.height())
        self.labelDescription.setObjectName('LabelDesc')
        self.labelDescription.setText('<strong>Loading Data</strong>')
        self.labelDescription.setAlignment(Qt.AlignCenter)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.resize(self.width() - 200 - 10, 50)
        self.progressBar.move(100, self.labelDescription.y() + 130)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat('%p%')
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, self.n)
        self.progressBar.setValue(20)

        self.labelLoading = QLabel(self.frame)
        self.labelLoading.resize(self.width() - 10, 50)
        self.labelLoading.move(0, self.progressBar.y() + 70)
        self.labelLoading.setObjectName('LabelLoading')
        self.labelLoading.setAlignment(Qt.AlignCenter)
        self.labelLoading.setText('SONIC is awaking...')

        
        sonic = QLabel(self.frame)
        pixmap = QPixmap('Img/sonic.png')
        scale = 2.9
        pixmap = pixmap.scaledToWidth(int(pixmap.width()/scale))
        pixmap = pixmap.scaledToHeight(int(pixmap.height()/scale))
        sonic.setPixmap(pixmap)
        sonic.setGeometry(QRect(390,75,pixmap.width(),pixmap.height()))

    def loading(self):
        self.progressBar.setValue(self.counter)



        if self.counter == int(self.n * 0.05):
            pass
        elif self.counter == int(self.n * 0.3):
            self.labelDescription.setText('<strong>Updating Symbols list</strong>')

        elif self.counter == int(self.n * 0.6):
            self.labelDescription.setText('<strong>Preparing Environment</strong>')
        elif self.counter >= self.n:
            
            self.myApp = Window(self)
            self.myApp.show()
            self.timer.stop()
            self.close()

            time.sleep(1)


        self.counter += 1



class TableModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                try:
                    return float(self._data.iloc[index.row(), index.column()])
                except: 
                    return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None





class Window(QMainWindow):
    def __init__(self, splash):
        super().__init__()

        self.splash = splash

        self.setWindowTitle("AutoTrader")
        self.setFixedSize(QSize(950,600))

        self.taskMgr = QThreadPool()

        self.wgt = QWidget()
        self.lyt = QGridLayout(self.wgt)
        
        self.status = self.statusBar()
         
        
        self.Tabs = QTabWidget()
        
        self.LiveTab = QWidget()
        self.LiveTab_lyt = QGridLayout(self.LiveTab)
        
        self.BacktestTab = QWidget()
        self.BacktestTab_wgt = QWidget()
        self.BacktestTab_lyt = QGridLayout(self.BacktestTab)


        self.bt_model = QButtonGroup()
        
        self.m1 = QRadioButton("Artificial Neural Network")
        self.m2 = QRadioButton("Convolutional Neural Network")
        self.m3 = QRadioButton("Recurrent Neural Network")

        self.m1.setChecked(True)

        self.bt_model.addButton(self.m1)
        self.bt_model.addButton(self.m2)
        self.bt_model.addButton(self.m3)


        self.bt_start = QPushButton("Start")
        self.bt_update = QPushButton("Update Correlation")
        self.bt_start.setStyleSheet("""
            QPushButton{     
                background-color: #FED037;
                color: #2F4454;
                font-size: 20px;
            }
        """)

        self.bt_update.setStyleSheet("""
            QPushButton{     
                background-color: #FED037;
                color: #2F4454;
                font-size: 18px;
            }
        """)


        self.bt_InpWgt = QWidget()
        self.bt_Inp_lyt = QGridLayout(self.bt_InpWgt)

        self.bt_risk = QComboBox()
        self.bt_risk.addItems(["Balance","Equity"])
        self.bt_sl = QComboBox()
        self.bt_sl.addItems(["Trailing", "Static"])
        self.bt_dd = QComboBox()
        self.bt_dd.addItems(["Trailing", "Static"])
        self.bt_ddt = QComboBox()
        self.bt_ddt.addItems(["Balance", "Equity"])

        self.bt_lots = QComboBox()
        self.bt_lots.addItems(["ON", "OFF"])
        self.bt_positions = QComboBox()
        self.bt_positions.addItems(["ON", "OFF"])
        self.bt_dprofit = QComboBox()
        self.bt_dprofit.addItems(["ON", "OFF"])


        
        self.order_type = QComboBox()
        self.order_type.addItems(["OFF", "ON"])

        self.takeProfit = QComboBox()
        self.takeProfit.addItems(["ON", "OFF"])

        
        self.bt_on_off1 = QComboBox()
        self.bt_on_off1.addItems(["OFF", "ON"])
        
        self.bt_Inp_lyt.addWidget(QLabel("Trade Next Day"), 1,0)
        self.bt_Inp_lyt.addWidget(self.bt_on_off1, 1,1)


        self.bt_Inp_lyt.addWidget(QLabel("Zone Recovery "),2,0)
        self.bt_Inp_lyt.addWidget(self.order_type,2,1)

        
        self.bt_symbol = QComboBox()
        self.bt_symbol.setEditable(True)
        self.mt5_symbols = [x.name for x in mt5.symbols_get()]
        self.bt_symbol.addItems([x.split("_")[0] for x in os.listdir('Data') if x.split("_")[0] in self.mt5_symbols])
        self.bt_Inp_lyt.addWidget(QLabel("Symbol "),3,0)
        self.bt_Inp_lyt.addWidget(self.bt_symbol,3,1)


        self.takeProfit_value = QLineEdit()
        #self.bt_Inp_lyt.addWidget(QLabel("Take Profit %"),4,0)
        #self.bt_Inp_lyt.addWidget(self.takeProfit,4,1)
        #self.bt_Inp_lyt.addWidget(self.takeProfit_value,4,2)



        self.risk_value = QLineEdit("1")
        self.max_lots_value = QLineEdit("0")
        self.max_positions_value = QLineEdit("0")
        self.max_daily_profit = QLineEdit("0")
        self.max_daily_dd = QLineEdit("0")

        self.bt_takeprofit = QLineEdit("1")
        self.bt_stoploss = QLineEdit("1")

        self.bt_Inp_lyt.addWidget(QLabel("TakeProfit %"), 5, 0)
        self.bt_Inp_lyt.addWidget(self.bt_takeprofit, 5, 1)
        self.bt_Inp_lyt.addWidget(QLabel("StopLoss %"), 6, 0)
        self.bt_Inp_lyt.addWidget(self.bt_stoploss, 6, 1)

        self.bt_Inp_lyt.addWidget(QLabel("Risk %"),7,0)
        self.bt_Inp_lyt.addWidget(self.bt_risk,7,1)
        self.bt_Inp_lyt.addWidget(self.risk_value,7,2,1,3)



        self.bt_Inp_lyt.addWidget(QLabel("Max Lots Per Trade"),8,0)
        self.bt_Inp_lyt.addWidget(self.max_lots_value,8,2,1,3)
        self.bt_Inp_lyt.addWidget(self.bt_lots,8,1)


        self.bt_Inp_lyt.addWidget(QLabel("Max Open Trades"),9,0)
        self.bt_Inp_lyt.addWidget(self.bt_positions,9,1)
        self.bt_Inp_lyt.addWidget(self.max_positions_value,9,2,1,3)

        self.bt_Inp_lyt.addWidget(QLabel("Max Daily Profit %"),10,0)
        self.bt_Inp_lyt.addWidget(self.max_daily_profit,10,2,1,3)
        self.bt_Inp_lyt.addWidget(self.bt_dprofit,10,1)

        self.bt_on_off = QComboBox()
        self.bt_on_off.addItems(["ON", "OFF"])

        self.bt_Inp_lyt.addWidget(QLabel("Max Daily DrawDown %"),11,0)
        self.bt_Inp_lyt.addWidget(self.bt_on_off,11,1)    
        self.bt_Inp_lyt.addWidget(self.bt_dd,11,2,1,2)
        self.bt_Inp_lyt.addWidget(self.max_daily_dd,11,4)




        edit_api = QPushButton("")
        edit_api.setCheckable(True)
        edit_api.setChecked(True)
        edit_api.clicked.connect(self.editAPI)
        edit_api.setStyleSheet("""
            QPushButton{
                background-color : #2F4454;
            }
        """)
        edit_api.setIcon(QIcon("Img/ClosedLock.png"))

        self.api_username = QLineEdit()
        self.api_password = QLineEdit()

        self.api_username.setEnabled(False)
        self.api_password.setEnabled(False)

        self.api_onoff = QComboBox()
        self.api_onoff.addItems(["OFF", "ON"])
        self.bt_api_key = QLineEdit()
        self.bt_api_key.setDisabled(True)
        #self.bt_Inp_lyt.addWidget(QLabel("API KEY"),10,0)
        #self.bt_Inp_lyt.addWidget(self.api_onoff,10,1)
        #self.bt_Inp_lyt.addWidget(self.bt_api_key,10,2,1,3)
        #self.bt_Inp_lyt.addWidget(edit_api,10,5,1,1)
        #self.bt_Inp_lyt.addWidget(QLabel("Username "),11,1,1,1)
        #self.bt_Inp_lyt.addWidget(self.api_username,11,2,1,1)
        #self.bt_Inp_lyt.addWidget(QLabel("Password"),11,3,1,1)
        #self.bt_Inp_lyt.addWidget(self.api_password,11,4,1,1)


        self.bt_initial_balance = QLineEdit("100000") 

        self.bt_Inp_lyt.addWidget(QLabel("Deposit"),12,0,1,1)
        self.bt_Inp_lyt.addWidget(self.bt_initial_balance,12,1,1,1)




        self.BacktestTab_lyt.addWidget(self.m1,0,0,1,1, alignment=Qt.AlignTop)
        self.BacktestTab_lyt.addWidget(self.m2,0,1,1,1, alignment=Qt.AlignTop)
        self.BacktestTab_lyt.addWidget(self.m3,0,2,1,2, alignment=Qt.AlignTop)

        self.bt_start.setStyleSheet("""
                QPushButton{
                background-color : #4ee44e;
                color: green;
                font-size:18px;
                }
        """)
        
        self.bt_start.clicked.connect(self.Run_BT)
        self.bt_update.clicked.connect(self.BTUpdate)

        
        
        self.status.showMessage("Checking IQFeed Connection...")
        self.api_connect = self.check_IQFeed_connection()

        
        
        

        self.api_status = "✅" if self.api_connect else "❌"
        self.mt5_status = "✅" if mt5.last_error()[1]=="Success" else "❌"

        self.BacktestTab_lyt.addWidget(self.bt_InpWgt,1,0,1,2, alignment=Qt.AlignTop)       
        self.BacktestTab_lyt.addWidget(QLabel(f"MT5 Account status\nIQFeed API Status"),1,2, alignment=Qt.AlignCenter)
        self.BacktestTab_lyt.addWidget(QLabel(f"{self.mt5_status} \n{self.api_status}"),1,3, alignment=Qt.AlignCenter)
        self.BacktestTab_lyt.addWidget(self.bt_start,2,3,1,1, alignment=Qt.AlignBottom)
        self.BacktestTab_lyt.addWidget(self.bt_update,2,2,1,1, alignment=Qt.AlignBottom)


        self.lyt.addWidget(self.Tabs)


        self.iqFeedTab = QWidget()
        self.iqFeedTab_lyt = QGridLayout(self.iqFeedTab)
        
        



        self.Tabs.addTab(self.BacktestTab,"Backtester")
        #self.Tabs.addTab(self.LiveTab,"Live Trader")
        #self.Tabs.addTab(self.iqFeedTab,"IQFeed Data")




        
        
        



        self.pb = QProgressBar()
        self.pb.setValue(0)
        self.lyt.addWidget(self.pb,10,0,1,2)
        self.setCentralWidget(self.wgt)

        #self.Update_Correlation()
       

        self.counter = 1
        self.n = 100

    def ShowTable(self,df,parent):
            table = QTableView()
            table.setSortingEnabled(True)
            table_model = TableModel(df)
            proxyModel =  QSortFilterProxyModel(self)

            proxyModel.setSourceModel(table_model)
            table.setModel(proxyModel)
            table.setParent(parent)
            
            
            table.show()
    def BTUpdate(self):
        worker = Worker(self.UpdateSymbols)
        self.taskMgr.start(worker)
    
    def UpdateSymbols(self, progress="", job_id=""):
        
        while True:
            self.symbols_list = []
            worker = Worker(Request_Symbols)
            worker.signals.started.connect(self.updateIQFeedTable, type=Qt.QueuedConnection)
            self.taskMgr.start(worker)
            self.status.showMessage("")
            try:  
                
                self.status.showMessage("Calculating Correlation...")
                worker = Worker(self.Update_Correlation)
                self.taskMgr.start(worker)
                
                with Pool() as p:
                    p.map(self.DownloadSave, self.symbols_list)
            except:
                pass
            time.sleep(5000)
            self.status.showMessage("Indexing Symbols...")
            
  

    def updateIQFeedTable(self, e=""):
        
        self.status.showMessage("Indexing Symbols...")
        if True:
                try:
                    time.sleep(1)
                    if True:
                        with open("Symbols_list.txt", "r") as f:
                            self.symbols_list = f.read().split("\n")

                        with open("SubscribedExchanges_list.txt", "r") as f:
                            self.exchanges_list = f.read().split("\n")
                    
                        
                    self.symbols_list = list(dict.fromkeys(self.symbols_list))
                    self.exchanges_list = list(dict.fromkeys(self.exchanges_list))

                    self.IQFeedData_df = pd.DataFrame()
                    self.IQFeedData_df["ID"] = range(1,len(self.symbols_list)+1)
                    self.IQFeedData_df["Symbols"] = self.symbols_list
                    self.IQFeedData_df["Exchanges"] = self.exchanges_list + [None] * (len(self.symbols_list) - len(self.exchanges_list))
                    self.ShowTable(self.IQFeedData_df,self.iqFeedTab)
                except:
                    pass


    def check_IQFeed_connection(self):
        # create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

        # get local machine name
        host = "127.0.0.1"                           

        port = 9100

        # connection to hostname on the port.
        s.connect((host, port))                               

        # Send request for Times&Sales data
        s.sendall(b"S,SET PROTOCOL,6.2\n\r")
        request = f"T\n\r"
        s.sendall(request.encode())
        s.__enter__()
        data = [None]
        while data[-1] != "!ENDMSG!,":
            # receive data from the server
            data = s.recv(1024).decode().split("\r\n")
            if 'S,CURRENT PROTOCOL,6.2' in data:
                return True

        return False


    def loading(self):
        self.pb.setValue(self.counter)
        self.counter += 1

    
    def Mt5_Download(self, InpSymbol, timeframe, start_pos, count):
        mt5.initialize()
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


    def Backtest(self,progress="", job_id=""):
        
        

        self.pb.setValue(25)
        nn = [self.m1.isChecked(),
        self.m2.isChecked(),
        self.m3.isChecked()]
        models = ["ANN", "CNN", "RNN"]
        
        model = models[nn.index(True)]
        symbol = self.bt_symbol.currentText()   

        

        if model == "ANN":
            predictions = ModelANN(symbol)
            
        if model == "CNN":
            predictions = ModelCNN(symbol)
            
        if model == "RNN":
            predictions = ModelRNN(symbol)
        
        if isinstance(predictions, bool):
            if not predictions:
                print("Model doesn't converge try another one for that symbol")
                return
                #self.ShowMessageBox(QMessageBox.Warning, f"{model} Doesn't converge for symbol {symbol}, try another model insted.")

        mt5.initialize()
        rates = mt5.copy_ticks_from(symbol, dt.datetime.now()-dt.timedelta(days=10), 999999, mt5.COPY_TICKS_ALL)
        df = pd.DataFrame(rates)
        df.index = df["time"].apply(lambda x: dt.datetime.fromtimestamp(x))
        df.columns = [col if col!="time" else "Date" for col in df.columns ]
        df["Date"] = df.index
        print(df["Date"])
        df = df["ask"].resample("1s").ohlc()
        df.dropna(inplace=True)

        df = df.head(10000)
        df.columns = [col if col!="open" else "Open" for col in df.columns ]
        df.columns = [col if col!="high" else "High" for col in df.columns ]
        df.columns = [col if col!="low" else "Low" for col in df.columns ]
        df.columns = [col if col!="close" else "Close" for col in df.columns ]
        

        backtest_df = df
        #backtest_df.index  = pd.to_datetime(backtest_df['Date'], format='%Y-%m-%d %H:%M:%S')

        self.bt_balance = float(self.bt_initial_balance.text())
        predictions = predictions.iloc[::-1]


        backtest_df = pd.concat([backtest_df, predictions])

        backtest_df.sort_index(inplace=True)
        
        print(backtest_df)
        print(predictions)

        backtest_df.to_csv("BacktestDF.csv", sep=';', decimal=',')

        self.pb.setValue(50)
        print("Backtest DF>> ", backtest_df)
            
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

        df = pd.read_csv("BacktestDF.csv", sep=";", decimal=",")

        df = df.iloc[::-1]

        df["ATR"] = atr(df, 14)

        df.drop("Date", axis=1, inplace=True)
        df.columns = [x if x!='Unnamed: 0' else "Date" for x in df.columns]
        df.sort_values("Date", ascending=True, inplace=True)
        df.index = [pd.Timestamp(x) for x in  df["Date"]]

        self.leverage = 30
        self.balance = float(self.bt_initial_balance.text()) 
        self.BuyPower = self.balance * self.leverage
        self.lotSize = max(0, float(self.max_lots_value.text()))
        self.prevStartOfTheDay = 0
        self.prevGridofTheDay = 0
        self.prevOpenofTheDay = 0
        self.prevClose = 0
        df["Balance"] = self.balance
        df["BuyPower"] = self.BuyPower
        df["Equity"] = self.balance
        self.PendingOrders = []
        self.OpenPositions = []
        self.MaxOpenPositions = int(self.max_positions_value.text())

        self.PnL = 0
        self.rowNum = 0

        df['Open Positions'] = [[] for x in  range(df.shape[0])]

        print(df)

        def TradeLogic(row):
            
            self.rowNum += 1
            if row.name.hour==row.name.minute==row.name.second==0:
                if row["Values"]<=1:
                    self.prevStartOfTheDay = row["Values"] 
                    
                else:
                    self.prevStartOfTheDay = random.random()

                row["GridSize"] = self.prevStartOfTheDay * row["ATR"]
                self.prevGridofTheDay = row["GridSize"]
                try:
                    self.prevOpenofTheDay = df[row.name:].iloc[1, df.columns.get_loc("Open")]
                    self.prevClose = df[row.name:].iloc[1, df.columns.get_loc("Close")]
                except:
                    pass


            else:
                
                row["Values"]  = self.prevStartOfTheDay
                row["GridSize"]  = self.prevGridofTheDay
                row["ATR+1"] = self.prevOpenofTheDay + self.prevGridofTheDay
                row["ATR+2"] = self.prevOpenofTheDay + self.prevGridofTheDay * 2
                row["ATR+3"] = self.prevOpenofTheDay + self.prevGridofTheDay * 3
                row["ATR+4"] = self.prevOpenofTheDay + self.prevGridofTheDay * 4
                row["ATR+5"] = self.prevOpenofTheDay + self.prevGridofTheDay * 5
                row["ATR-1"] = self.prevOpenofTheDay - self.prevGridofTheDay
                row["ATR-2"] = self.prevOpenofTheDay - self.prevGridofTheDay * 2
                row["ATR-3"] = self.prevOpenofTheDay - self.prevGridofTheDay * 3
                row["ATR-4"] = self.prevOpenofTheDay - self.prevGridofTheDay * 4
                row["ATR-5"] = self.prevOpenofTheDay - self.prevGridofTheDay * 5
    
                BuyCond = True in [self.prevClose<self.prevOpenofTheDay +  self.prevGridofTheDay * i and row["Close"]>self.prevOpenofTheDay +  self.prevGridofTheDay * i for i in range(5)]
                SellCond = True in [self.prevClose>self.prevOpenofTheDay -  self.prevGridofTheDay * i and row["Close"]<self.prevOpenofTheDay -  self.prevGridofTheDay * i for i in range(5)]

                if BuyCond and len(row["Open Positions"])<self.MaxOpenPositions: # Buy Condition (Position Type, price, size, profit)
                    row["Open Positions"] = self.OpenPositions + [["Buy", row["Close"], self.lotSize, 0]]
                    row["BuyPower"] -= self.lotSize * row["Close"]
                    self.OpenPositions = row["Open Positions"]
                    self.prevOpenofTheDay = row["Close"]
                    print("Opened Buy Postion")
                elif SellCond and len(row["Open Positions"])<self.MaxOpenPositions: # Sell Condition
                    row["Open Positions"] = self.OpenPositions + [["Sell", row["Close"], self.lotSize, 0]]
                    row["BuyPower"] -= self.lotSize * row["Close"]
                    self.OpenPositions = row["Open Positions"]
                    self.prevOpenofTheDay = row["Close"]
                    print("Opened Sell Postion")
                else:
                    row["Open Positions"] = self.OpenPositions

                XOpenPositions = row["Open Positions"]
                XOpenPositions = []
                PnL = 0
                for p in row["Open Positions"]: # Update PNL
                    new_record = []
                    if p[0]=="Buy":
                        new_record.append(p[0])
                        new_record.append(p[1])
                        new_record.append(self.lotSize)
                        new_record.append(row["Close"]-p[3])
                        self.PnL += (row["Close"]-p[3]) * self.lotSize
                        #print("Opened Buy Position")
                    if p[0]=="Sell":
                        new_record.append(p[0])
                        new_record.append(p[1])
                        new_record.append(self.lotSize)
                        new_record.append(p[3]-row["Close"])
                        self.PnL += (p[3]-row["Close"]) * self.lotSize
                        #print("Opened Sell Position")
                    XOpenPositions.append(new_record)
                self.OpenPositions = XOpenPositions
                row["Open Positions"] = XOpenPositions

                count_long = len([x for x in XOpenPositions if x[0]=="Buy"])
                count_shorts = len([x for x in XOpenPositions if x[0]=="Sell"])
                
                print(f"Opened Longs : {count_long}\nOpened Shorts : {count_shorts}")


            self.pb.setValue(int((self.rowNum*100/df.shape[0])))
            print(self.PnL)

            return row


        df = df.apply(TradeLogic, axis=1)

        df.to_csv("BacktestDF1.csv", sep=";", decimal=",")

        df.dropna(subset=["Date.1"], inplace=True, axis=0)
        df.dropna(subset=["ATR+1"], inplace=True, axis=0)


    def ShowMessageBox(self,msg_type, msg):
        msg = QMessageBox()
        
    
        # setting message for Message Box
        msg.setText(msg)
        
        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        # start the app
        retval = msg.exec_()

    def Run_BT(self, e=""):
        if self.bt_start.text() == "Start":
            self.bt_start.setText("Stop")
            
            #worker = Worker(self.Backtest)
            #self.taskMgr.start(worker)

            self.bt_start.setStyleSheet("""

                QPushButton{
                    background-color: #FB6D4c;
                    color:#8A0000;
                    font-size:18px;
                }
            
            """)

            self.Backtest()

        elif self.bt_start.text() == "Stop":
            self.bt_start.setText("Start")
            
            self.bt_start.setStyleSheet("""

                QPushButton{
                    background-color : #4ee44e;
                    color: green;
                    font-size:18px;
                }
            
            """)
    @staticmethod
    def DownloadSave(symbol, row=14400, progress="", job_id=""):
        if symbol[:2] == 'IQ':
            symbol = symbol[2:]
            print("Downloading IQFEED : "+symbol+"...")
            df = Request_OHLCV(symbol,60, row)
            df.to_csv(mt5.terminal_info().data_path.replace("\\", "/")+f"/MQL5/Files/Data/"+symbol+".csv", sep=";", decimal=",", index=False, mode="w")
            
        

    def ClearDir(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    @staticmethod
    def Download_MT5_Symbols(symbol, row=1440, e=""):
        symbol = symbol[2:]
        print("Downloading MetaTrader5 : "+symbol+"...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, row)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df.to_csv(mt5.terminal_info().data_path.replace("\\", "/")+f"/MQL5/Files/Data/"+symbol+".csv", sep=";", decimal=",", index=False, mode="w")

    def Update_Correlation(self,gl_dict=None, progress="", job_id=""):
        row = 14400
        method = "Open" # You can choose close as well
        big_df = pd.DataFrame()
        min_rows = 999999999999999999   
        with open("Symbols_list.txt", "r") as f:
            self.symbols_list = ["MT"+x.name for x in mt5.symbols_get()] + ["IQ"+x for x in f.read().split("\n")] 

        self.temp = self.symbols_list 
        self.ClearDir('mt5.terminal_info().data_path.replace("\\", "/")+f"/MQL5/Files/Data/')
        
        mt5_list = [x for x in self.symbols_list if x[:2] == "MT"]
        with Pool() as p:
            p.map(self.Download_MT5_Symbols, mt5_list)
            
                
            
        print(self.symbols_list)
        
        self.temp = self.symbols_list
        for i, symbol in enumerate(self.symbols_list):
            symbol = symbol[2:]
            self.status.showMessage(f"Processing {symbol}")
            try:
                df = pd.read_csv(mt5.terminal_info().data_path.replace("\\", "/")+f"/MQL5/Files/Data/"+symbol+".csv", decimal=",", sep=";")
                
                if df.shape[0]>1000:
                    df[symbol] = df[method]
                    big_df = pd.concat([big_df, df[symbol]],axis=1)
                else:    
                    print('No Data for '+symbol)
                    self.symbols_list.pop(i)
            except:
                print('No Data for '+symbol)
                try:
                    self.temp.pop(self.temp.index("IQ"+symbol))
                except:
                    self.temp.pop(self.temp.index("MT"+symbol))

            #print(f"Size Of BigData >> {sys.getsizeof(big_df)} ")
            try:
                self.splash.progressBar.setValue(min(int(i*100/len(self.symbols_list))+5,100))
            except:
                pass    
        big_df = big_df.iloc[:min_rows, :]
        self.status.showMessage("All Symbols Data are Downloaded!")
        self.symbols_list = self.temp
        big_df.to_csv('BigDataCorr.csv', sep=';',decimal=',')
        big_df = big_df.corr("pearson")

        self.symbols_list = [x[2:] for x in self.symbols_list]
        print(self.symbols_list)
        # print("HeatMap Correlation :",big_df)
        self.symbols_list_df = pd.DataFrame()
        self.symbols_list_df['Symbol'] = self.symbols_list
        self.symbols_list_df = self.symbols_list_df.T
        # print(self.symbols_list)
        df = self.symbols_list_df.to_numpy(dtype=np.str_,copy=True)

        df = np.array(self.symbols_list, dtype=np.str_, copy=True)

        saved = False
        while not saved:
            if True:
                print("Saving Symbols List...")
                # df.tofile(mt5.terminal_info().commondata_path.replace("\\", "/")+f"/Files/Data/symbols_list.txt")
                with open(mt5.terminal_info().data_path.replace("\\", "/")+f"/MQL5/Files/Data/symbols_list.txt", 'w') as file:
                    file.write('\n'.join(self.symbols_list))
                saved = True
                print("Symbols list saved")
            if saved:
                break


        print(big_df)
        self.ClearDir('Data')
        
        big_df.dropna(inplace=True)

        start = time.time()
        chart_symbol_df = None
        for i, symbol in enumerate(self.symbols_list):
            
            print("Calculating Correlation for" + symbol)
            
            try:
                symbol_corr_df = pd.DataFrame(big_df[symbol])
                symbol_corr_df['Symbol'] = symbol_corr_df.index
                symbol_corr_df.index = range(symbol_corr_df.shape[0])
                symbol_corr_df = symbol_corr_df.dropna(axis=0)
                symbol_corr_df = symbol_corr_df.sort_values(symbol, ascending=False)

                Correlation_matrix = pd.concat([symbol_corr_df.head(150),symbol_corr_df.tail(150)], axis=0)
                Correlation_matrix['Direction'] = Correlation_matrix[symbol] / Correlation_matrix[symbol].abs()
                Correlation_matrix[symbol] = Correlation_matrix[symbol].abs()
                Correlation_matrix = Correlation_matrix.sort_values(symbol, ascending=False)
                Correlation_matrix.index = range(Correlation_matrix.shape[0])
                
                if False:#gl_dict:
                    Correlation_matrix['O-C %'] = Correlation_matrix['Symbol'].apply(lambda x: gl_dict[x][0] if x in  gl_dict else 0)
                    Correlation_matrix['H-L %'] = Correlation_matrix['Symbol'].apply(lambda x: gl_dict[x][1] if x in  gl_dict else 0)
                    Correlation_matrix['Chng Direction'] =  Correlation_matrix['O-C %'] /  Correlation_matrix['O-C %'].abs() 
                    Correlation_matrix['O-C %'] =  Correlation_matrix['O-C %'].abs()

                Correlation_matrix = Correlation_matrix.sort_values([symbol], ascending=False)
                
                if symbol == symbol:
                    chart_symbol_df = Correlation_matrix["Symbol"]

                
                Correlation_matrix.to_csv(f"Data/{symbol}_Correlation.csv",sep=";",decimal=",")

                Correlation_matrix['Symbol'] = [self.symbols_list.index(x) for x in list(Correlation_matrix['Symbol'])]


                Correlation_matrix = Correlation_matrix.to_numpy(dtype=np.float64)

                saved = False


                while not saved:
                    try:
                        self.status.showMessage("Saving Symbols Correlation Data...")
                        Correlation_matrix.tofile(
                            mt5.terminal_info().commondata_path.replace("\\", "/")+f"/Data/{symbol}_Correlation.bin")
                        
                        saved = True
                        
                    except :
                        saved = False
                        print(f"Saving {symbol} Bin Correlation Data...")

                self.splash.progressBar.setValue(min(int(i*100/len(self.symbols_list))+33,100))
                # print(symbol, df[symbol].idxmax(), df[symbol].max())
                # print(symbol, df[symbol].idxmin(), df[symbol].min())
            except:
                print('Cannot do calculations for ' + symbol)
        
        self.status.showMessage("")
        try:
            self.splash.progressBar.setValue(99)
            self.splash.counter = 100
        except:
            pass 
        



        return big_df, chart_symbol_df

        # plt.figure(figsize=(11,8))
        # sns.heatmap(big_df, cmap="Greens")
        # plt.show()


    def Download(self, symbol, timeframe, start_pos, count):
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
            df = pd.DataFrame(rates)
            print("Data request result : ", mt5.last_error())
            return df

    def editAPI(self, e=""):
        if self.sender().isChecked():
            self.sender().setIcon(QIcon("Img/ClosedLock.png"))
            self.bt_api_key.setDisabled(True)
            self.api_username.setDisabled(True)
            self.api_password.setDisabled(True)
        else:
            self.sender().setIcon(QIcon("Img/OpenLock.png"))
            self.bt_api_key.setDisabled(False)
            self.api_username.setDisabled(False)
            self.api_password.setDisabled(False)






if __name__ == '__main__':
    # don't auto scale when drag app to a different monitor.
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        #LabelTitle {
            font-size: 60px;
            color: #fed037;
        }

        QLabel{
            font-size : 18px;
        }

        QStatusBar{
            color:white;
        }

        QComboBox{
            font-size : 18px
        }

        QLineEdit{
            font-size : 18px
        }

        #LabelDesc {
            font-size: 30px;
            color: #c2ced1;
        }

        #LabelLoading {
            font-size: 30px;
            color: #e8e8eb;
        }

        QFrame {
            background-color: #2F4454;
            color: rgb(220, 220, 220);
        }

        QMainWindow {
            background-color: #2F4454;
            color: rgb(220, 220, 220);
        }


        QTabWidget::pane {
        border: 1px solid lightgray;
        top:-1px; 
        background: rgb(245, 245, 245);; 
        } 

        QTabBar::tab {
        background: #315067; 
        border: 1px solid lightgray; 
        color: white;
        padding: 15px;
        } 

        QTabBar::tab:selected { 
        background: #2F4454; 
        margin-bottom: -1px; 
        }


        QRadioButton{
            background-color : #FED037;
            color : #2F4454;
            padding:10px;
            font-size : 20px;

        }

        QRadioButton::indicator {
            width: 13px;
            height: 13px;
            border-radius: 50px;
        }

        QRadioButton::checked{
            background-color : #4ee44e;
            color: green;
        }

        QRadioButton::indicator::checked {
            background-color:#4ee44e;
            image: url(:/Img/check.png);
        }
        QRadioButton::indicator::unchecked {
            image: url(:Img/check.png);
        }

        QProgressBar {
            background-color: #FED037;
            color: #2F4454;
            border-style: none;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
        }

        QProgressBar::chunk {
            border-radius: 10px;
            background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #8b8000, stop:1 #376E6F);
        }

        QHeaderView::section {
        background-color: #2F4454;
        }

    ''')
    
    splash = SplashScreen()
    splash.show()

    app.exec_()