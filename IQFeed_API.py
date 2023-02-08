import socket
import pandas as pd 
import numpy as np
import datetime as dt
import time
import shutil



# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# get local machine name
host = "127.0.0.1"                           

port = 9100

# connection to hostname on the port.
s.connect((host, port))  

exchange_list = pd.read_csv("exchange_list.csv", sep=",")
exchange_names = list(exchange_list["Short Name"])


commands = open("IQFeedSymbolsCommands.txt", "r").read().split("\n")

def Request_Symbols(progress="", job_id=""):
    """ Request Full list of symbols """

    s.settimeout(5)                             
    chunk = []
    Symbols = []
    subscribed_exchanges = exchange_names
    #for i in range(79): # Pull data for all listed 78 exchange
        #for j in range(1,42):
            # Send request for Times&Sales data
    
    s.sendall(b"S,SET PROTOCOL,6.2\n\r")
    for cmd in commands: 
        request = cmd+"\n\r"
        s.sendall(request.encode())
        s.__enter__()
        data = [","]
        while "!ENDMSG!" not in data[-1]:
            try:
                data = s.recv(1024).decode().split("\r\n")
                while("" in data):
                    data.remove("")
                
                if "No file available" in data:
                    break
                for d in data:
                    try:
                        symbol = d.split(",")
                        symbol = symbol[symbol.index("LM")+1]
                        if symbol != '':
                            Symbols.append(symbol)
                        if data[0]!=None:
                            if "No file available" in data[0] :
                                #print(f"No Data available for exchange : {exchange_names[i]}")
                                #subscribed_exchanges.pop(i)
                                break
                    except:
                        pass
                
            except socket.timeout:
                #print("Timeout reached")
                break
        with open("Symbols_list1.txt", "w") as f:
            f.write("\n".join(Symbols))
        with open("Chunk.txt", "w") as f:
            f.write("\n".join(str(chunk)))
        
        with open("SubscribedExchanges_list1.txt", "w") as f:
            f.write("\n".join(subscribed_exchanges))

        
        shutil.copyfile('Symbols_list1.txt', 'Symbols_list.txt') 
        shutil.copyfile('SubscribedExchanges_list1.txt', 'SubscribedExchanges_list.txt') 
    #print("Total Subscribed symbols >> ",len(Symbols))
    return Symbols, subscribed_exchanges 

    

def Request_last_trades(symbol, num_ticks):
    """ Request Times&Sales data for a given number of ticks """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

    # get local machine name
    host = "127.0.0.1"                           

    port = 9100

    # connection to hostname on the port.
    s.connect((host, port))  
        
    s.settimeout(5)                            

    # Send request for Times&Sales data
    s.sendall(b"S,SET PROTOCOL,6.2\n\r")
    request = f"HTX,{symbol},{num_ticks}\n"
    s.sendall(request.encode())
    s.__enter__()
    
    Response = []
    
    try:
        data = [None]
        while data[-1] != "!ENDMSG!,":
            # receive data from the server
            data = s.recv(1024).decode().split("\r\n")

            while("" in data):
                data.remove("")
            Response.append(data)
    except socket.timeout:
        pass       
    except: 
        pass
  
    Data = [[x.split(",") for x in y ] for y in Response[1:]]
    final_data = []
    for arr in Data:
        for arr1 in arr:
            final_data.append(arr1)
    FormattedData = []
    for row in final_data:
        if 'LH' in row:
            FormattedRow = []
            for x in row:
                try:
                    FormattedRow.append(eval(x))
                except:
                    try:
                        FormattedRow.append(dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f"))
                    except:
                        FormattedRow.append(x)
            if "NaN" not in FormattedRow and len(FormattedRow)>6 and FormattedRow[-3] in [0,1,2,3]:
                print(FormattedRow)
                FormattedData.append(FormattedRow[1:7]+[FormattedRow[-3]])
    Trades_df = pd.DataFrame(FormattedData, columns=["Date", "Last", "Last Size", "Volume", "Bid", "Ask","Type"])
    s.close()

    return Trades_df


def Request_OHLCV(symbol, interval, num_ticks):
    """ Request OHLCV data for a given number of candles and interval """

    #print(f"Downloading {symbol} ")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    s.settimeout(20)
    # get local machine name
    host = "127.0.0.1"                           

    port = 9100

    # connection to hostname on the port.
    s.connect((host, port))                    
    # Send request for Times&Sales data
    s.sendall(b"S,SET PROTOCOL,6.2\n\r")
    request = f"HIX,{symbol},{interval},{num_ticks}\n"
    s.sendall(request.encode())
    s.__enter__()
    
    Response = []
    data = [None]
    try:
        while data[-1] != "!ENDMSG!,":
            # receive data from the server
            data = s.recv(1024).decode().split("\r\n")

            while("" in data):
                data.remove("")
            Response.append(data)
    except socket.timeout:
        print("Timeout reached for " + symbol)    
    except :
        pass
  
    Data = [[x.split(",") for x in y ] for y in Response[1:]]
    final_data = []
    for arr in Data:
        for arr1 in arr:
            final_data.append(arr1)
    FormattedData = []
    for row in final_data:
        if 'LH' in row:
            FormattedRow = []
            for x in row:
                try:
                    FormattedRow.append(eval(x))
                except:
                    try:
                        FormattedRow.append(dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S.%f"))
                    except:
                        FormattedRow.append(x)
            if "NaN" not in FormattedRow and len(FormattedRow)>6 :
                FormattedData.append(FormattedRow[1:7])
    Trades_df = pd.DataFrame(FormattedData, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    

    return Trades_df


if __name__ == "__main__":
   print(Request_last_trades("AAPL", 50))