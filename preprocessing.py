import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
        
def get_data(symbol, start, end):
    df = yf.download(symbol, start = start, end = end, auto_adjust = False,multi_level_index=False, progress = False)
    df.columns = [col.lower() for col in df.columns.values]
    df = df.rename(columns={'volume': 'volumefrom'})
    df.index = df.index.rename('index')
    return df

def generate_variables(data):
    # Basic variables

    data['HA_close'] = data[['close','high','low','open']].mean(axis = 1)
    data['HA_open'] = data[['close','open']].shift(1).mean(axis = 1)
    data['HA_high'] = data[['high','HA_open','HA_close']].max(axis = 1)
    data['HA_low'] = data[['low','HA_open','HA_close']].min(axis = 1)
    
    # Technical indicator variables

    data['return'] = data['close'] - data['close'].shift(1) 
    
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = (data['high'] - data['close'].shift(1)).apply(lambda x: abs(x))
    data['tr3'] = (data['low'] - data['close'].shift(1)).apply(lambda x: abs(x))
    data['tr'] = data[['tr1','tr2','tr3']].max(axis = 1)
    data['atr'] = data['tr'].rolling(10).mean()
    
    for i in range(1,13):
        data[f'stock_{i}'] = data['atr'] - data['atr'].shift(i*30)
    
    data['supertrend_14'] = (data['high'].rolling(14).max() + data['low'].rolling(14).min()) / 2 + data['atr'] * 2
    data['supertrend_21'] = (data['high'].rolling(21).max() + data['low'].rolling(21).min()) / 2 + data['atr'] * 1
    
    data['typical_price'] = data[['close','high','low']].mean(axis = 1)
    data['raw_money_flow'] = data['typical_price'] * data['volumefrom']
    data['is_positive_flow'] = data['typical_price'] > data['typical_price'].shift(1)
    data['positive_flow'] = data['is_positive_flow'].apply(lambda x: 1 if x else 0) * data['raw_money_flow']
    data['negative_flow'] = data['is_positive_flow'].apply(lambda x: 0 if x else 1) * data['raw_money_flow']
    data['money_flow_ratio'] = data['positive_flow'].rolling(14).sum() / (data['negative_flow'].rolling(14).sum()+0.000000001)
    data['mfi'] = 100 - 100/(1+data['money_flow_ratio'])
    
    data['u_smma'] = data['return'].apply(lambda x: x if x > 0  else 0)
    data['d_smma'] = data['return'].apply(lambda x: -x if x < 0  else 0)
    u_smma = [np.nan] * 14
    d_smma = [np.nan] * 14
    for i in range(14,len(data)):
        u_smma.append(data['u_smma'].iloc[i-14:i].ewm(alpha = 1/14).mean().iloc[-1])
        d_smma.append(data['d_smma'].iloc[i-14:i].ewm(alpha = 1/14).mean().iloc[-1])
    data['u_smma'] = u_smma
    data['d_smma'] = d_smma
    data['rsi'] = 100 * data['u_smma'] / (data['u_smma'] + data['d_smma'] + 0.000000001)
    
    data['donchian_upper'] = data['high'].rolling(20).max()
    data['donchian_lower'] = data['low'].rolling(20).min()
    
    data['avg_stock'] = data[['stock_1','stock_3','stock_6','stock_12']].mean(axis = 1)

    data = data.drop(columns = ['tr1','tr2','tr3','tr'])
    data = data.drop(columns = ['typical_price','raw_money_flow','is_positive_flow','positive_flow','negative_flow','money_flow_ratio'])
    data = data.drop(columns = ['u_smma','d_smma'])
    return data

def generate_signals(data):
    data['buy_signal'] = data['high'] >= data['donchian_upper']
    data['sell_signal'] = data['low'] <= data['donchian_lower']
    
    # cancel buy signal if the previous day had one
    signals = [data.buy_signal.iloc[0]]
    for i in range(1,len(data)):
        if signals[-1]:
            signals.append(False)
        else:
            signals.append(data.buy_signal.iloc[i])
    data['buy_signal'] = signals

    return data.dropna() # drops 369 rows to increase computational speed 

def signal_returns(data):
    fee = 0.003
    data = data.reset_index()

    signal_returns = []
    for i in range(len(data)):
        row = data.iloc[i]
        if row['buy_signal']:
            buy_price = row.close * (1+fee)
            sells = data.iloc[i:]
            sells = sells[sells['sell_signal']]
            try:
                sell_price = data.iloc[sells.index[0]+1].open * (1-fee)  
                signal_return = (sell_price - buy_price)/buy_price *100
                signal_returns.append(signal_return)
            except:
                signal_returns.append(0)
        else:
            signal_returns.append(0)
    data['signal_return'] = signal_returns

    return data

def sell_returns(data):
    fee = 0.01
    df = pd.DataFrame()
    for i in range(len(data)-1):
        row = data.iloc[i]
        if row.buy_signal:
            buy_price = row.close * (1+fee) 
            future = data.iloc[i+1: min(len(data), i+121)]
            future.loc[:,'sell_return'] = (future['open'] * (1-fee)) / buy_price - 1
            if future.sell_return.max() > 0.1:
                subset_10 = future[future['sell_return'] > 0.1]
                subset_10 = subset_10.sort_values('sell_return',ascending = True).reset_index()
                count = subset_10.index[-1]
                subset_10['reward'] = (subset_10.index)/count + 1
                subset_10.index = subset_10['index']
                subset_10 = subset_10[['reward']]
                future = future.merge(subset_10, how = 'left', left_index = True, right_index = True).fillna(-1)
                df = pd.concat([df, future])
    return df

def normalize_data(data):
    normalized = pd.DataFrame()
    
    for col in ['close','high','low','donchian_upper','donchian_lower']:
        normalized[col] = data[col]/data['open']
    for col in ['HA_close','HA_high','HA_low']:
        normalized[col] = data[col]/data['HA_open']
    
    normalized['atr'] = data['atr'] / (data['atr'].shift(1) + 0.000000001)
    
    norm_min = data.iloc[:,11:23].min(axis=1)
    norm_max = data.iloc[:,11:23].max(axis=1)
    for i in range(1,13):
        normalized[f'stock_{i}'] = (data[f'stock_{i}'] - norm_min) / (norm_max - norm_min + 0.000000001)
    normalized['avg_stock'] = (data['avg_stock'] - norm_min) / (norm_max - norm_min + 0.000000001)
    
    normalized['mfi'] = data['mfi'] * 0.01
    normalized['rsi'] = data['rsi'] * 0.01
    
    normalized['supertrend_14'] = (normalized['high'].rolling(14).max() + normalized['low'].rolling(14).min()) / 2 + normalized['atr'] * 2
    normalized['supertrend_21'] = (normalized['high'].rolling(21).max() + normalized['low'].rolling(21).min()) / 2 + normalized['atr'] * 1
    normalized['return'] = normalized['close'] - normalized['close'].shift(1)

    normalized['buy_signal'] = data['buy_signal']
    normalized['sell_signal'] = data['sell_signal']
    
    return normalized
    
def wrapper_func(symbol,start,end):
    try:
        data = get_data(symbol,start,end)
        data = generate_variables(data)
        data = generate_signals(data)
        buy_data = signal_returns(data)
        sell_data = sell_returns(data)
            
        buy_norm = normalize_data(buy_data)
        buy_norm['signal_return'] = buy_data['signal_return']
        buy_norm = buy_norm[buy_norm['buy_signal']]
        buy_norm = buy_norm.dropna().reset_index(drop=True)
        
        sell_norm = normalize_data(sell_data)
        sell_norm['sell_return'] = sell_data['sell_return']
        sell_norm['reward'] = sell_data['reward']
        sell_norm = sell_norm.dropna().reset_index(drop=True)
        
    except:
        print(f'{symbol} not found')
        return None, None, None
    return data, buy_norm, sell_norm

def donchian_signals(symbol):
    try:
        data = yf.download(symbol, period = '1mo', auto_adjust = False, multi_level_index=False, progress = False)
        data.columns = [col.lower() for col in data.columns.values]
        data['donchian_upper'] = data['high'].rolling(20).max()
        data['donchian_lower'] = data['low'].rolling(20).min()
        data['buy_signal'] = data['high'] >= data['donchian_upper']
        data['sell_signal'] = data['low'] <= data['donchian_lower']
    except:
        print(f'{symbol} not found')
        return None
    return data

def test_wrapper(symbol, start, end):    
    try:
        data = get_data(symbol,start,end)
        data = generate_variables(data)
        data = generate_signals(data)
        return normalize_data(data).dropna()
    except:
        print(f'{symbol} not found')
        return None