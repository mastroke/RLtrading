from Env2 import TradingEnv

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['close', 'open', 'high', 'low']].to_numpy()[start:end]

    timeStamps = env.df.loc[:, 'timestamp']
    timeStamps = timeStamps[env.frame_bound[0] - env.window_size:env.frame_bound[1]]

    return prices, signal_features,timeStamps

class MyForexEnv(TradingEnv):
    _process_data = my_process_data