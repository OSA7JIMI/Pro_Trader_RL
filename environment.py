import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BuyKnowledge(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data = pd.DataFrame(), train_mode = True):
        super().__init__()
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (2,), dtype = np.float64)
        self.observation_space = spaces.Box(low = -1.0, high = 100.0, shape = (27,), dtype = np.float64)
        
        self.train_mode = train_mode
        
        if self.train_mode:
            data = pd.read_csv('buy_norm.csv')
            
            above = data[data['signal_return'] > 10]
            below = data[data['signal_return'] <= 10]
            frac = len(above)/len(below)
            if frac >1:
                frac = 1/frac
                data = pd.concat([above.sample(frac = frac), below]).sort_index().reset_index(drop=True)
            else:
                data = pd.concat([above, below.sample(frac = frac)]).sort_index().reset_index(drop=True)
                
            self.returns = data['signal_return'] > 10
            
            # max_reward = len(data[data['signal_return']<=10]) + len(data[data['signal_return']>10])*3
            # print(f'Maximum reward: {max_reward}')
            
            data = data.iloc[:,:-3]
        
        self.data = data
        self.num_steps = len(data)
        print(f'# of steps: {self.num_steps}')
        
    def step(self, action):
        if self.train_mode:
            if action[0] > action[1]:
                if self.returns.iloc[self.cur_step]:
                    reward = 1
                else:
                    reward = 0
            else:
                if self.returns.iloc[self.cur_step]:
                    reward = 0
                else:
                    reward = 1
        else:
            reward = 1
                
        self.cur_step += 1
        return self.data.iloc[self.cur_step], reward, self.cur_step == self.num_steps -1, False, {}

    def reset(self, seed = None, options = None):
        self.cur_step= 0
        return self.data.iloc[0], {}

    def render(self):
        pass
        
    def close(self):
        pass

class SellKnowledge(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data = pd.DataFrame(), train_mode = True):
        super().__init__()
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (2,), dtype = np.float64)
        self.observation_space = spaces.Box(low = -1.0, high = 100.0, shape = (28,), dtype = np.float64)
        
        self.train_mode = train_mode

        if self.train_mode:
            data = pd.read_csv('sell_norm.csv')
            data = data.sample(frac = 1).reset_index(drop=True)
            self.rewards = data.reward
            max_reward = len(data[data['reward']<=0]) * 0.5 + data[data['reward']>0].reward.sum()
            print(f'Maximum reward: {max_reward}')
            data = data.drop(['buy_signal','sell_signal','reward'],axis = 1)
        
        self.data = data
        self.num_steps = len(data)
        print(f'# of steps: {self.num_steps}')
        
    def step(self, action):
        if self.train_mode:
            if action[0] > action[1]:
                reward = self.rewards.iloc[self.cur_step]
            else:
                if self.data.sell_return.iloc[self.cur_step] > 10:
                    reward = -1
                else:
                    reward = 0.5
        else:
            reward = 1
                
        self.cur_step += 1
        return self.data.iloc[self.cur_step], reward, self.cur_step == self.num_steps -1, False, {}

    def reset(self, seed = None, options = None):
        self.cur_step= 0
        return self.data.iloc[0], {}

    def render(self):
        pass
        
    def close(self):
        pass