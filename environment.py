import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BuyKnowledge(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data = pd.DataFrame(), train_mode = True, sampling = True):
        super().__init__()
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (2,), dtype = np.float64)
        self.observation_space = spaces.Box(low = -1.0, high = 100.0, shape = (27,), dtype = np.float64)
        
        self.train_mode = train_mode
        
        if self.train_mode:
            data = pd.read_csv('buy_norm.csv')

            above = data[data['signal_return'] > 10]
            below = data[data['signal_return'] <= 10]
            frac = len(above)/len(below)
            
            if sampling: # buy signals with a return of 10 % or more are used in the same proportion as those that do not achieve this return [Jeong & Gu]
                data = pd.concat([above.sample(frac = frac), below]).sort_index().reset_index(drop=True)
            else: 
                penalty_big = (1-2*frac)/(frac-1) * 0.85
                reward_big = int((1-frac)*(1+penalty_big)/frac) + 1
                self.penalty_big = penalty_big
                self.reward_big = reward_big
                print(f'Penalty: {penalty_big}')
                print(f'Reward: {reward_big}')
                print(f'Maximum reward: {len(below) + reward_big*len(above)}')
                
            data = data.sample(frac=1) 
            self.returns = data['signal_return'] > 10
            data = data.iloc[:,:-3]
                    
        self.data = data
        self.sampling = sampling
        self.num_steps = len(data)
        print(f'# of steps: {self.num_steps}')
        
    def step(self, action):
        if self.train_mode:
            if action[0] > action[1]:
                if self.sampling:
                    if self.returns.iloc[self.cur_step]:
                        reward = 1
                    else:
                        reward = 0
                else:
                    if self.returns.iloc[self.cur_step]:
                        reward = self.reward_big
                    else:
                        reward = self.penalty_big
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
                if self.data.sell_return.iloc[self.cur_step] > 0.1:
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

def run_agent(env, model):
    obs,info = env.reset()
    actions = []
    done = False
    while not done:
        try:
            action, _states = model.predict(pd.to_numeric(obs))
            actions.append(action)
        except:
            print(f'Prediction failed at {env.cur_step}')
            actions.append([-1,-1])
        obs, reward, done, trunc, info = env.step(action)

    return actions