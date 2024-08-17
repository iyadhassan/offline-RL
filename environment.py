import gym
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        
        # Explicitly define Discrete action space
        self.action_space = gym.spaces.Discrete(5)
        
        # Observation space: 8 features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.total_reward = 0
        self.position = 0 
    
    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.position = 0
        return self._get_observation()
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        price_change = (next_price - current_price) / current_price
        
        # Execute trade
        old_position = self.position
        if action == 0:  # Strong Sell
            self.position = max(-2, self.position - 1)
        elif action == 1:  # Sell
            self.position = max(-2, self.position - 1)
        elif action == 2:  # Hold
            pass
        elif action == 3:  # Buy
            self.position = min(2, self.position + 1)
        elif action == 4:  # Strong Buy
            self.position = min(2, self.position + 1)
        
        # Calculate reward
        if old_position != self.position:
            transaction_cost = 0.001 * abs(self.position - old_position)
        else:
            transaction_cost = 0
        
        reward = self.position * price_change - transaction_cost
        
        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'Returns']].values.astype(np.float32)