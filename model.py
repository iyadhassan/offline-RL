import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig

def create_cql_model(env, device):
    config = DiscreteCQLConfig(
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        n_critics=2,
        bootstrap=True,
        share_encoder=False,
        reward_scaler=None,
        target_update_interval=100
    )
    
    model = config.create(device=device)
    model.build_with_env(env)
    
    return model

def create_dataset(env, n_episodes=100):
    observations = []
    actions = []
    rewards = []
    terminals = []

    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, _ = env.step(action)
            
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)
            
            observation = next_observation

    return MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),  # Ensure actions are integers
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool)
    )

def create_cql_model(env, device):
    config = DiscreteCQLConfig(
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        n_critics=2,
        bootstrap=True,
        share_encoder=False,
        reward_scaler=None,
        target_update_interval=100
    )
    
    model = config.create(device=device)
    model.build_with_env(env)
    
    return model