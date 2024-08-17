import torch
from data_loader import load_data
from environment import StockTradingEnv
from model import create_cql_model, create_dataset
from train import train_cql, load_cql_model
from test import test_model

def main(symbol, start_date, end_date, n_train_steps, n_episodes, device='cuda:0'):
    # Load and preprocess data
    train_data, test_data, _ = load_data(symbol, start_date, end_date)
    print("*")
    # Create environments
    train_env = StockTradingEnv(train_data)
    test_env = StockTradingEnv(test_data)
    print("*")
    # Create CQL model
    model = create_cql_model(train_env, device)
    print("*")
    # Create dataset
    dataset = create_dataset(train_env, n_episodes=n_episodes)
    print("*")
    # Train model
    trained_model = train_cql(model, train_env, dataset, n_train_steps)
    print("*")
    # Backtest model
    backtest_results, statistics = test_model(trained_model, test_env, test_data[['Close']])
    print("*")
    return trained_model, backtest_results, statistics

if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-07-31"
    n_train_steps = 10000
    n_episodes = 100
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    trained_model, backtest_results, statistics = main(symbol, start_date, end_date, n_train_steps, n_episodes, device)
if __name__ == "__main__":
    symbol = "AAPL"  # Stock symbol (e.g., AAPL for Apple)
    start_date = "2010-01-01"
    end_date = "2023-07-31"
    n_train_steps = 100000  # Adjust this for longer/shorter training
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    trained_model, backtest_results, statistics = main(symbol, start_date, end_date, n_train_steps, device)