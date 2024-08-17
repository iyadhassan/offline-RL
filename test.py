import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import stats

def backtest_model(model, env, initial_balance=10000):
    observation = env.reset()
    done = False
    balance = initial_balance
    positions = []
    portfolio_values = []
    dates = []
    
    while not done:
        action = model.predict([observation])[0]
        observation, reward, done, _ = env.step(action)
        
        current_price = env.data.iloc[env.current_step]['Close']
        date = env.data.iloc[env.current_step].name
        
        position = env.position / 2  # Normalize position to [-1, 1] range
        
        positions.append(position)
        portfolio_value = balance * (1 + position * (current_price / env.data.iloc[0]['Close'] - 1))
        portfolio_values.append(portfolio_value)
        dates.append(date)
        
        balance = portfolio_value
    
    return pd.DataFrame({
        'Date': dates,
        'Position': positions,
        'Portfolio Value': portfolio_values
    }).set_index('Date')

def calculate_statistics(backtest_results, benchmark_returns):
    portfolio_returns = backtest_results['Portfolio Value'].pct_change().dropna()
    
    total_return = (backtest_results['Portfolio Value'].iloc[-1] / backtest_results['Portfolio Value'].iloc[0]) - 1
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    
    beta, alpha, _, _, _ = stats.linregress(benchmark_returns.values, portfolio_returns.values)
    
    max_drawdown = (backtest_results['Portfolio Value'] / backtest_results['Portfolio Value'].cummax() - 1).min()
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Beta': beta,
        'Alpha': alpha,
        'Max Drawdown': max_drawdown
    }

def plot_backtest_results(backtest_results, benchmark_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Portfolio Value vs Benchmark
    ax1.plot(backtest_results.index, backtest_results['Portfolio Value'], label='Portfolio Value')
    ax1.plot(benchmark_data.index, benchmark_data['Close'], label='Benchmark (Buy & Hold)')
    ax1.set_ylabel('Value')
    ax1.set_title('Portfolio Value vs Benchmark')
    ax1.legend()
    
    # Position
    ax2.plot(backtest_results.index, backtest_results['Position'])
    ax2.set_ylabel('Position')
    ax2.set_title('Trading Positions (1: Long, 0: Hold, -1: Short)')
    
    plt.xlabel('Date')
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def test_model(model, env, benchmark_data):
    backtest_results = backtest_model(model, env)
    
    # Align benchmark data with backtest results
    aligned_benchmark = benchmark_data.reindex(backtest_results.index)
    benchmark_returns = aligned_benchmark['Close'].pct_change().dropna()
    
    statistics = calculate_statistics(backtest_results, benchmark_returns)
    
    print("Backtest Statistics:")
    for key, value in statistics.items():
        print(f"{key}: {value:.4f}")
    
    plot_backtest_results(backtest_results, aligned_benchmark)
    
    return backtest_results, statistics