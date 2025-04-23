# A modified implementation of the Pro Trader RL stock trading system proposed by Da Woon Jeong and Yeong Hyeon Gu

Source: https://www.sciencedirect.com/science/article/pii/S0957417424013319#s003

## Disclaimer: This repository should not be taken as financial advice. Please read the source paper and conduct your own research on stock trading and reinforcement learning before proceeding.

## Usage guide

### Required modules:
1. gymnasium
2. numpy
3. pandas
4. stable-baselines3
5. yfinance

### Steps:
1. Install the required modules
2. Download the repository
3. Set the variables named "fee" in preprocessing.py to match the fee % on your exchange (default is 0.1%)
4. Set your definition of a good return in environment.py and pro_trader.ipynb (default is 10%)
5. Run get_training_data.ipynb using the S&P 500 dataset (or the S&P 400 if you prefer)
6. Run the model training notebooks with your desired hyperparameters and number of epochs
7. Run pro_trader.ipynb (convert to .py for use in task scheduler)
