import numpy as np

def backtest_simple(stocks, vector_sum):
    stocks = stocks.dropna(axis = 1, how = "all")

    data_cutoff = stocks.reset_index().isna().any(axis = 1)[::-1].idxmax() - len(stocks) + 1
    stocks = stocks.iloc[data_cutoff:].values
    stocks = 1 + (stocks[1:, :] - stocks[:-1, :]) / (stocks[:-1, :] + 1e-8) * abs(vector_sum[None, :]) / (vector_sum[None, :] + 1e-8)

    stocks = np.vstack((abs(vector_sum[None, :]), stocks))

    performance = np.cumprod(stocks, axis = 0)

    return np.array([stock for stock in performance.T if stock.any()])