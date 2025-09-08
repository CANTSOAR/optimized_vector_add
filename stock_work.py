import numpy as np
import pandas as pd
import requests
import yfinance as yf

def get_data(ticker_list, interval = None, start = None, end = None):
    interval = interval if interval else "1d"
    if interval and start and end:
        data = yf.download(ticker_list, interval = interval, start = start, end = end, group_by = "ticker")
    else:
        data = yf.download(ticker_list, interval = interval, period = "max", group_by = "ticker")

    closes = data.loc[:, (slice(None), "Close")]
    closes.columns = [ticker for ticker, _ in closes.columns]

    volumes = data.loc[:, (slice(None), "Volume")]
    volumes.columns = [ticker for ticker, _ in volumes.columns]

    return closes, volumes

def get_factors(interval = None, start = None, end = None):
    """
    XLC: Communication Services
    XLY: Consumer Discretionary
    XLP: Consumer Staples
    XLE: Energy
    XLF: Financials
    XLV: Health Care
    XLI: Industrials
    XLB: Materials
    XLRE: Real Estate
    XLK: Technology
    XLU: Utilities
    """
    return get_data(['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU'], interval, start, end)

def regression_results(stocks, factors):
    stocks = stocks.dropna(axis = 1, how = "all")

    stock_data_cutoff = stocks.reset_index().isna().any(axis = 1)[::-1].idxmax() - len(stocks) + 1
    factor_data_cutoff = factors.reset_index().isna().any(axis = 1)[::-1].idxmax() - len(factors) + 1
    data_cutoff = max(stock_data_cutoff, factor_data_cutoff)

    stocks = stocks.iloc[data_cutoff:].values
    stocks = (stocks[1:, :] - stocks[:-1, :]) / (stocks[:-1, :] + 1e-8)

    factors = factors.iloc[data_cutoff:].values
    factors = (factors[1:, :] - factors[:-1, :]) / (factors[:-1, :] + 1e-8)
    factors = np.hstack((np.ones((len(factors), 1)), factors))

    loadings = np.linalg.inv(factors.T @ factors) @ factors.T @ stocks
    resids = stocks - factors @ loadings

    alphas = loadings[0, :]
    betas = loadings[1:, :]
    idios = np.diag(np.sum(resids, axis = 0) ** 2 / (len(factors) - len(factors[0]) - 1))
    covariance = np.cov(factors[:, 1:], rowvar = False)

    return alphas, betas, idios, covariance

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    }

    try:
        response = requests.post(url, headers = headers)
    except:
        return []
    
    tables = pd.read_html(response.text, attrs = {"id": "constituents"})
    sp500_table = tables[0]

    tickers = sp500_table["Symbol"].tolist()
    return tickers