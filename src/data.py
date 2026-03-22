"""
Data pipeline: download DJIA stocks, compute technical indicators, split chronologically.

Features computed per ticker:
  - Daily return (pct_change)
  - Price deviation from short/long-window moving averages (configurable, default 5/20)
  - RSI-14 (Relative Strength Index)
  - MACD signal (12/26/9 EMA crossover)
  - Bollinger Band width (long-window, 2-sigma)
  - Normalised volume (volume / long-window MA volume)
  - ATR (Average True Range, normalised by close)
  - OBV (On-Balance Volume, normalised by rolling MA)

Features are z-score normalised using training-set statistics only.
Raw returns and close prices are kept as separate columns for reward/portfolio calculation.
"""
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "GS", "HD", "MCD", "V", "DIS"]
START = "2009-01-01"
END = "2024-12-31"

# Features used by the agent network (will be z-score normalised)
FEATURE_COLS = [
    "return", "dev5", "dev20", "rsi", "macd_signal", "bb_width", "vol_norm",
    "atr", "obv_norm",
]


def download(tickers=TICKERS, start=START, end=END):
    """Download daily OHLCV data from Yahoo Finance.

    Returns (close, high, low, volume) DataFrames with ticker columns.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle yfinance multi-level column index for single-ticker lists
    if len(tickers) == 1:
        close = raw["Close"].dropna()
        high = raw["High"].dropna()
        low = raw["Low"].dropna()
        volume = raw["Volume"].dropna()
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
            high = high.to_frame(name=tickers[0])
            low = low.to_frame(name=tickers[0])
            volume = volume.to_frame(name=tickers[0])
        elif hasattr(close.columns, 'droplevel'):
            try:
                close.columns = close.columns.droplevel(0)
                high.columns = high.columns.droplevel(0)
                low.columns = low.columns.droplevel(0)
                volume.columns = volume.columns.droplevel(0)
            except Exception:
                pass
    else:
        close = raw["Close"].dropna()
        high = raw["High"].dropna()
        low = raw["Low"].dropna()
        volume = raw["Volume"].dropna()

    return close, high, low, volume


def _compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)


def _compute_macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Compute MACD signal line crossover (MACD - signal line)."""
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # histogram


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range, normalised by close price."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / (close + 1e-8)


def _compute_obv(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute On-Balance Volume, normalised by its rolling MA."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(period).mean()
    return obv / (obv_ma.abs() + 1e-8)


def _compute_bollinger_width(price: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Compute Bollinger Band width normalised by the middle band."""
    ma = price.rolling(period).mean()
    std = price.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / (ma + 1e-8)


def compute_features(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """Compute feature DataFrame for each ticker.

    Parameters
    ----------
    short_window : int
        Short-term lookback for MA deviation (default 5).
    long_window : int
        Long-term lookback for MA deviation, Bollinger, and volume MA (default 20).
    """
    result = {}

    for ticker in close.columns:
        p = close[ticker].dropna()
        h = high[ticker].dropna() if ticker in high.columns else p
        lo = low[ticker].dropna() if ticker in low.columns else p
        v = volume[ticker].dropna() if ticker in volume.columns else pd.Series(0, index=p.index)

        daily_return = p.pct_change()

        # Moving average deviations (configurable windows)
        ma_short = p.rolling(short_window).mean()
        ma_long = p.rolling(long_window).mean()
        dev5 = (p - ma_short) / (ma_short + 1e-8)
        dev20 = (p - ma_long) / (ma_long + 1e-8)

        # Technical indicators
        rsi = _compute_rsi(p) / 100.0  # scale to [0, 1]
        macd_signal = _compute_macd(p)
        bb_width = _compute_bollinger_width(p, period=long_window)

        # Volume normalised by its long-window moving average
        vol_ma = v.rolling(long_window).mean()
        vol_norm = v / (vol_ma + 1e-8)

        # ATR and OBV
        atr = _compute_atr(h, lo, p)
        obv_norm = _compute_obv(p, v, period=long_window)

        df = pd.DataFrame({
            "close": p,
            "raw_return": daily_return,
            "return": daily_return,
            "dev5": dev5,
            "dev20": dev20,
            "rsi": rsi,
            "macd_signal": macd_signal,
            "bb_width": bb_width,
            "vol_norm": vol_norm,
            "atr": atr,
            "obv_norm": obv_norm,
        }).dropna()

        result[ticker] = df

    return result


def split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Chronological split to avoid look-ahead bias."""
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def normalise(train, val, test):
    """Z-score normalise FEATURE_COLS using training-set statistics only."""
    mean = train[FEATURE_COLS].mean()
    std = train[FEATURE_COLS].std() + 1e-8
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train[FEATURE_COLS] = (train[FEATURE_COLS] - mean) / std
    val[FEATURE_COLS] = (val[FEATURE_COLS] - mean) / std
    test[FEATURE_COLS] = (test[FEATURE_COLS] - mean) / std
    return train, val, test


def load(ticker="AAPL", short_window=5, long_window=20):
    """Download data for a single ticker, compute features, split, and normalise."""
    close, high, low, volume = download([ticker])
    feats = compute_features(close, high, low, volume,
                             short_window=short_window, long_window=long_window)
    df = feats[ticker]
    train, val, test = split(df)
    return normalise(train, val, test)


def load_all(tickers=TICKERS, short_window=5, long_window=20):
    """Download all tickers at once, compute features, split, and normalise each.

    Returns dict[ticker, (train_df, val_df, test_df)].
    """
    close, high, low, volume = download(tickers)
    feats = compute_features(close, high, low, volume,
                             short_window=short_window, long_window=long_window)
    result = {}
    for ticker in feats:
        train, val, test = split(feats[ticker])
        result[ticker] = normalise(train, val, test)
    return result


if __name__ == "__main__":
    train, val, test = load("AAPL")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Features: {FEATURE_COLS}")
    print(train[FEATURE_COLS].describe().round(3))
