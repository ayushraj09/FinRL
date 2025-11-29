"""
Production entry point - Uses existing PaperTradingAlpaca class.
"""

from paper_trading import get_trading_decisions
from finrl.config import INDICATORS
import pandas as pd
import json

# Configuration
API_KEY = "PKPGA7BQIFZ7UV3V5ZYUWEXPUY"
API_SECRET = "HRvDc53DYAP2gJZbxn71MRCyZYnm5G5PFpCcAbhipf8Y"
API_BASE_URL = 'https://paper-api.alpaca.markets'

TICKER_LIST = [
    "AXP", "AMGN", "AAPL", "AMZN", "BA", "CAT", "CSCO", "CVX", "GS", "HD", 
    "HON", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", 
    "NKE", "NVDA", "PG", "UNH", "V", "VZ", "WMT", "DIS", "DOW", "VIXY"
]

MODEL_PATH = "agent_ppo.zip"


def validate_and_format_redis_data(redis_raw_data: dict) -> pd.DataFrame:
    """
    Validate and format Redis data into required DataFrame structure.
    
    Expected Redis data format (from your friend's server):
    {
        'AAPL': {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.5,
            'volume': 1000000,
            'macd': 1.23,
            'boll_ub': 155.0,
            'boll_lb': 148.0,
            'rsi_30': 65.4,
            'cci_30': 25.6,
            'dx_30': 18.7,
            'close_30_sma': 150.2,
            'close_60_sma': 149.8
        },
        ... (for all 30 tickers)
        'VIXY': {
            'close': 25.4,
            ...
        }
    }
    
    Returns:
        DataFrame with columns: ['tic', 'open', 'high', 'low', 'close', 'volume'] + INDICATORS
    """
    # Required columns in correct order (from processor_alpaca.py line 144, 489)
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    REQUIRED_COLUMNS = ['tic'] + OHLCV_COLUMNS + INDICATORS
    
    # Validate all tickers are present
    if len(redis_raw_data) != len(TICKER_LIST):
        raise ValueError(f"Expected {len(TICKER_LIST)} tickers, got {len(redis_raw_data)}")
    
    for ticker in TICKER_LIST:
        if ticker not in redis_raw_data:
            raise ValueError(f"Missing ticker in Redis data: {ticker}")
    
    # Build DataFrame
    data_rows = []
    for ticker in TICKER_LIST:
        ticker_data = redis_raw_data[ticker]
        
        row = {'tic': ticker}
        
        # Validate and add OHLCV data
        for col in OHLCV_COLUMNS:
            if col not in ticker_data:
                raise ValueError(f"Missing {col} for ticker {ticker}")
            row[col] = ticker_data[col]
        
        # Validate and add technical indicators
        for indicator in INDICATORS:
            if indicator not in ticker_data:
                raise ValueError(f"Missing indicator {indicator} for ticker {ticker}")
            row[indicator] = ticker_data[indicator]
        
        data_rows.append(row)
    
    # Create DataFrame with correct column order
    df = pd.DataFrame(data_rows, columns=REQUIRED_COLUMNS)
    
    print(f"✅ Validated Redis data: {len(df)} tickers with {len(REQUIRED_COLUMNS)} columns")
    return df


def fetch_redis_data():
    """
    TODO: Replace with your friend's Redis data fetching code.
    
    Integration point - Add your Redis server connection here.
    
    Example integration:
    ```python
    import redis
    import json
    
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    raw_data = redis_client.get('market_data_key')
    market_data = json.loads(raw_data)
    return market_data
    ```
    
    Expected return format (dict):
    {
        'AAPL': {'open': 150.0, 'high': 152.0, 'low': 149.0, 'close': 151.5, 
                 'volume': 1000000, 'macd': 1.23, 'boll_ub': 155.0, ...},
        'MSFT': {...},
        ... (all 30 tickers)
    }
    """
    # YOUR REDIS CODE HERE
    # Example:
    # import redis
    # redis_client = redis.Redis(host='localhost', port=6379, db=0)
    # raw_data = redis_client.get('market_data')
    # return json.loads(raw_data)
    
    raise NotImplementedError("Add your Redis data fetching code here")


def main():
    """Get trading decisions for current minute."""
    # Get data from Redis
    redis_raw_data = fetch_redis_data()
    
    # Validate and format into DataFrame
    redis_data = validate_and_format_redis_data(redis_raw_data)
    
    # Get trading decisions
    decisions = get_trading_decisions(
        redis_data=redis_data,
        model_path=MODEL_PATH,
        ticker_list=TICKER_LIST,
        tech_indicators=INDICATORS,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_base_url=API_BASE_URL
    )
    
    return decisions


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
