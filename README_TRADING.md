# Simple Trading JSON Output

Uses existing `finrl.meta.paper_trading.alpaca.PaperTradingAlpaca` class to get model predictions and output as JSON.

## Files
- `paper_trading_json.py` - Extends `PaperTradingAlpaca` to output JSON
- `main.py` - Entry point with Redis integration

## Redis Integration

### Step 1: Prepare your Redis data

Your Redis server should provide data for all 30 tickers in this format:

```python
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
        'close_60_sma': 149.8,
        'VIXY': 20.5
    },
    'MSFT': { ... },
    ... (all 30 tickers including VIXY)
}
```

### Step 2: Add your Redis connection in `main.py`

Replace the `fetch_redis_data()` function:

```python
def fetch_redis_data():
    import redis
    import json
    
    # Connect to your Redis server
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Fetch data (adjust key name as needed)
    raw_data = redis_client.get('market_data_key')
    
    # Parse JSON
    market_data = json.loads(raw_data)
    
    return market_data
```

### Step 3: Run

```bash
python main.py
```

## Data Validation

The system automatically validates:
- ✅ All 30 tickers present (as defined in TICKER_LIST)
- ✅ OHLCV data: `open`, `high`, `low`, `close`, `volume`
- ✅ Technical indicators (8): `macd`, `boll_ub`, `boll_lb`, `rsi_30`, `cci_30`, `dx_30`, `close_30_sma`, `close_60_sma`
- ✅ Correct column order (as per FinRL's processor_alpaca.py)

## Output Format

```json
{
  "buy": {
    "AAPL": 25,
    "MSFT": 15
  },
  "sell": {
    "NVDA": 30
  },
  "timestamp": "2025-11-19 14:30:00"
}
```

## How It Works

1. Fetches data from Redis server
2. Validates and formats into DataFrame
3. Uses `PaperTradingAlpaca` to:
   - Get current portfolio from Alpaca API
   - Prepare state vector (333 dims)
   - Get model predictions
   - Apply trading logic (min_action=10, turbulence check)
4. Returns buy/sell decisions as JSON (no actual orders executed)