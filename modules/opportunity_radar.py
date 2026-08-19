# modules/opportunity_radar.py
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calcola VWAP cumulativo su DataFrame con colonne:
    ['high', 'low', 'close', 'volume'].
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return vwap


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def screen_opportunities(
    prices: dict[str, pd.DataFrame],
    market_caps: pd.Series,
    dollar_volumes: pd.Series,
    earnings_days: pd.Series,
    macro_ok: bool,
    cfg: dict,
) -> pd.DataFrame:
    """
    prices: dict ticker -> DataFrame con ['open','high','low','close','volume']
    market_caps: Series ticker -> market cap
    dollar_volumes: Series ticker -> dollar volume medio
    earnings_days: Series ticker -> giorni a prossimi earnings
    macro_ok: True se regime macro non critico
    cfg: dizionario con soglie
    """
    results = []

    for ticker, df in prices.items():
        if df.empty:
            continue

        # Indicatori
        vwap = compute_vwap(df)
        ema20 = compute_ema(df["close"], 20)
        ema50 = compute_ema(df["close"], 50)
        rsi = compute_rsi(df["close"], window=14)

        last_price = float(df["close"].iloc[-1])
        last_vwap = float(vwap.iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        last_volume = float(df["volume"].iloc[-1])
        avg_volume = float(df["volume"].rolling(20).mean().iloc[-1])

        # Filtro base: cap + DV + earnings + macro
        mcap = float(market_caps.get(ticker, 0))
        dv = float(dollar_volumes.get(ticker, 0))
        days_earn = float(earnings_days.get(ticker, 999))

        # Helper per costruire reasons
        def append_result(level: str, reasons: list[str]):
            results.append(
                {
                    "ticker": ticker,
                    "level": level,
                    "price": last_price,
                    "vwap": last_vwap,
                    "ema20": last_ema20,
                    "ema50": last_ema50,
                    "rsi": last_rsi,
                    "volume": last_volume,
                    "avg_volume": avg_volume,
                    "market_cap": mcap,
                    "dollar_volume": dv,
                    "days_to_earnings": days_earn,
                    "reasons": reasons,
                }
            )

        # Market cap
        if mcap < cfg.get("min_market_cap", 10_000_000_000):
            append_result(
                "🔴 Avoid",
                ["Market cap inferiore alla soglia large/mega-cap."],
            )
            continue

        # Dollar volume
        if dv < cfg.get("min_dollar_volume", 50_000_000):
            append_result(
                "🔴 Avoid",
                ["Dollar volume insufficiente."],
            )
            continue

        # Earnings proximity
        if days_earn <= cfg.get("max_earnings_proximity", 5):
            append_result(
                "🔴 Avoid",
                [f"Earnings tra {int(days_earn)} giorni (troppo vicini)."],
            )
            continue

        # Macro regime
        if not macro_ok:
            append_result(
                "🔴 Avoid",
                ["Regime macro critico."],
            )
            continue

        # Valutazione tecnica
        price_above_vwap = last_price > last_vwap
        ema20_above_ema50 = last_ema20 > last_ema50
        rsi_min = cfg.get("rsi_min", 45)
        rsi_max = cfg.get("rsi_max", 65)
        rsi_ok_long = rsi_min <= last_rsi <= rsi_max
        vol_above_avg = last_volume > avg_volume

        reasons = []

        if price_above_vwap:
            reasons.append("Prezzo sopra VWAP.")
        else:
            reasons.append("Prezzo sotto VWAP.")

        if ema20_above_ema50:
            reasons.append("EMA20 sopra EMA50.")
        else:
            reasons.append("EMA20 sotto EMA50.")

        reasons.append(
            f"RSI a {last_rsi:.1f} "
            f"({'fascia controllata' if rsi_ok_long else 'fascia estrema'}).",
        )

        if vol_above_avg:
            reasons.append("Volume sopra la media.")
        else:
            reasons.append("Volume nella media o sotto.")

        # Assegnazione livello
        if price_above_vwap and ema20_above_ema50 and rsi_ok_long and vol_above_avg:
            level = "🟢 Opportunity"
        elif (
            price_above_vwap
            and ema20_above_ema50
            and (cfg.get("rsi_min", 45) <= last_rsi <= cfg.get("rsi_max", 70))
        ):
            level = "🟡 Watch"
        else:
            level = "🔴 Avoid"

        append_result(level, reasons)

    return pd.DataFrame(results)
def prepare_radar_data_for_demo() -> tuple:
    """
    Funzione temporanea per testare l'integrazione del Radar.
    Restituisce:
      prices, market_caps, dollar_volumes, earnings_days, macro_ok, cfg
    """
    # Esempio con 3 ticker
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    np.random.seed(42)

    tickers = ["AAPL", "MSFT", "NVDA"]
    prices = {}
    for t in tickers:
        close = 100 + np.cumsum(np.random.randn(120))
        high = close + np.abs(np.random.randn(120))
        low = close - np.abs(np.random.randn(120))
        open_ = close + np.random.randn(120)
        volume = 1_000_000 + np.random.randint(0, 500_000, size=120)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        prices[t] = df

    market_caps = pd.Series(
        {
            "AAPL": 2_500_000_000_000,
            "MSFT": 2_800_000_000_000,
            "NVDA": 1_200_000_000_000,
        }
    )

    dollar_volumes = pd.Series(
        {
            "AAPL": 80_000_000,
            "MSFT": 70_000_000,
            "NVDA": 60_000_000,
        }
    )

    earnings_days = pd.Series(
        {
            "AAPL": 20,
            "MSFT": 15,
            "NVDA": 3,  # vicino agli earnings
        }
    )

    macro_ok = True

    cfg = {
        "min_market_cap": 10_000_000_000,
        "min_dollar_volume": 50_000_000,
        "max_earnings_proximity": 5,
        "rsi_min": 45,
        "rsi_max": 65,
    }

    return prices, market_caps, dollar_volumes, earnings_days, macro_ok, cfg