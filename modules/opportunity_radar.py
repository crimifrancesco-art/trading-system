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
# ── Strategie disponibili ────────────────────────────────────────────────
STRATEGIES = {
    "RSI+VWAP": "Prezzo vs VWAP e RSI in fascia controllata (trend-following).",
    "ADX+EMA": "Forza del trend (ADX) confermata da EMA20>EMA50.",
    "MACD": "Momentum: crossover MACD e signal line.",
    "Keltner Channel": "Breakout/pullback su bande di volatilità Keltner.",
    "Donchian Channel": "Breakout su massimi/minimi N periodi (turtle-style).",
    "RSI+Bollinger": "Mean-reversion su bande di Bollinger con RSI.",
    "OBV+Hull MA": "Conferma di volume (OBV) con trend Hull MA.",
    "SAR+Chop": "Parabolic SAR filtrato da Choppiness Index (evita lateralità).",
    "ADX+Pattern": "Forza del trend + pattern di prezzo (pullback/breakout).",
}


def tradingview_url(ticker: str) -> str:
    """
    Costruisce l'URL TradingView (versione italiana) per un ticker.
    Gestisce suffissi comuni (.MI per Milano, ecc.) in modo basico.
    """
    symbol = ticker.replace(".MI", "").replace(".", "-")
    return f"https://it.tradingview.com/symbols/{symbol}/"


def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0.0
    minus_dm[minus_dm < 0] = 0.0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=window, adjust=False).mean()
    return adx.fillna(0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    return macd_line, signal_line


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def compute_atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return (atr / close) * 100.0


def evaluate_strategy(strategy: str, df: pd.DataFrame) -> dict:
    """
    Applica la strategia scelta sull'ultimo valore disponibile.
    Restituisce dict con: passed (bool), reasons (list[str]).
    """
    close = df["close"]
    last_close = float(close.iloc[-1])
    reasons = []
    passed = True

    if strategy == "RSI+VWAP":
        vwap = compute_vwap(df)
        rsi = compute_rsi(close)
        price_ok = last_close > vwap.iloc[-1]
        rsi_ok = 45 <= rsi.iloc[-1] <= 65
        passed = price_ok and rsi_ok
        reasons.append("Prezzo sopra VWAP." if price_ok else "Prezzo sotto VWAP.")
        reasons.append(f"RSI {rsi.iloc[-1]:.1f} ({'fascia controllata' if rsi_ok else 'estremo'}).")

    elif strategy == "ADX+EMA":
        adx = compute_adx(df)
        ema20 = compute_ema(close, 20)
        ema50 = compute_ema(close, 50)
        adx_ok = adx.iloc[-1] > 20
        ema_ok = ema20.iloc[-1] > ema50.iloc[-1]
        passed = adx_ok and ema_ok
        reasons.append(f"ADX {adx.iloc[-1]:.1f} ({'trend forte' if adx_ok else 'trend debole'}).")
        reasons.append("EMA20 sopra EMA50." if ema_ok else "EMA20 sotto EMA50.")

    elif strategy == "MACD":
        macd_line, signal_line = compute_macd(close)
        bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
        rising = macd_line.iloc[-1] > macd_line.iloc[-2]
        passed = bullish and rising
        reasons.append("MACD sopra signal line." if bullish else "MACD sotto signal line.")
        reasons.append("Momentum in accelerazione." if rising else "Momentum in decelerazione.")

    elif strategy == "Keltner Channel":
        ema20 = compute_ema(close, 20)
        atr_pct = compute_atr_pct(df)
        atr_abs = (atr_pct / 100.0) * close
        upper = ema20 + 1.5 * atr_abs
        near_upper = last_close >= upper.iloc[-1] * 0.98
        passed = near_upper
        reasons.append("Prezzo vicino/oltre banda superiore Keltner." if near_upper else "Prezzo sotto banda superiore Keltner.")

    elif strategy == "Donchian Channel":
        window = 20
        highest = df["high"].rolling(window).max()
        breakout = last_close >= highest.iloc[-2]
        passed = breakout
        reasons.append("Breakout su massimo Donchian 20 periodi." if breakout else "Nessun breakout Donchian.")

    elif strategy == "RSI+Bollinger":
        upper, ma, lower = compute_bollinger(close)
        rsi = compute_rsi(close)
        near_lower = last_close <= lower.iloc[-1] * 1.02
        rsi_oversold = rsi.iloc[-1] < 40
        passed = near_lower and rsi_oversold
        reasons.append("Prezzo vicino banda inferiore Bollinger." if near_lower else "Prezzo lontano da banda inferiore.")
        reasons.append(f"RSI {rsi.iloc[-1]:.1f} ({'ipervenduto' if rsi_oversold else 'neutro'}).")

    elif strategy == "OBV+Hull MA":
        obv = (np.sign(close.diff()) * df["volume"]).fillna(0).cumsum()
        obv_rising = obv.iloc[-1] > obv.iloc[-5]
        hull = close.rolling(9).mean()  # approssimazione semplificata
        price_above_hull = last_close > hull.iloc[-1]
        passed = obv_rising and price_above_hull
        reasons.append("OBV in aumento (accumulo)." if obv_rising else "OBV in calo (distribuzione).")
        reasons.append("Prezzo sopra media Hull." if price_above_hull else "Prezzo sotto media Hull.")

    elif strategy == "SAR+Chop":
        adx = compute_adx(df)
        choppy = adx.iloc[-1] < 20
        passed = not choppy
        reasons.append("Mercato direzionale (ADX>20)." if not choppy else "Mercato laterale (choppy).")

    elif strategy == "ADX+Pattern":
        adx = compute_adx(df)
        pullback = close.iloc[-1] > close.iloc[-3] and close.iloc[-2] < close.iloc[-3]
        passed = adx.iloc[-1] > 20 and pullback
        reasons.append(f"ADX {adx.iloc[-1]:.1f}.")
        reasons.append("Pattern di pullback rilevato." if pullback else "Nessun pattern di pullback.")

    else:
        passed = False
        reasons.append("Strategia non riconosciuta.")

    return {"passed": passed, "reasons": reasons}
# ── V45.07: Strategie selezionabili e Radar avanzato ─────────────────────
STRATEGIES = {
    "RSI+VWAP": "Prezzo sopra VWAP e RSI in fascia controllata: trend-following selettivo.",
    "ADX+EMA": "Trend con ADX in crescita e struttura EMA20 sopra EMA50.",
    "MACD": "Momentum positivo: MACD sopra signal line e in accelerazione.",
    "Keltner Channel": "Breakout o pressione verso la banda superiore Keltner.",
    "Donchian Channel": "Breakout sui massimi Donchian a 20 periodi.",
    "RSI+Bollinger": "Mean-reversion: prezzo vicino alla banda inferiore con RSI debole.",
    "OBV+Hull MA": "Accumulo su OBV e prezzo sopra Hull MA.",
    "SAR+Chop": "Trend direzionale filtrato contro fasi laterali.",
    "ADX+Pattern": "Trend forte con pullback tecnico confermato.",
}


def tradingview_url(ticker: str) -> str:
    exchange = "MIL" if ticker.upper().endswith(".MI") else "NASDAQ"
    symbol = ticker.upper().replace(".MI", "")
    return f"https://it.tradingview.com/chart/?symbol={exchange}%3A{symbol}"


def compute_atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean()
    return (atr / close.replace(0, np.nan)) * 100.0


def compute_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0.0)


def _macd(close: pd.Series):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + n_std * std, mid, mid - n_std * std


def _safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def evaluate_strategy(strategy: str, df: pd.DataFrame) -> dict:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    volume = df["volume"].astype(float)

    if len(df) < 60:
        return {"passed": False, "reasons": ["Storico insufficiente per la strategia."]}

    last_close = _safe_float(close.iloc[-1])
    rsi = compute_rsi(close)
    vwap = compute_vwap(df)
    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    adx = compute_adx(df)
    atr_pct = compute_atr_pct(df)
    reasons = []

    if strategy == "RSI+VWAP":
        price_ok = last_close > _safe_float(vwap.iloc[-1])
        rsi_ok = 45 <= _safe_float(rsi.iloc[-1]) <= 65
        passed = price_ok and rsi_ok
        reasons = [
            "Prezzo sopra VWAP." if price_ok else "Prezzo sotto VWAP.",
            f"RSI {_safe_float(rsi.iloc[-1]):.1f} ({'fascia controllata' if rsi_ok else 'fuori fascia'}).",
        ]

    elif strategy == "ADX+EMA":
        adx_ok = _safe_float(adx.iloc[-1]) >= 20
        ema_ok = _safe_float(ema20.iloc[-1]) > _safe_float(ema50.iloc[-1])
        passed = adx_ok and ema_ok
        reasons = [
            f"ADX {_safe_float(adx.iloc[-1]):.1f} ({'trend direzionale' if adx_ok else 'trend debole'}).",
            "EMA20 sopra EMA50." if ema_ok else "EMA20 sotto EMA50.",
        ]

    elif strategy == "MACD":
        macd_line, signal_line = _macd(close)
        macd_ok = _safe_float(macd_line.iloc[-1]) > _safe_float(signal_line.iloc[-1])
        rising = _safe_float(macd_line.iloc[-1]) > _safe_float(macd_line.iloc[-2])
        passed = macd_ok and rising
        reasons = [
            "MACD sopra signal line." if macd_ok else "MACD sotto signal line.",
            "Momentum in accelerazione." if rising else "Momentum in rallentamento.",
        ]

    elif strategy == "Keltner Channel":
        atr_abs = (atr_pct / 100.0) * close
        upper_band = ema20 + 1.5 * atr_abs
        near_upper = last_close >= _safe_float(upper_band.iloc[-1]) * 0.98
        passed = near_upper
        reasons = [
            "Prezzo in pressione sulla banda Keltner superiore."
            if near_upper else "Prezzo lontano dalla banda Keltner superiore."
        ]

    elif strategy == "Donchian Channel":
        upper = high.rolling(20).max()
        breakout = last_close >= _safe_float(upper.iloc[-2])
        passed = breakout
        reasons = [
            "Breakout Donchian a 20 periodi." if breakout else "Nessun breakout Donchian."
        ]

    elif strategy == "RSI+Bollinger":
        upper, middle, lower = _bollinger(close)
        near_lower = last_close <= _safe_float(lower.iloc[-1]) * 1.02
        oversold = _safe_float(rsi.iloc[-1]) < 40
        passed = near_lower and oversold
        reasons = [
            "Prezzo vicino alla banda Bollinger inferiore."
            if near_lower else "Prezzo non vicino alla banda Bollinger inferiore.",
            f"RSI {_safe_float(rsi.iloc[-1]):.1f} ({'ipervenduto' if oversold else 'non ipervenduto'}).",
        ]

    elif strategy == "OBV+Hull MA":
        obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        obv_ok = _safe_float(obv.iloc[-1]) > _safe_float(obv.iloc[-5])
        hull_proxy = close.rolling(9).mean()
        hull_ok = last_close > _safe_float(hull_proxy.iloc[-1])
        passed = obv_ok and hull_ok
        reasons = [
            "OBV in accumulo." if obv_ok else "OBV in distribuzione.",
            "Prezzo sopra Hull MA." if hull_ok else "Prezzo sotto Hull MA.",
        ]

    elif strategy == "SAR+Chop":
        adx_ok = _safe_float(adx.iloc[-1]) >= 20
        passed = adx_ok
        reasons = [
            f"ADX {_safe_float(adx.iloc[-1]):.1f}: "
            + ("contesto direzionale." if adx_ok else "contesto laterale/choppy.")
        ]

    elif strategy == "ADX+Pattern":
        adx_ok = _safe_float(adx.iloc[-1]) >= 20
        pullback = (
            _safe_float(close.iloc[-2]) < _safe_float(close.iloc[-3])
            and last_close > _safe_float(close.iloc[-2])
        )
        passed = adx_ok and pullback
        reasons = [
            f"ADX {_safe_float(adx.iloc[-1]):.1f}.",
            "Pullback con ripresa rilevato." if pullback else "Nessun pullback con ripresa.",
        ]

    else:
        passed = False
        reasons = ["Strategia non riconosciuta."]

    return {"passed": passed, "reasons": reasons}


def screen_opportunities_v2(
    strategy: str,
    prices: dict,
    market_caps: pd.Series,
    dollar_volumes: pd.Series,
    earnings_days: pd.Series,
    macro_ok: bool,
    names: dict,
    cfg: dict,
) -> pd.DataFrame:
    rows = []
    min_market_cap = cfg.get("min_market_cap", 10_000_000_000)
    min_dollar_volume_m = cfg.get("min_dollar_volume_millions", 50)
    earnings_buffer = cfg.get("max_earnings_proximity", 5)

    for ticker, df in prices.items():
        required = {"high", "low", "close", "volume"}
        if df is None or df.empty or not required.issubset(df.columns) or len(df) < 60:
            continue

        close = df["close"].astype(float)
        rsi = compute_rsi(close)
        atr_pct = compute_atr_pct(df)
        avg_volume = _safe_float(df["volume"].rolling(20).mean().iloc[-1])
        current_volume = _safe_float(df["volume"].iloc[-1])
        vol_ratio = current_volume / avg_volume if avg_volume else 0.0

        market_cap = _safe_float(market_caps.get(ticker, 0))
        dollar_volume_m = _safe_float(dollar_volumes.get(ticker, 0)) / 1_000_000
        days_to_earnings = _safe_float(earnings_days.get(ticker, 999), 999)

        base_reasons = []
        base_ok = True

        if market_cap < min_market_cap:
            base_ok = False
            base_reasons.append("Market cap sotto soglia large/mega-cap.")
        if dollar_volume_m < min_dollar_volume_m:
            base_ok = False
            base_reasons.append("Dollar volume insufficiente.")
        if days_to_earnings <= earnings_buffer:
            base_ok = False
            base_reasons.append(f"Earnings tra {int(days_to_earnings)} giorni.")
        if not macro_ok:
            base_ok = False
            base_reasons.append("Regime macro critico.")

        strategy_result = evaluate_strategy(strategy, df)
        if not base_ok:
            level = "🔴 Avoid"
            reasons = base_reasons + strategy_result["reasons"]
        elif strategy_result["passed"] and vol_ratio >= 1.0:
            level = "🟢 Opportunity"
            reasons = strategy_result["reasons"] + ["Volume sopra la media."]
        elif strategy_result["passed"]:
            level = "🟡 Watch"
            reasons = strategy_result["reasons"] + ["Volume sotto media."]
        else:
            level = "🔴 Avoid"
            reasons = strategy_result["reasons"]

        quality_score = (
            (3 if market_cap >= 200_000_000_000 else 2 if market_cap >= min_market_cap else 0)
            + (3 if vol_ratio >= 1.5 else 2 if vol_ratio >= 1.0 else 0)
            + (3 if strategy_result["passed"] else 0)
        )

        liquidity_grade = (
            "L3-Institutional" if dollar_volume_m >= 100
            else "L2-Professional" if dollar_volume_m >= 20
            else "L1-Retail"
        )

        rows.append(
            {
                "Ticker": ticker,
                "TradingView": tradingview_url(ticker),
                "Nome": names.get(ticker, ticker),
                "Livello": level,
                "RSI": round(_safe_float(rsi.iloc[-1]), 1),
                "Prezzo": round(_safe_float(close.iloc[-1]), 2),
                "Dollar_Vol_M": round(dollar_volume_m, 1),
                "Quality_Score": quality_score,
                "Liq_Grade": liquidity_grade,
                "ATR_pct": round(_safe_float(atr_pct.iloc[-1]), 2),
                "Vol_Ratio": round(vol_ratio, 2),
                "Days_to_Earnings": int(days_to_earnings) if days_to_earnings < 999 else None,
                "Motivazione": " | ".join(reasons),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        order = {"🟢 Opportunity": 0, "🟡 Watch": 1, "🔴 Avoid": 2}
        result["_rank"] = result["Livello"].map(order).fillna(9)
        result = result.sort_values(
            ["_rank", "Quality_Score", "Vol_Ratio"],
            ascending=[True, False, False],
        ).drop(columns="_rank").reset_index(drop=True)

    return result
