from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


ASSET_UNIVERSE = {
    # ── Commodity ─────────────────────────────────────────────────────────
    "XAUUSD": {
        "name": "Gold Spot",
        "category": "Commodity",
        "group": "Metalli preziosi",
        "yfinance": "GC=F",
        "tradingview": "OANDA:XAUUSD",
        "alt_tradingview": "TVC:GOLD",
        "currency": "USD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "FOMC / CPI / NFP",
    },
    "XAGUSD": {
        "name": "Silver Spot",
        "category": "Commodity",
        "group": "Metalli preziosi",
        "yfinance": "SI=F",
        "tradingview": "OANDA:XAGUSD",
        "alt_tradingview": "TVC:SILVER",
        "currency": "USD",
        "max_leverage": 2.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "FOMC / CPI / NFP",
    },
    "USOIL": {
        "name": "WTI Crude Oil",
        "category": "Commodity",
        "group": "Energia",
        "yfinance": "CL=F",
        "tradingview": "TVC:USOIL",
        "alt_tradingview": "NYMEX:CL1!",
        "currency": "USD",
        "max_leverage": 2.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "EIA petroleum / OPEC+",
    },
    "UKOIL": {
        "name": "Brent Crude Oil",
        "category": "Commodity",
        "group": "Energia",
        "yfinance": "BZ=F",
        "tradingview": "TVC:UKOIL",
        "alt_tradingview": "ICEEUR:BRN1!",
        "currency": "USD",
        "max_leverage": 2.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "EIA petroleum / OPEC+",
    },
    "NATGAS": {
        "name": "Natural Gas",
        "category": "Commodity",
        "group": "Energia",
        "yfinance": "NG=F",
        "tradingview": "NYMEX:NG1!",
        "alt_tradingview": "TVC:NATGAS",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 3.0,
        "point_value": 1.0,
        "event": "EIA natural gas storage",
    },
    "COPPER": {
        "name": "Copper",
        "category": "Commodity",
        "group": "Metalli industriali",
        "yfinance": "HG=F",
        "tradingview": "COMEX:HG1!",
        "alt_tradingview": "TVC:COPPER",
        "currency": "USD",
        "max_leverage": 2.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "China PMI / USD",
    },
    # ── Forex ─────────────────────────────────────────────────────────────
    "EURUSD": {
        "name": "EUR / USD",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "EURUSD=X",
        "tradingview": "FX:EURUSD",
        "alt_tradingview": "OANDA:EURUSD",
        "currency": "USD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "FOMC / ECB / CPI / NFP",
    },
    "GBPUSD": {
        "name": "GBP / USD",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "GBPUSD=X",
        "tradingview": "FX:GBPUSD",
        "alt_tradingview": "OANDA:GBPUSD",
        "currency": "USD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "FOMC / BoE / CPI / NFP",
    },
    "USDJPY": {
        "name": "USD / JPY",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "JPY=X",
        "tradingview": "FX:USDJPY",
        "alt_tradingview": "OANDA:USDJPY",
        "currency": "JPY",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "FOMC / BoJ / CPI / NFP",
    },
    # ── Crypto spot ───────────────────────────────────────────────────────
    "BTCUSD": {
        "name": "Bitcoin",
        "category": "Crypto",
        "group": "Crypto large cap",
        "yfinance": "BTC-USD",
        "tradingview": "BITSTAMP:BTCUSD",
        "alt_tradingview": "BINANCE:BTCUSDT",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "ETHUSD": {
        "name": "Ethereum",
        "category": "Crypto",
        "group": "Crypto large cap",
        "yfinance": "ETH-USD",
        "tradingview": "BITSTAMP:ETHUSD",
        "alt_tradingview": "BINANCE:ETHUSDT",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "SOLUSD": {
        "name": "Solana",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "SOL-USD",
        "tradingview": "BINANCE:SOLUSDT",
        "alt_tradingview": "COINBASE:SOLUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "PLATINUM": {
        "name": "Platinum",
        "category": "Commodity",
        "group": "Metalli preziosi",
        "yfinance": "PL=F",
        "tradingview": "TVC:PLATINUM",
        "alt_tradingview": "NYMEX:PL1!",
        "currency": "USD",
        "max_leverage": 2.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 1.0,
        "event": "USD / domanda industriale",
    },
    "PALLADIUM": {
        "name": "Palladium",
        "category": "Commodity",
        "group": "Metalli preziosi",
        "yfinance": "PA=F",
        "tradingview": "TVC:PALLADIUM",
        "alt_tradingview": "NYMEX:PA1!",
        "currency": "USD",
        "max_leverage": 1.5,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "USD / domanda auto",
    },
    "WHEAT": {
        "name": "Wheat",
        "category": "Commodity",
        "group": "Agricole",
        "yfinance": "ZW=F",
        "tradingview": "CBOT:ZW1!",
        "alt_tradingview": "TVC:WHEAT",
        "currency": "USD",
        "max_leverage": 1.5,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "USDA WASDE / meteo",
    },
    "CORN": {
        "name": "Corn",
        "category": "Commodity",
        "group": "Agricole",
        "yfinance": "ZC=F",
        "tradingview": "CBOT:ZC1!",
        "alt_tradingview": "TVC:CORN",
        "currency": "USD",
        "max_leverage": 1.5,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "USDA WASDE / meteo",
    },
    "SOYBEAN": {
        "name": "Soybeans",
        "category": "Commodity",
        "group": "Agricole",
        "yfinance": "ZS=F",
        "tradingview": "CBOT:ZS1!",
        "alt_tradingview": "TVC:SOYBEAN",
        "currency": "USD",
        "max_leverage": 1.5,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "USDA WASDE / Cina",
    },
    "COFFEE": {
        "name": "Coffee",
        "category": "Commodity",
        "group": "Agricole",
        "yfinance": "KC=F",
        "tradingview": "ICEUS:KC1!",
        "alt_tradingview": "TVC:COFFEE",
        "currency": "USD",
        "max_leverage": 1.5,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Meteo Brasile / USDA",
    },
    "USDCHF": {
        "name": "USD / CHF",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "CHF=X",
        "tradingview": "FX:USDCHF",
        "alt_tradingview": "OANDA:USDCHF",
        "currency": "CHF",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "FOMC / SNB / CPI",
    },
    "AUDUSD": {
        "name": "AUD / USD",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "AUDUSD=X",
        "tradingview": "FX:AUDUSD",
        "alt_tradingview": "OANDA:AUDUSD",
        "currency": "USD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "RBA / Cina / FOMC",
    },
    "USDCAD": {
        "name": "USD / CAD",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "CAD=X",
        "tradingview": "FX:USDCAD",
        "alt_tradingview": "OANDA:USDCAD",
        "currency": "CAD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "BoC / petrolio / FOMC",
    },
    "NZDUSD": {
        "name": "NZD / USD",
        "category": "Forex",
        "group": "Major FX",
        "yfinance": "NZDUSD=X",
        "tradingview": "FX:NZDUSD",
        "alt_tradingview": "OANDA:NZDUSD",
        "currency": "USD",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "RBNZ / FOMC",
    },
    "EURGBP": {
        "name": "EUR / GBP",
        "category": "Forex",
        "group": "Cross FX",
        "yfinance": "EURGBP=X",
        "tradingview": "FX:EURGBP",
        "alt_tradingview": "OANDA:EURGBP",
        "currency": "GBP",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "ECB / BoE / CPI",
    },
    "EURJPY": {
        "name": "EUR / JPY",
        "category": "Forex",
        "group": "Cross FX",
        "yfinance": "EURJPY=X",
        "tradingview": "FX:EURJPY",
        "alt_tradingview": "OANDA:EURJPY",
        "currency": "JPY",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "ECB / BoJ",
    },
    "GBPJPY": {
        "name": "GBP / JPY",
        "category": "Forex",
        "group": "Cross FX",
        "yfinance": "GBPJPY=X",
        "tradingview": "FX:GBPJPY",
        "alt_tradingview": "OANDA:GBPJPY",
        "currency": "JPY",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "BoE / BoJ",
    },
    "EURCHF": {
        "name": "EUR / CHF",
        "category": "Forex",
        "group": "Cross FX",
        "yfinance": "EURCHF=X",
        "tradingview": "FX:EURCHF",
        "alt_tradingview": "OANDA:EURCHF",
        "currency": "CHF",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "ECB / SNB",
    },
    "AUDJPY": {
        "name": "AUD / JPY",
        "category": "Forex",
        "group": "Cross FX",
        "yfinance": "AUDJPY=X",
        "tradingview": "FX:AUDJPY",
        "alt_tradingview": "OANDA:AUDJPY",
        "currency": "JPY",
        "max_leverage": 3.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.0,
        "point_value": 100000.0,
        "event": "RBA / BoJ / risk sentiment",
    },
    "XRPUSD": {
        "name": "XRP",
        "category": "Crypto",
        "group": "Crypto large cap",
        "yfinance": "XRP-USD",
        "tradingview": "BITSTAMP:XRPUSD",
        "alt_tradingview": "BINANCE:XRPUSDT",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / news regolatorie",
    },
    "BNBUSD": {
        "name": "BNB",
        "category": "Crypto",
        "group": "Crypto large cap",
        "yfinance": "BNB-USD",
        "tradingview": "BINANCE:BNBUSDT",
        "alt_tradingview": "COINBASE:BNBUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "ADAUSD": {
        "name": "Cardano",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "ADA-USD",
        "tradingview": "BINANCE:ADAUSDT",
        "alt_tradingview": "COINBASE:ADAUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "DOGEUSD": {
        "name": "Dogecoin",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "DOGE-USD",
        "tradingview": "BINANCE:DOGEUSDT",
        "alt_tradingview": "COINBASE:DOGEUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / sentiment",
    },
    "LINKUSD": {
        "name": "Chainlink",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "LINK-USD",
        "tradingview": "BINANCE:LINKUSDT",
        "alt_tradingview": "COINBASE:LINKUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "AVAXUSD": {
        "name": "Avalanche",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "AVAX-USD",
        "tradingview": "BINANCE:AVAXUSDT",
        "alt_tradingview": "COINBASE:AVAXUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "LTCUSD": {
        "name": "Litecoin",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "LTC-USD",
        "tradingview": "BITSTAMP:LTCUSD",
        "alt_tradingview": "BINANCE:LTCUSDT",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "DOTUSD": {
        "name": "Polkadot",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "DOT-USD",
        "tradingview": "BINANCE:DOTUSDT",
        "alt_tradingview": "COINBASE:DOTUSD",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
    "BCHUSD": {
        "name": "Bitcoin Cash",
        "category": "Crypto",
        "group": "Crypto satellite",
        "yfinance": "BCH-USD",
        "tradingview": "BITSTAMP:BCHUSD",
        "alt_tradingview": "BINANCE:BCHUSDT",
        "currency": "USD",
        "max_leverage": 1.0,
        "risk_pct_default": 0.25,
        "rr_min": 2.5,
        "point_value": 1.0,
        "event": "Weekend liquidity / macro risk",
    },
}


@dataclass
class TacticalTicket:
    asset: str
    direction: str
    entry: float
    stop: float
    target: float
    capital_eur: float
    risk_pct: float
    leverage: float
    size_units: float
    notional_eur: float
    margin_eur: float
    risk_eur: float
    reward_risk: float
    allowed: bool
    blockers: list[str]


def get_asset(asset: str) -> dict[str, Any]:
    if asset not in ASSET_UNIVERSE:
        raise KeyError(f"Asset non supportato: {asset}")
    return ASSET_UNIVERSE[asset].copy()


def calc_reward_risk(
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> float:
    risk = abs(entry - stop)

    if risk <= 0:
        return 0.0

    reward = (
        target - entry
        if direction.upper() == "LONG"
        else entry - target
    )

    return round(max(reward, 0.0) / risk, 2)


def calc_tactical_score(
    trend_aligned: bool,
    momentum_positive: bool,
    volume_confirmed: bool,
    event_clear: bool,
    reward_risk: float,
    rr_min: float,
    regime_aligned: bool,
) -> int:
    score = 0
    score += 20 if regime_aligned else 0
    score += 20 if trend_aligned else 0
    score += 15 if momentum_positive else 0
    score += 10 if volume_confirmed else 0
    score += 15 if event_clear else 0
    score += 20 if reward_risk >= rr_min else 0
    return int(score)


def calc_position_size(
    capital_eur: float,
    risk_pct: float,
    entry: float,
    stop: float,
    point_value: float = 1.0,
) -> tuple[float, float]:
    risk_eur = max(capital_eur, 0.0) * max(risk_pct, 0.0) / 100.0
    stop_distance = abs(entry - stop)

    if stop_distance <= 0 or point_value <= 0:
        return 0.0, risk_eur

    size_units = risk_eur / (stop_distance * point_value)
    return round(size_units, 8), round(risk_eur, 2)


def calc_leverage(
    capital_eur: float,
    entry: float,
    size_units: float,
) -> tuple[float, float, float]:
    notional_eur = abs(entry * size_units)

    if capital_eur <= 0:
        return notional_eur, 0.0, 0.0

    leverage = notional_eur / capital_eur
    margin_eur = notional_eur / leverage if leverage > 0 else 0.0

    return (
        round(notional_eur, 2),
        round(leverage, 2),
        round(margin_eur, 2),
    )


def build_trade_ticket(
    asset: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    capital_eur: float,
    risk_pct: float,
    leverage_requested: float,
    event_blocked: bool = False,
    daily_loss_pct: float = 0.0,
    daily_loss_limit_pct: float = 1.0,
) -> TacticalTicket:
    config = get_asset(asset)
    direction = direction.upper()

    blockers: list[str] = []

    if direction not in {"LONG", "SHORT"}:
        blockers.append("Direzione non valida")

    if entry <= 0 or stop <= 0 or target <= 0:
        blockers.append("Entry, stop e target devono essere maggiori di zero")

    if direction == "LONG" and stop >= entry:
        blockers.append("LONG: lo stop deve essere sotto l'entry")

    if direction == "SHORT" and stop <= entry:
        blockers.append("SHORT: lo stop deve essere sopra l'entry")

    reward_risk = calc_reward_risk(direction, entry, stop, target)

    if reward_risk < config["rr_min"]:
        blockers.append(
            f"R:R {reward_risk:.2f} inferiore al minimo {config['rr_min']:.2f}"
        )

    if risk_pct > config["risk_pct_default"] * 2:
        blockers.append(
            f"Rischio {risk_pct:.2f}% oltre il limite prudenziale "
            f"{config['risk_pct_default'] * 2:.2f}%"
        )

    if leverage_requested > config["max_leverage"]:
        blockers.append(
            f"Leva richiesta {leverage_requested:.2f}x oltre il limite "
            f"{config['max_leverage']:.2f}x per {asset}"
        )

    if event_blocked:
        blockers.append(f"Blocco evento: {config['event']}")

    if daily_loss_pct >= daily_loss_limit_pct:
        blockers.append(
            f"Kill switch: perdita giornaliera {daily_loss_pct:.2f}% "
            f">= limite {daily_loss_limit_pct:.2f}%"
        )

    size_units, risk_eur = calc_position_size(
        capital_eur=capital_eur,
        risk_pct=risk_pct,
        entry=entry,
        stop=stop,
        point_value=config["point_value"],
    )

    # La size viene ridotta quando la leva richiesta è inferiore al nozionale
    # necessario per la size calcolata sullo stop.
    notional_eur = abs(entry * size_units)
    max_notional = max(capital_eur, 0.0) * max(leverage_requested, 0.0)

    if max_notional > 0 and notional_eur > max_notional:
        size_units = max_notional / entry if entry > 0 else 0.0
        notional_eur = abs(entry * size_units)
        risk_eur = abs(entry - stop) * size_units * config["point_value"]
        blockers.append("Size ridotta dal limite di leva richiesto")

    effective_leverage = (
        notional_eur / capital_eur
        if capital_eur > 0
        else 0.0
    )

    margin_eur = (
        notional_eur / leverage_requested
        if leverage_requested > 0
        else notional_eur
    )

    allowed = len(blockers) == 0

    return TacticalTicket(
        asset=asset,
        direction=direction,
        entry=round(entry, 8),
        stop=round(stop, 8),
        target=round(target, 8),
        capital_eur=round(capital_eur, 2),
        risk_pct=round(risk_pct, 2),
        leverage=round(effective_leverage, 2),
        size_units=round(size_units, 8),
        notional_eur=round(notional_eur, 2),
        margin_eur=round(margin_eur, 2),
        risk_eur=round(risk_eur, 2),
        reward_risk=reward_risk,
        allowed=allowed,
        blockers=blockers,
    )


def ticket_to_frame(ticket: TacticalTicket) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Asset": ticket.asset,
                "Direzione": ticket.direction,
                "Entry": ticket.entry,
                "Stop": ticket.stop,
                "Target": ticket.target,
                "Capitale EUR": ticket.capital_eur,
                "Rischio %": ticket.risk_pct,
                "Rischio EUR": ticket.risk_eur,
                "Size unità": ticket.size_units,
                "Nozionale EUR": ticket.notional_eur,
                "Margine stimato EUR": ticket.margin_eur,
                "Leva effettiva": ticket.leverage,
                "R:R": ticket.reward_risk,
                "Stato": "CONSENTITO" if ticket.allowed else "BLOCCATO",
                "Blocchi": " | ".join(ticket.blockers),
            }
        ]
    )


def make_tradingview_export(
    assets: list[str],
    section: str = "TACTICAL_LAB",
) -> str:
    symbols = []

    for asset in assets:
        if asset in ASSET_UNIVERSE:
            symbols.append(ASSET_UNIVERSE[asset]["tradingview"])

    return f"###{section}," + ",".join(symbols) + ","


def ticket_to_csv(ticket: TacticalTicket) -> bytes:
    return ticket_to_frame(ticket).to_csv(index=False).encode("utf-8")


def ticket_to_xlsx(ticket: TacticalTicket) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        ticket_to_frame(ticket).to_excel(
            writer,
            index=False,
            sheet_name="Trade_Ticket",
        )

    return output.getvalue()
# ─────────────────────────────────────────────────────────────────────────
# Tactical Scanner — dati, indicatori e ranking candidati
# ─────────────────────────────────────────────────────────────────────────

def _fetch_tactical_history_raw(
    yf_symbol: str,
    period: str = "6mo",
    interval: str = "1h",
) -> pd.DataFrame:
    """Scarica OHLCV di riferimento via Yahoo Finance."""
    import yfinance as yf

    data = yf.download(
        yf_symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance può restituire colonne MultiIndex in alcune versioni.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    return data.dropna(how="all")


def calculate_tactical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calcola trend, RSI, ATR e volume proxy senza dipendenze TA esterne."""
    if data is None or data.empty:
        return pd.DataFrame()

    frame = data.copy()

    required = {"Close", "High", "Low"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    close = pd.to_numeric(frame["Close"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")

    frame["EMA20"] = close.ewm(span=20, adjust=False).mean()
    frame["EMA50"] = close.ewm(span=50, adjust=False).mean()
    frame["EMA200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    frame["RSI14"] = 100 - (100 / (1 + rs))

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    frame["ATR14"] = true_range.rolling(14).mean()
    frame["ATR_pct"] = frame["ATR14"] / close * 100

    if "Volume" in frame.columns:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        frame["Vol_Ratio"] = volume / volume.rolling(20).mean()
    else:
        frame["Vol_Ratio"] = 1.0

    frame["High_20"] = high.rolling(20).max()
    frame["Low_20"] = low.rolling(20).min()

    return frame.dropna(subset=["EMA50", "EMA200", "RSI14", "ATR14"])


def analyze_tactical_asset(
    asset: str,
    period: str = "6mo",
    interval: str = "1h",
) -> dict[str, Any]:
    """Analizza un singolo asset e restituisce una riga scanner."""
    config = get_asset(asset)
    data = fetch_tactical_history(
        config["yfinance"],
        period=period,
        interval=interval,
    )
    indicators = calculate_tactical_indicators(data)

    if indicators.empty:
        return {
            "Asset": asset,
            "Nome": config["name"],
            "Categoria": config["category"],
            "Stato": "NO DATA",
            "Tactical_Score": 0,
            "TradingView": config["tradingview"],
            "Errore": "Dati non disponibili",
        }

    row = indicators.iloc[-1]

    price = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    rsi = float(row["RSI14"])
    atr = float(row["ATR14"])
    atr_pct = float(row["ATR_pct"])
    vol_ratio = float(row.get("Vol_Ratio", 1.0) or 1.0)

    bullish = price > ema50 > ema200
    bearish = price < ema50 < ema200

    if bullish:
        direction = "LONG"
        trend = "Bullish"
        stop = min(ema50, price - atr * 1.5)
        target = price + max(price - stop, atr) * config["rr_min"]
        setup = "Trend pullback" if price <= ema20 * 1.01 else "Trend continuation"
        momentum = rsi >= 52
    elif bearish:
        direction = "SHORT"
        trend = "Bearish"
        stop = max(ema50, price + atr * 1.5)
        target = price - max(stop - price, atr) * config["rr_min"]
        setup = "Trend pullback" if price >= ema20 * 0.99 else "Trend continuation"
        momentum = rsi <= 48
    else:
        direction = "WAIT"
        trend = "Range / misto"
        stop = 0.0
        target = 0.0
        setup = "Setup incompleto"
        momentum = False

    reward_risk = (
        calc_reward_risk(direction, price, stop, target)
        if direction in {"LONG", "SHORT"}
        else 0.0
    )

    trend_ok = direction in {"LONG", "SHORT"}
    volume_ok = vol_ratio >= 1.0
    regime_ok = trend_ok
    event_clear = True

    score = calc_tactical_score(
        trend_aligned=trend_ok,
        momentum_positive=momentum,
        volume_confirmed=volume_ok,
        event_clear=event_clear,
        reward_risk=reward_risk,
        rr_min=config["rr_min"],
        regime_aligned=regime_ok,
    )

    # Per futures/commodity il volume sotto media indebolisce il setup.
    if vol_ratio < 0.8:
        score = max(score - 15, 0)
    elif vol_ratio < 1.0:
        score = max(score - 5, 0)

    if score >= 80:
        status = "A — Piano possibile"
    elif score >= 65:
        status = "B — Attendere trigger"
    elif score >= 50:
        status = "C — Setup incompleto"
    else:
        status = "Evitare"

    return {
        "Asset": asset,
        "Nome": config["name"],
        "Categoria": config["category"],
        "Gruppo": config["group"],
        "Prezzo": round(price, 6),
        "Direzione": direction,
        "Trend": trend,
        "Setup": setup,
        "EMA20": round(ema20, 6),
        "EMA50": round(ema50, 6),
        "EMA200": round(ema200, 6),
        "RSI14": round(rsi, 2),
        "ATR": round(atr, 6),
        "ATR_%": round(atr_pct, 2),
        "Volume_x": round(vol_ratio, 2),
        "Stop_stimato": round(stop, 6),
        "Target_stimato": round(target, 6),
        "R_R_stimato": round(reward_risk, 2),
        "Tactical_Score": score,
        "Stato": status,
        "TradingView": config["tradingview"],
        "Errore": "",
    }


def scan_tactical_universe(
    category: str | None = None,
    period: str = "6mo",
    interval: str = "1h",
) -> pd.DataFrame:
    """Scansiona l'universo e restituisce le opportunità ordinate per score."""
    rows = []

    for asset, config in ASSET_UNIVERSE.items():
        if category and config["category"] != category:
            continue

        try:
            rows.append(
                analyze_tactical_asset(
                    asset=asset,
                    period=period,
                    interval=interval,
                )
            )
        except Exception as error:
            rows.append(
                {
                    "Asset": asset,
                    "Nome": config["name"],
                    "Categoria": config["category"],
                    "Stato": "ERRORE",
                    "Tactical_Score": 0,
                    "TradingView": config["tradingview"],
                    "Errore": str(error)[:120],
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Tactical_Score", "Asset"],
        ascending=[False, True],
    ).reset_index(drop=True)


_HISTORY_CACHE: dict = {}
_HISTORY_TTL_SECONDS = 300


def fetch_tactical_history(
    yf_symbol: str,
    period: str = "6mo",
    interval: str = "1h",
) -> pd.DataFrame:
    """Wrapper con cache in memoria (TTL 5 minuti) per ridurre le chiamate a Yahoo."""
    import time

    key = (yf_symbol, period, interval)
    now = time.time()
    hit = _HISTORY_CACHE.get(key)
    if hit and now - hit[0] < _HISTORY_TTL_SECONDS:
        return hit[1].copy()

    data = _fetch_tactical_history_raw(yf_symbol, period, interval)
    if data is not None and not data.empty:
        _HISTORY_CACHE[key] = (now, data.copy())
    return data
