# ==============================================================================
# STRATEGIE VISUALI v32 — AGGIORNATE SU IMMAGINI FINGRAD
# ==============================================================================

STRATEGIES_V32 = {
    # ── FOTO 1: KELTNER CHANNEL & MACD ───────────────────────────────────
    "Keltner Channel & MACD": {
        "img": "assets/leg_macd_ema.svg",  # Assicurati che questo file sia la Foto 1
        "desc": "Breakout dal Canale di Keltner confermato da Momentum MACD.",
        "params": {
            "keltner_period": 20,
            "keltner_mult": 1.5, # ATR Multiplier
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        },
        "logic": lambda df: _strategy_keltner_macd(df),
        "rules": [
            "LONG: Close > Lower Keltner AND MACD > Signal",
            "EXIT: Close hits Upper Keltner OR MACD < Signal"
        ]
    },

    # ── FOTO 2: MOMENTUM & TREND FOLLOWING (COMPOSITO) ───────────────────
    # Sostituisce il vecchio duplicato "ADX+EMA". Usa l'infografica completa.
    "Momentum & Trend Pro": {
        "img": "assets/leg_momentum_trend.svg", # Crea/Salva la Foto 2 qui
        "desc": "Confluenza di 4 indicatori: MACD, RSI, Supertrend, Parabolic SAR.",
        "params": {
            "rsi_period": 14,
            "supertrend_factor": 3.0,
            "sar_start": 0.02,
            "sar_max": 0.2
        },
        "logic": lambda df: _strategy_momentum_composite(df),
        "rules": [
            "Entry: MACD Cross UP + RSI > 30 + Supertrend Green + Price > SAR",
            "Exit: Supertrend Red OR Price < SAR"
        ]
    },

    # ── FOTO 3: ADX & CANDLESTICK PATTERN ────────────────────────────────
    "ADX + Piercing Line": {
        "img": "assets/leg_adx_candle.svg", # Salva la Foto 3 qui
        "desc": "Reversal pattern 'Piercing Line' filtrato da forza trend ADX.",
        "params": {
            "adx_period": 14,
            "adx_threshold": 25
        },
        "logic": lambda df: _strategy_adx_piercing(df),
        "rules": [
            "Entry: Pattern Piercing Line detected AND ADX > 25",
            "StopLoss: Low of the Piercing Line candle"
        ]
    },

    # ── FOTO 4: OBV & HULL MOVING AVG ───────────────────────────────────
    "OBV + Hull MA Trend": {
        "img": "assets/leg_obv_hma.svg", # Salva la Foto 4 qui
        "desc": "Trend following veloce con conferma volumetrica OBV.",
        "params": {
            "hma_period": 55,
            "obv_period": 14 # Per smoothing opzionale
        },
        "logic": lambda df: _strategy_obv_hma(df),
        "rules": [
            "SHORT: Close < HMA AND OBV Slope < 0 (falling)",
            "EXIT: Close > HMA OR OBV Rising"
        ]
    },

    # ── FOTO 5: RSI & BOLLINGER BANDS ───────────────────────────────────
    "RSI + Bollinger Breakout": {
        "img": "assets/leg_rsi_bb.svg", # Salva la Foto 5 qui
        "desc": "Mean reversion estrema: rottura bande con RSI in ipercomprato/venduto.",
        "params": {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        },
        "logic": lambda df: _strategy_rsi_bollinger(df),
        "rules": [
            "SHORT: Candle breaks Upper Band AND RSI > 70",
            "EXIT: Candle breaks Lower Band AND RSI < 30"
        ]
    }
}

# ==============================================================================
# IMPLEMENTAZIONE LOGICHE (Funzioni Helper)
# ==============================================================================

def _strategy_keltner_macd(df):
    """Foto 1: Keltner Channel & MACD"""
    # Calcolo Keltner (EMA +/- ATR*Mult)
    ema = df['Close'].ewm(span=20).mean()
    atr = _calc_atr(df, 20)
    kc_upper = ema + (atr * 1.5)
    kc_lower = ema - (atr * 1.5)
    
    # Calcolo MACD
    macd_line, signal_line, _ = _calc_macd(df, 12, 26, 9)
    
    # Segnali
    long_entry = (df['Close'] > kc_lower) & (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    exit_cond = (df['Close'] >= kc_upper) | (macd_line < signal_line)
    
    return _generate_signals(long_entry, exit_cond)

def _strategy_momentum_composite(df):
    """Foto 2: Momentum & Trend Following (Composito)"""
    # 1. MACD
    macd_l, sig_l, _ = _calc_macd(df, 12, 26, 9)
    macd_cross_up = (macd_l > sig_l) & (macd_l.shift(1) <= sig_l.shift(1))
    
    # 2. RSI > 30 (Filtro non oversold estremo)
    rsi = _calc_rsi(df, 14)
    rsi_ok = rsi > 30
    
    # 3. Supertrend (Semplificato: ATR based)
    st_line, st_dir = _calc_supertrend(df, factor=3.0)
    st_green = st_dir == 1 # 1 = Long/Green
    
    # 4. Parabolic SAR
    sar = _calc_parabolic_sar(df)
    price_above_sar = df['Close'] > sar
    
    # Entry: Tutte le condizioni vere
    entry = macd_cross_up & rsi_ok & st_green & price_above_sar
    
    # Exit: Supertrend diventa rosso OPPURE Prezzo sotto SAR
    exit_cond = (st_dir == -1) | (df['Close'] < sar)
    
    return _generate_signals(entry, exit_cond)

def _strategy_adx_piercing(df):
    """Foto 3: ADX + Piercing Line"""
    # ADX > 25
    adx = _calc_adx(df, 14)
    adx_strong = adx > 25
    
    # Pattern Piercing Line (Bearish Red followed by Bullish Green that closes > 50% of Red body)
    # Nota: Implementazione semplificata del pattern
    is_red = df['Close'] < df['Open']
    is_green = df['Close'] > df['Open']
    
    prev_red = is_red.shift(1)
    curr_green = is_green
    
    # Corpo candela precedente
    prev_body = abs(df['Open'].shift(1) - df['Close'].shift(1))
    # Chiusura attuale deve essere > 50% del corpo precedente
    piercing_condition = (df['Close'] > df['Open'].shift(1) - (0.5 * prev_body))
    
    pattern_detected = prev_red & curr_green & piercing_condition
    
    entry = pattern_detected & adx_strong
    
    # Stop Loss: Minimo del pattern (Low della candela verde o rossa, whichever is lower)
    sl = np.minimum(df['Low'], df['Low'].shift(1))
    
    return _generate_signals(entry, pd.Series(False, index=df.index), stop_loss=sl)

def _strategy_obv_hma(df):
    """Foto 4: OBV + Hull Moving Average"""
    # Hull MA (WMA(2*WMA(n/2) - WMA(n)))
    hma = _calc_hma(df, period=55)
    
    # OBV Falling (Slope negativa su periodo breve)
    obv = _calc_obv(df)
    obv_slope = obv.diff(5) # Semplice slope a 5 periodi
    obv_falling = obv_slope < 0
    
    # Short Entry
    short_entry = (df['Close'] < hma) & obv_falling
    
    # Exit
    exit_cond = (df['Close'] > hma) | (obv_slope > 0)
    
    return _generate_signals(short_entry, exit_cond, direction='SHORT')

def _strategy_rsi_bollinger(df):
    """Foto 5: RSI + Bollinger Bands"""
    # Bollinger
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)
    
    # RSI
    rsi = _calc_rsi(df, 14)
    
    # Short Entry: Breakout Upper Band + RSI > 70
    short_entry = (df['Close'] > bb_upper) & (rsi > 70)
    
    # Exit: Breakdown Lower Band + RSI < 30
    exit_cond = (df['Close'] < bb_lower) & (rsi < 30)
    
    return _generate_signals(short_entry, exit_cond, direction='SHORT')

# ... (Includere le funzioni helper _calc_atr, _calc_macd, _calc_rsi, etc. come già presenti nel tuo codice)
