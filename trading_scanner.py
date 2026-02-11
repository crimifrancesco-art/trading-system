#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os, time, sys
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

params = {
    "EARLY_dist_ema_high": 0.02, "EARLY_dist_ema_mid": 0.05,
    "EARLY_score_high": 8, "EARLY_score_mid": 4,
    "PRO_price_above_ema_points": 3, "PRO_rsi_min": 40, "PRO_rsi_max": 70,
    "PRO_rsi_points": 3, "PRO_macd_points": 2, "PRO_vol_ratio_min": 1.2,
    "PRO_vol_points": 2, "PRO_state_threshold": 8,
    "REA_bins": 50, "REA_dist_poc_high": 0.02, "REA_dist_poc_mid": 0.05,
    "REA_poc_points_high": 4, "REA_poc_points_mid": 2,
    "REA_val_vah_dist": 0.02, "REA_val_vah_points": 3,
    "REA_vol_ratio_min": 1.5, "REA_vol_points": 3,
    "REA_hot_threshold": 7, "REA_watch_threshold": 4, "TOP_n": 15
}

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '')
print(f"🚀 Trading Scanner - {datetime.now()}")

def load_universe():
    sp500 = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")["Symbol"].tolist()
    nasdaq = pd.read_csv("https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv")["Symbol"].tolist()
    ftse = ["UCG.MI","ISP.MI","ENEL.MI","ENI.MI","LDO.MI","PRY.MI","TEN.MI","SPM.MI","STM.MI","STLAM.MI","MONC.MI","PST.MI","MB.MI","A2A.MI"]
    return list(dict.fromkeys(sp500 + nasdaq + ftse))

def analizza_early_pro(ticker, params):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data) < 40: return None
        close, volume = data["Close"], data["Volume"]
        nome = yf.Ticker(ticker).info.get("longName", ticker)[:30]
        prezzo = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        dist = abs(prezzo-ema20)/ema20
        early = params["EARLY_score_high"] if dist<params["EARLY_dist_ema_high"] else (params["EARLY_score_mid"] if dist<params["EARLY_dist_ema_mid"] else 0)
        pro = (3 if prezzo>ema20 else 0)
        delta = close.diff()
        rsi = 100-(100/(1+(delta.where(delta>0,0).rolling(14).mean()/(-delta.where(delta<0,0).rolling(14).mean()))))
        rsi_val = float(rsi.iloc[-1])
        pro += 3 if params["PRO_rsi_min"]<rsi_val<params["PRO_rsi_max"] else 0
        macd = close.ewm(12).mean()-close.ewm(26).mean()
        pro += 2 if float((macd-macd.ewm(9).mean()).iloc[-1])>0 else 0
        vol_ratio = float(volume.iloc[-1]/volume.rolling(20).mean().iloc[-1])
        pro += 2 if vol_ratio>params["PRO_vol_ratio_min"] else 0
        return {"Nome":nome,"Ticker":ticker,"Prezzo":round(prezzo,2),"Early_Score":early,"Pro_Score":pro,"Dist_EMA20_%":round((prezzo/ema20-1)*100,1),"RSI":round(rsi_val,1),"Vol_vs_Avg":round(vol_ratio,2)}
    except: return None

def analizza_rea(ticker, params):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if len(data)<60: return None
        nome = yf.Ticker(ticker).info.get("longName",ticker)[:30]
        close, high, low, vol = data["Close"], data["High"], data["Low"], data["Volume"]
        prezzo = float(close.iloc[-1])
        tp = (high+low+close)/3
        bins = np.linspace(float(low.min()),float(high.max()),params["REA_bins"]+1)
        vp = pd.DataFrame({"P":pd.cut(tp,bins,labels=bins[:-1]),"V":vol}).groupby("P")["V"].sum()
        poc = float(vp.idxmax())
        dist = abs(prezzo-poc)/poc
        score = params["REA_poc_points_high"] if dist<params["REA_dist_poc_high"] else (params["REA_poc_points_mid"] if dist<params["REA_dist_poc_mid"] else 0)
        vol_ratio = float(vol.iloc[-1]/vol.rolling(20).mean().iloc[-1])
        score += params["REA_vol_points"] if vol_ratio>params["REA_vol_ratio_min"] else 0
        return {"Nome":nome,"Ticker":ticker,"Prezzo":round(prezzo,2),"Rea_Score":score,"POC":round(poc,2),"Dist_POC_%":round(dist*100,1),"Vol_Spike":round(vol_ratio,2)}
    except: return None

tickers = load_universe()
print(f"📊 Scansione {len(tickers)} titoli...")
r1, r2 = [], []
for i,t in enumerate(tickers):
    if i%100==0: print(f"  {i}/{len(tickers)}")
    x = analizza_early_pro(t,params)
    if x: r1.append(x)
    y = analizza_rea(t,params)
    if y: r2.append(y)
    if (i+1)%25==0: time.sleep(1)

df_ep = pd.DataFrame(r1)
df_rea = pd.DataFrame(r2)
df_ep["Stato"] = df_ep.apply(lambda x: "PRO" if x["Pro_Score"]>=8 else ("EARLY" if x["Early_Score"]>=8 else "-"), axis=1)
df_rea["Stato"] = df_rea.apply(lambda x: "REA-HOT" if x["Rea_Score"]>=7 else ("REA-WATCH" if x["Rea_Score"]>=4 else "-"), axis=1)

TOP = params["TOP_n"]
top_early = df_ep.sort_values(["Early_Score","Pro_Score"],ascending=False).head(TOP)
top_pro = df_ep.sort_values(["Pro_Score","Early_Score"],ascending=False).head(TOP)
top_rea = df_rea.sort_values(["Rea_Score","Vol_Spike"],ascending=False).head(TOP)

excel = f"TradingSystem_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
with pd.ExcelWriter(excel, engine='openpyxl') as w:
    top_early.to_excel(w, sheet_name='TOP_EARLY', index=False)
    top_pro.to_excel(w, sheet_name='TOP_PRO', index=False)
    top_rea.to_excel(w, sheet_name='TOP_REA', index=False)
print(f"✅ Excel: {excel}")

top_early["Ticker"].to_csv("TV_EARLY.csv", index=False, header=["symbol"])
top_pro["Ticker"].to_csv("TV_PRO.csv", index=False, header=["symbol"])
top_rea["Ticker"].to_csv("TV_REA.csv", index=False, header=["symbol"])
print("✅ CSV TradingView salvati")

if SPREADSHEET_ID and os.path.exists("gsheets_credentials.json"):
    try:
        gc = gspread.authorize(Credentials.from_service_account_file("gsheets_credentials.json", scopes=['https://www.googleapis.com/auth/spreadsheets']))
        sh = gc.open_by_key(SPREADSHEET_ID)
        for name,df in [("TOP_EARLY",top_early),("TOP_PRO",top_pro),("TOP_REA",top_rea)]:
            ws = sh.worksheet(name) if name in [w.title for w in sh.worksheets()] else sh.add_worksheet(name,200,20)
            ws.clear()
            set_with_dataframe(ws, df)
        print(f"✅ Google Sheets aggiornato: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}")
    except Exception as e:
        print(f"⚠️ Google Sheets error: {e}")
