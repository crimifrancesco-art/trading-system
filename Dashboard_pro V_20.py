import io
import time
import sqlite3
import locale
import json
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from fpdf import FPDF
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Import modular functions
from utils.formatting import fmt_currency, fmt_int, fmt_marketcap, add_formatted_cols, prepare_display_df
from utils.db import init_db, reset_watchlist_db, add_to_watchlist, load_watchlist, update_watchlist_note, delete_from_watchlist, DB_PATH
from utils.scanner import load_universe, scan_ticker

# -----------------------------------------------------------------------------
# CONFIGURAZIONE BASE PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Scanner – Versione PRO 20.0",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Trading Scanner – Versione PRO 20.0")
st.caption(
    "EARLY •
