"""
Financial Data Analysis & Forecasting
Project 3 — Python (Pandas, NumPy, Matplotlib, Seaborn) + Custom ARIMA/Holt-Winters
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_stock_data(ticker, start_price, trend, volatility, n_days=1095):
    """Simulate 3 years of daily stock prices via geometric Brownian motion."""
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq='B')
    dt = 1 / 252
    mu = trend
    sigma = volatility
    returns = np.random.normal((mu - 0.5 * sigma**2) * dt,
                               sigma * np.sqrt(dt), n_days)
    prices = start_price * np.exp(np.cumsum(returns))
    volume = np.random.randint(5_000_000, 50_000_000, n_days)
    n = len(dates)  # use actual date count (freq='B' can differ from n_days)
    return pd.DataFrame({
        'Date': dates, 'Ticker': ticker,
        'Open':   (prices[:n] * np.random.uniform(0.995, 1.005, n)).round(2),
        'High':   (prices[:n] * np.random.uniform(1.001, 1.020, n)).round(2),
        'Low':    (prices[:n] * np.random.uniform(0.980, 0.999, n)).round(2),
        'Close':  prices[:n].round(2),
        'Volume': volume[:n]
    })

tickers = {
    'AAPL': (182.0, 0.22, 0.28),
    'MSFT': (375.0, 0.18, 0.25),
    'GOOGL': (140.0, 0.15, 0.30),
    'TSLA': (240.0, 0.10, 0.55),
    'AMZN': (178.0, 0.20, 0.32),
}

raw_frames = [generate_stock_data(t, *p) for t, p in tickers.items()]
raw_df = pd.concat(raw_frames, ignore_index=True)

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

def inject_issues(df):
    """Inject realistic data quality issues for demonstration."""
    df = df.copy()
    idx = np.random.choice(df.index, size=int(len(df) * 0.015), replace=False)
    df.loc[idx, 'Close'] = np.nan                          # missing prices
    neg_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[neg_idx, 'Volume'] = -df.loc[neg_idx, 'Volume']  # negative volume
    dup_idx = np.random.choice(df.index, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)  # duplicates
    return df

dirty_df = inject_issues(raw_df)

def clean_data(df):
    print("=== DATA CLEANING REPORT ===")
    print(f"Rows before cleaning : {len(df):,}")
    df = df.drop_duplicates(subset=['Date', 'Ticker'])
    print(f"Duplicates removed   : {len(dirty_df) - len(df)}")

    missing = df['Close'].isna().sum()
    print(f"Missing Close values : {missing}")
    df = df.sort_values(['Ticker', 'Date'])
    df['Close'] = df.groupby('Ticker')['Close'].transform(
        lambda s: s.interpolate(method='linear'))

    neg_vol = (df['Volume'] < 0).sum()
    print(f"Negative volumes     : {neg_vol}")
    df['Volume'] = df['Volume'].abs()

    # OHLC consistency
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low']  = df[['Open', 'Low',  'Close']].min(axis=1)

    # Derived fields
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['MA_20'] = df.groupby('Ticker')['Close'].transform(
        lambda s: s.rolling(20).mean())
    df['MA_50'] = df.groupby('Ticker')['Close'].transform(
        lambda s: s.rolling(50).mean())
    df['Volatility_20'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda s: s.rolling(20).std() * np.sqrt(252))
    df['Month']  = df['Date'].dt.to_period('M')
    df['Year']   = df['Date'].dt.year
    df['Quarter']= df['Date'].dt.to_period('Q')
    print(f"Rows after cleaning  : {len(df):,}")
    print()
    return df

stock_df = clean_data(dirty_df)

# ─────────────────────────────────────────────
# 3. SYNTHETIC FINANCIAL STATEMENTS
# ─────────────────────────────────────────────

quarters = pd.period_range('2021Q1', periods=16, freq='Q')
revenue_base   = 90_000  # $M
revenue_growth = np.linspace(0, 0.28, 16) + np.random.normal(0, 0.015, 16)
revenue        = revenue_base * (1 + revenue_growth)

cogs_ratio     = np.random.uniform(0.35, 0.42, 16)
gross_profit   = revenue * (1 - cogs_ratio)
opex           = revenue * np.random.uniform(0.20, 0.25, 16)
ebitda         = gross_profit - opex
depreciation   = revenue * 0.04
ebit           = ebitda - depreciation
interest_exp   = np.random.uniform(800, 1200, 16)
ebt            = ebit - interest_exp
tax_rate       = 0.21
net_income     = ebt * (1 - tax_rate)
eps            = net_income / 15_000   # ~15B shares

financials = pd.DataFrame({
    'Quarter':      quarters,
    'Revenue':      revenue,
    'COGS':         revenue * cogs_ratio,
    'Gross_Profit': gross_profit,
    'OPEX':         opex,
    'EBITDA':       ebitda,
    'EBIT':         ebit,
    'Net_Income':   net_income,
    'EPS':          eps,
    'Gross_Margin': gross_profit / revenue * 100,
    'Net_Margin':   net_income / revenue * 100,
    'EBITDA_Margin':ebitda / revenue * 100,
})

# ─────────────────────────────────────────────
# 4. HOLT-WINTERS TRIPLE EXPONENTIAL SMOOTHING (from scratch)
# ─────────────────────────────────────────────

def holt_winters(series, alpha=0.3, beta=0.1, gamma=0.1, season_len=4, n_forecast=8):
    """Triple Exponential Smoothing — additive seasonality."""
    y = np.array(series, dtype=float)
    n = len(y)
    # Initialise
    level   = np.zeros(n + n_forecast)
    trend_  = np.zeros(n + n_forecast)
    season  = np.zeros(n + n_forecast)
    fitted  = np.zeros(n + n_forecast)

    # Seed level and trend from first two seasons
    level[0] = np.mean(y[:season_len])
    trend_[0]= (np.mean(y[season_len:2*season_len]) - np.mean(y[:season_len])) / season_len
    for i in range(season_len):
        season[i] = y[i] - level[0]

    for t in range(1, n):
        prev_level = level[t-1]
        prev_trend = trend_[t-1]
        prev_season= season[t - season_len] if t >= season_len else season[t % season_len]
        level[t]  = alpha * (y[t] - prev_season) + (1 - alpha) * (prev_level + prev_trend)
        trend_[t] = beta  * (level[t] - prev_level) + (1 - beta) * prev_trend
        season[t] = gamma * (y[t] - level[t]) + (1 - gamma) * prev_season
        fitted[t] = level[t] + trend_[t] + season[t]

    # Forecast
    forecast_vals = []
    for h in range(1, n_forecast + 1):
        t = n - 1 + h
        f = (level[n-1] + h * trend_[n-1] + season[n - season_len + (h - 1) % season_len])
        forecast_vals.append(f)

    residuals = y[1:] - fitted[1:n]
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / y[1:])) * 100
    return fitted[1:n], np.array(forecast_vals), rmse, mape

# Forecast revenue (quarterly, season=4)
fitted_rev, forecast_rev, rmse_rev, mape_rev = holt_winters(
    financials['Revenue'], alpha=0.3, beta=0.08, gamma=0.15,
    season_len=4, n_forecast=8)

# Forecast AAPL daily closing price (simple Holt-Winters, season=5 trading days)
aapl = stock_df[stock_df['Ticker'] == 'AAPL'].sort_values('Date')['Close'].values
# Use weekly seasonality on daily data
fitted_aapl, forecast_aapl, rmse_aapl, mape_aapl = holt_winters(
    aapl, alpha=0.25, beta=0.05, gamma=0.05, season_len=5, n_forecast=90)

print("=== FORECASTING METRICS ===")
print(f"Revenue  — RMSE: ${rmse_rev:,.0f}M   MAPE: {mape_rev:.2f}%")
print(f"AAPL     — RMSE: ${rmse_aapl:.2f}   MAPE: {mape_aapl:.2f}%")
print()

# ─────────────────────────────────────────────
# 5. EDA VISUALISATIONS (saved as PNG for embedding)
# ─────────────────────────────────────────────

PALETTE = ['#00D4FF', '#FF6B35', '#7FFF00', '#FF3CAC', '#FFD700']
BG      = '#0D1117'
CARD    = '#161B22'
TEXT    = '#E6EDF3'
GRID    = '#21262D'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID,
    'font.family': 'monospace', 'font.size': 10,
})

# ── Fig 1: Stock Price Trends ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
axes = axes.flatten()
for ax, (ticker, color) in zip(axes[:5], zip(tickers.keys(), PALETTE)):
    sub = stock_df[stock_df['Ticker'] == ticker].sort_values('Date')
    ax.plot(sub['Date'], sub['Close'], color=color, lw=1.2, alpha=0.9, label='Close')
    ax.plot(sub['Date'], sub['MA_20'],  color='white', lw=0.8, ls='--', alpha=0.6, label='MA20')
    ax.plot(sub['Date'], sub['MA_50'],  color='#888', lw=0.8, ls=':', alpha=0.6, label='MA50')
    ax.fill_between(sub['Date'], sub['Close'], alpha=0.08, color=color)
    ax.set_title(ticker, fontsize=14, fontweight='bold', color=color, pad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(True, alpha=0.3)
axes[5].axis('off')
fig.suptitle('Stock Price Trends — 3 Year History', fontsize=18, fontweight='bold', color=TEXT, y=1.01)
plt.tight_layout()
plt.savefig('fig1_stock_trends.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 2: Correlation heatmap ─────────────────
pivot = stock_df.pivot_table(index='Date', columns='Ticker', values='Daily_Return')
corr  = pivot.corr()
fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.5,
            linecolor=GRID, ax=ax, annot_kws={'size': 11, 'color': TEXT})
ax.set_title('Return Correlation Matrix', fontsize=14, fontweight='bold', color=TEXT, pad=12)
plt.tight_layout()
plt.savefig('fig2_correlation.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 3: Revenue Forecast ────────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
q_labels = [str(q) for q in financials['Quarter']]
actual_rev = financials['Revenue'].values
ax.plot(range(len(actual_rev)), actual_rev/1000, 'o-', color=PALETTE[0],
        lw=2, ms=5, label='Actual Revenue')
ax.plot(range(1, len(fitted_rev)+1), fitted_rev/1000, '--', color=PALETTE[1],
        lw=1.5, alpha=0.8, label='Fitted (Holt-Winters)')
future_x = range(len(actual_rev), len(actual_rev)+len(forecast_rev))
ax.plot(future_x, forecast_rev/1000, 's--', color=PALETTE[2],
        lw=2, ms=6, label='Forecast (Next 8Q)')
# Confidence band ±10%
ax.fill_between(future_x,
                forecast_rev/1000 * 0.90,
                forecast_rev/1000 * 1.10,
                alpha=0.15, color=PALETTE[2], label='90% CI')
ax.axvline(len(actual_rev)-0.5, color='white', ls=':', lw=1, alpha=0.5)
ax.text(len(actual_rev)-0.3, ax.get_ylim()[1]*0.95, 'Forecast →',
        color='white', fontsize=9, alpha=0.7)
ax.set_xticks(range(len(actual_rev)+len(forecast_rev)))
all_labels = q_labels + [f'Q{i+1}F' for i in range(len(forecast_rev))]
ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Revenue ($B)', fontsize=11)
ax.set_title(f'Quarterly Revenue & Holt-Winters Forecast  |  MAPE: {mape_rev:.1f}%',
             fontsize=14, fontweight='bold', color=TEXT)
ax.legend(fontsize=9, framealpha=0.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_revenue_forecast.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 4: Margin Analysis ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
quarters_str = [str(q) for q in financials['Quarter']]
for col, color, label in [
    ('Gross_Margin',  PALETTE[0], 'Gross Margin'),
    ('EBITDA_Margin', PALETTE[1], 'EBITDA Margin'),
    ('Net_Margin',    PALETTE[2], 'Net Margin'),
]:
    axes[0].plot(quarters_str, financials[col], marker='o', ms=4, lw=1.8,
                 color={'Gross_Margin': PALETTE[0], 'EBITDA_Margin': PALETTE[1],
                        'Net_Margin': PALETTE[2]}[col], label=label)
axes[0].set_title('Profit Margins (%)', fontsize=13, fontweight='bold', color=TEXT)
axes[0].set_xticklabels(quarters_str, rotation=45, ha='right', fontsize=7)
axes[0].legend(fontsize=9, framealpha=0.2)
axes[0].grid(True, alpha=0.3)
# Revenue vs Net Income bar
x = np.arange(len(quarters_str))
w = 0.4
axes[1].bar(x - w/2, financials['Revenue']/1000,  w, color=PALETTE[0], alpha=0.8, label='Revenue')
axes[1].bar(x + w/2, financials['Net_Income']/1000, w, color=PALETTE[2], alpha=0.8, label='Net Income')
axes[1].set_xticks(x)
axes[1].set_xticklabels(quarters_str, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('$B', fontsize=11)
axes[1].set_title('Revenue vs Net Income', fontsize=13, fontweight='bold', color=TEXT)
axes[1].legend(fontsize=9, framealpha=0.2)
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig4_margins.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 5: AAPL Price Forecast ─────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
aapl_dates = stock_df[stock_df['Ticker']=='AAPL'].sort_values('Date')['Date'].values
ax.plot(aapl_dates, aapl, color=PALETTE[0], lw=1, alpha=0.85, label='AAPL Close')
ax.plot(aapl_dates[1:len(fitted_aapl)+1], fitted_aapl, color=PALETTE[1],
        lw=1, ls='--', alpha=0.7, label='Fitted')
future_dates = pd.date_range(aapl_dates[-1], periods=91, freq='B')[1:]
ax.plot(future_dates, forecast_aapl, color=PALETTE[2], lw=2, label='90-Day Forecast')
ax.fill_between(future_dates,
                forecast_aapl * 0.92, forecast_aapl * 1.08,
                alpha=0.15, color=PALETTE[2], label='±8% Band')
ax.axvline(pd.Timestamp(aapl_dates[-1]), color='white', ls=':', lw=1, alpha=0.4)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title(f'AAPL — Holt-Winters 90-Day Price Forecast  |  MAPE: {mape_aapl:.1f}%',
             fontsize=14, fontweight='bold', color=TEXT)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=9, framealpha=0.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_aapl_forecast.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 6: Annual Returns Bar ──────────────────
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
annual_ret = (stock_df.groupby(['Ticker', 'Year'])['Close']
              .apply(lambda s: (s.iloc[-1] / s.iloc[0] - 1) * 100)
              .reset_index(name='Annual_Return'))
years = sorted(annual_ret['Year'].unique())
x = np.arange(len(years))
n_tickers = len(tickers)
bar_w = 0.14
for i, (ticker, color) in enumerate(zip(tickers.keys(), PALETTE)):
    sub = annual_ret[annual_ret['Ticker'] == ticker]
    heights = [sub[sub['Year']==y]['Annual_Return'].values[0]
               if y in sub['Year'].values else 0 for y in years]
    bars = ax.bar(x + i*bar_w - (n_tickers/2)*bar_w, heights, bar_w,
                  color=color, alpha=0.85, label=ticker)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x()+bar.get_width()/2, h + (1 if h>0 else -3),
                f'{h:.0f}%', ha='center', va='bottom', fontsize=7, color=color)
ax.axhline(0, color='white', lw=0.5, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11)
ax.set_ylabel('Annual Return (%)', fontsize=11)
ax.set_title('Annual Stock Returns by Ticker', fontsize=14, fontweight='bold', color=TEXT)
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig6_annual_returns.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 7: Volatility Over Time ─────────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
for ticker, color in zip(tickers.keys(), PALETTE):
    sub = stock_df[stock_df['Ticker']==ticker].sort_values('Date')
    ax.plot(sub['Date'], sub['Volatility_20']*100, color=color, lw=1.2, label=ticker)
ax.set_ylabel('Annualised 20-Day Volatility (%)', fontsize=11)
ax.set_title('Rolling Volatility — All Tickers', fontsize=14, fontweight='bold', color=TEXT)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10, framealpha=0.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig7_volatility.png', dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()

print("=== ALL FIGURES SAVED ===")

# ─────────────────────────────────────────────
# 6. EXPORT DATA FOR DASHBOARD
# ─────────────────────────────────────────────

# Latest metrics per ticker
latest = stock_df.groupby('Ticker').last().reset_index()
first  = stock_df.groupby('Ticker').first().reset_index()
summary = pd.DataFrame({
    'Ticker':    latest['Ticker'],
    'Last_Price':latest['Close'].round(2),
    'YTD_Return':((latest['Close'].values / first['Close'].values - 1)*100).round(2),
    'Volatility_pct': (latest['Volatility_20']*100).round(2),
    'MA20':       latest['MA_20'].round(2),
    'MA50':       latest['MA_50'].round(2),
})
print("\n=== SUMMARY TABLE ===")
print(summary.to_string(index=False))

# JSON for dashboard
aapl_close = stock_df[stock_df['Ticker']=='AAPL'].sort_values('Date')
msft_close = stock_df[stock_df['Ticker']=='MSFT'].sort_values('Date')
googl_close= stock_df[stock_df['Ticker']=='GOOGL'].sort_values('Date')

dashboard_data = {
    'summary': summary.to_dict('records'),
    'revenue': {
        'quarters': q_labels,
        'actual':   [round(v/1000, 2) for v in actual_rev],
        'fitted':   [None] + [round(v/1000, 2) for v in fitted_rev],
        'forecast_q': [f'Q{i+1}F' for i in range(len(forecast_rev))],
        'forecast':  [round(v/1000, 2) for v in forecast_rev],
        'ci_low':    [round(v/1000*0.90, 2) for v in forecast_rev],
        'ci_high':   [round(v/1000*1.10, 2) for v in forecast_rev],
    },
    'margins': {
        'quarters': q_labels,
        'gross':    financials['Gross_Margin'].round(1).tolist(),
        'ebitda':   financials['EBITDA_Margin'].round(1).tolist(),
        'net':      financials['Net_Margin'].round(1).tolist(),
    },
    'aapl': {
        'dates':   [str(d)[:10] for d in aapl_close['Date']],
        'close':   aapl_close['Close'].round(2).tolist(),
        'ma20':    aapl_close['MA_20'].round(2).tolist(),
        'ma50':    aapl_close['MA_50'].round(2).tolist(),
        'forecast_dates': [str(d)[:10] for d in future_dates],
        'forecast': [round(v, 2) for v in forecast_aapl],
    },
    'annual_returns': annual_ret.to_dict('records'),
    'correlation':    [[round(v, 3) for v in row] for row in corr.values],
    'corr_labels':    list(corr.columns),
    'metrics': {
        'rev_mape': round(mape_rev, 2),
        'aapl_mape': round(mape_aapl, 2),
        'rev_rmse': round(rmse_rev, 0),
    }
}

with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2)

print("\n=== JSON DATA EXPORTED ===")
print("Run financial_analysis.py first, then open financial_dashboard.html")
