import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# 網頁基本設定
st.set_page_config(page_title="AI動能妖股雷達 (極速過濾版)", page_icon="🚀", layout="wide")

# ==========================================
# 核心功能模組
# ==========================================
@st.cache_data(ttl=3600)
def get_tw_stock_list():
    stock_dict = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for m in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            df = pd.read_html(StringIO(res.text))[0].iloc[1:]
            for _, row in df.iterrows():
                try:
                    code_name = str(row[0]).split()
                    if len(code_name) == 2:
                        code, name = code_name
                        cat = str(row[4])
                        if len(code) == 4:
                            suffix = ".TW" if m == 2 else ".TWO"
                            stock_dict[f"{code}{suffix}"] = {"name": name, "sector": cat}
                except Exception: 
                    continue
    except Exception as e: 
        print(f"獲取台股清單發生錯誤: {e}")
    return stock_dict

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(StringIO(res.text))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        names = df['Security'].tolist()
        sectors = df['GICS Sector'].tolist()
        return {t: {"name": n, "sector": s} for t, n, s in zip(tickers, names, sectors)}
    except Exception: 
        return {}

def check_market(symbol):
    try:
        data = yf.download(symbol, period="50d", progress=False)
        close = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'].iloc[-1], pd.Series) else float(data['Close'].iloc[-1])
        ma20 = float(data['Close'].rolling(20).mean().iloc[-1].iloc[0]) if isinstance(data['Close'].rolling(20).mean().iloc[-1], pd.Series) else float(data['Close'].rolling(20).mean().iloc[-1])
        return close >= ma20, close, ma20
    except Exception:
        return True, 0, 0

# 🌟 核心升級：利用政府 Open API 進行初階量能過濾
def get_active_tw_tickers(vol_limit):
    active_tickers = set()
    try:
        # 抓取上市今日行情
        twse_res = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", timeout=10).json()
        for row in twse_res:
            try:
                vol = float(row.get('TradeVolume', 0)) / 1000 # 轉換為張
                if vol >= vol_limit:
                    active_tickers.add(f"{row['Code']}.TW")
            except: pass
            
        # 抓取上櫃今日行情
        tpex_res = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes", timeout=10).json()
        for row in tpex_res:
            try:
                vol = float(row.get('TradingVolume', 0)) / 1000 # 轉換為張
                if vol >= vol_limit:
                    active_tickers.add(f"{row['SecuritiesCompanyCode']}.TWO")
            except: pass
    except Exception as e:
        print(f"Open API 抓取失敗: {e}")
    return active_tickers

def process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records):
    try:
        if df is None or df.empty or len(df) < 120: return 
        df = df.dropna()
        
        close = df['Close']
        vol = df['Volume']
        high = df['High']
        low = df['Low']
        open_p = df['Open']
        
        c_close = float(close.iloc[-1])
        c_open = float(open_p.iloc[-1])
        c_high = float(high.iloc[-1])
        c_low = float(low.iloc[-1])
        
        k_len = c_high - c_low
        if k_len > 0:
            upper_shadow = (c_high - max(c_open, c_close)) / k_len
            if upper_shadow > 0.5: return 
                
        vol_20_mean = float(vol.tail(20).mean())
        vol_ratio = float(vol.iloc[-1]) / (vol_20_mean + 1e-9)

        is_black_k = c_close < c_open
        if vol_ratio > 2.5 and is_black_k: return 
        
        roll_5 = close.rolling(5)
        roll_20 = close.rolling(20)
        roll_60 = close.rolling(60)
        
        ma5 = float(roll_5.mean().iloc[-1])
        if c_close < ma5: return  
        
        ma5_bias = ((c_close - ma5) / (ma5 + 1e-9)) * 100
        ma20 = float(roll_20.mean().iloc[-1])
        ma60 = float(roll_60.mean().iloc[-1])
        
        if "台股" in market_name:
            avg_vol = float((vol.tail(5).mean()) / 1000)
        else:
            avg_vol = float(vol.tail(5).mean())
        
        stock_ret_20 = float((c_close / float(close.iloc[-21]) - 1) * 100)
        rs_20 = stock_ret_20 - mkt_ret_20
        
        past_120_max = float(close.iloc[-121:-1].max())
        dist_120_high = float((c_close / past_120_max - 1) * 100) if past_120_max > 0 else 0

        daily_ret = close.pct_change()
        hist_vol = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        std20 = float(roll_20.std().iloc[-1])
        bb_upper = float(ma20 + 2 * std20)
        bb_width = float((bb_upper - (ma20 - 2 * std20)) / (ma20 + 1e-9) * 100)
        
        trend_str = (ma5 / (ma60 + 1e-9) - 1) * 100
        p_to_ma20 = (c_close / (ma20 + 1e-9) - 1) * 100
        p_to_bbupper = (c_close / (bb_upper + 1e-9) - 1) * 100
        roc_10 = float((c_close - close.iloc[-11]) / (close.iloc[-11] + 1e-9) * 100)
        
        if np.isnan(hist_vol) or np.isnan(roc_10): return

        records.append({
            'ID': ticker.replace(".TW", "").replace(".TWO", ""),
            '股名': stock_dict[ticker]['name'],
            '板塊產業': stock_dict[ticker]['sector'],
            '收盤價': round(c_close, 2),
            'MA5 (防守線)': round(ma5, 2),
            '5MA乖離率(%)': round(ma5_bias, 2),
            '爆量倍數': round(vol_ratio, 2),
            'RS相對強度': round(rs_20, 2),       
            '120日高距離(%)': round(dist_120_high, 2), 
            'Avg_Vol': avg_vol, 
            vol_label: round(avg_vol / 1000000, 2) if "美股" in market_name else int(avg_vol),
            'F_RS': rs_20, 'F_120_High': dist_120_high, 'F_Vol_Ratio': vol_ratio, 
            'F_Hist_Vol': hist_vol, 'F_BB_Width': bb_width, 'F_Trend_Strength': trend_str, 
            'F_P_to_MA20': p_to_ma20, 'F_P_to_BBUpper': p_to_bbupper, 'F_ROC_10': roc_10
        })
    except Exception:
        pass

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_calculate_features(market_name, token="", user_vol_limit=1000):
    records = []
    start_date_str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    api_limit_hit = False

    if "台股" in market_name:
        stock_dict = get_tw_stock_list()
        vol_label = "5日均量(張)"
        try:
            mkt_data = yf.download('^TWII', period="1y", auto_adjust=True, progress=False)['Close']
            mkt_ret_20 = float((mkt_data.iloc[-1] / mkt_data.iloc[-21]) - 1) * 100
        except:
            mkt_ret_20 = 0.0
            
        if not stock_dict: return pd.DataFrame(), vol_label, False
        
        # 🎯 初階過濾：先抓 Open API，只把有量的股票名單留下來
        active_tickers = get_active_tw_tickers(user_vol_limit)
        
        # 如果 Open API 失敗，才 fallback 使用全部名單
        target_list = [t for t in stock_dict.keys() if t in active_tickers] if active_tickers else list(stock_dict.keys())
        
        for ticker in target_list:
            if api_limit_hit: break
            stock_id = ticker.replace(".TW", "").replace(".TWO", "")
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start_date_str}
            if token: params["token"] = token
            
            try:
                time.sleep(0.05) 
                res = requests.get(url, params=params, timeout=10)
                data = res.json()
                
                if data.get("msg") != "success":
                    if "limit" in str(data.get("msg")).lower():
                        api_limit_hit = True
                    continue
                    
                df_data = data.get("data", [])
                if not df_data: continue
                
                df = pd.DataFrame(df_data)
                df = df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                
                process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records)
            except Exception:
                continue
    else:
        # 美股邏輯維持不變
        stock_dict = get_sp500_tickers()
        vol_label = "5日均量(M)"
        try:
            mkt_data = yf.download('^GSPC', period="1y", auto_adjust=True, progress=False)['Close']
            mkt_ret_20 = float((mkt_data.iloc[-1] / mkt_data.iloc[-21]) - 1) * 100
        except:
            mkt_ret_20 = 0.0

        if not stock_dict: return pd.DataFrame(), vol_label, False

        all_tickers = list(stock_dict.keys())
        for i in range(0, len(all_tickers), 50):
            batch = all_tickers[i:i+50]
            try:
                time.sleep(1)
                data = yf.download(batch, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
                for ticker in batch:
                    df = data[ticker] if len(batch) > 1 else data
                    process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records)
            except Exception:
                continue
                
    return pd.DataFrame(records), vol_label, api_limit_hit

# ==========================================
# 網頁介面設計
# ==========================================
st.title("🚀 AI 動能妖股雷達 (極速過濾版)")
st.markdown("內建【相對強度 RS】與【半年新高突破】，精準狙擊無懼大盤的真正領頭羊。")

st.sidebar.header("⚙️ 雷達設定")
market = st.sidebar.radio("選擇掃描市場", ["🇹🇼 台股 (API: FinMind)", "🇺🇸 美股 (API: yfinance)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 台灣 API 授權")
st.sidebar.markdown("雖然已內建智慧量能過濾，但若您將標準設得非常低，仍建議填寫[免費註冊的 Token](https://finmindtrade.com/) 以策安全。")
finmind_token = st.sidebar.text_input("FinMind API Token", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 策略微調")
user_vol_limit = st.sidebar.number_input("最小均量限制 (台:張 / 美:百萬股)", min_value=100, max_value=20000, value=1000, step=100)
user_bias_limit = st.sidebar.slider("乖離率扣分門檻 (%)", min_value=1, max_value=15, value=5)
user_penalty = st.sidebar.number_input("超過門檻每 1% 扣幾分?", min_value=1, max_value=20, value=5, step=1)

if st.button("開始全面掃描", type="primary"):
    is_bull, idx_close, idx_ma = check_market("^TWII" if "台股" in market else "^GSPC")

    if is_bull:
        st.success(f"🟢 【大盤偏多】目前指數 ({idx_close:.2f}) 站上月線 ({idx_ma:.2f})，適合動能策略！")
    else:
        st.error(f"🔴 【大盤偏空】目前指數 ({idx_close:.2f}) 跌破月線 ({idx_ma:.2f})，極易假突破，建議空手觀望！")

    with st.status(f"🔍 啟動 {market} 運算中 (已開啟智慧量能過濾，預計 15~30 秒完成)...", expanded=True) as status:
        # 將 user_vol_limit 傳入，先過濾再掃描
        df_all, vol_label, api_limit_hit = fetch_and_calculate_features(market, finmind_token, user_vol_limit)
        
        if df_all.empty:
            status.update(label="❌ 掃描失敗或無符合標的", state="error", expanded=False)
            st.error("目前無法取得數據。如果尚未填寫 Token，可能是額度已耗盡，請稍候再試。")
            st.stop()
            
        df_records = df_all[df_all['Avg_Vol'] >= user_vol_limit].copy()
        
        if df_records.empty:
            status.update(label="❌ 無符合條件的標的", state="error", expanded=False)
            st.warning(f"目前沒有任何標的符合條件，請嘗試調低標準。")
            st.stop()

        features = ['F_RS', 'F_Vol_Ratio', 'F_Hist_Vol', 'F_120_High', 'F_BB_Width', 'F_Trend_Strength', 'F_P_to_MA20', 'F_P_to_BBUpper', 'F_ROC_10']
        weights =  [20.0,  15.0,          15.0,         10.0,         10.0,         10.0,               10.0,          5.0,              5.0] 

        for f in features: df_records[f + '_Rank'] = df_records[f].rank(pct=True)
        
        df_records['AI 總分'] = 0.0
        for f, w in zip(features, weights): df_records['AI 總分'] += df_records[f + '_Rank'] * w
        
        df_records['乖離懲罰分'] = df_records['5MA乖離率(%)'].apply(lambda x: (x - user_bias_limit) * -user_penalty if x > user_bias_limit else 0)
        df_records['AI 總分'] = df_records['AI 總分'] + df_records['乖離懲罰分']
        df_records['AI 總分'] = df_records['AI 總分'].round(2)
        
        top20 = df_records.sort_values(by='AI 總分', ascending=False).head(20)
        
        if api_limit_hit:
            status.update(label="⚠️ 已達 API 額度上限，結算目前已掃描名單！", state="complete", expanded=False)
            st.toast("⚠️ FinMind 額度已滿，已為您結算目前掃描到的股票。")
        else:
            status.update(label="✅ 掃描與運算完成！", state="complete", expanded=False)

    display_cols = ['ID', '股名', '板塊產業', '收盤價', 'MA5 (防守線)', '5MA乖離率(%)', '爆量倍數', 'RS相對強度', '120日高距離(%)', vol_label, 'AI 總分']
    st.dataframe(top20[display_cols], width="stretch", hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🔥 資金匯聚熱區 (前 20 名板塊統計)")
    sector_counts = top20['板塊產業'].value_counts().reset_index()
    sector_counts.columns = ['板塊產業', '進榜檔數']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sector_counts, hide_index=True, width="stretch")
    with col2:
        st.bar_chart(sector_counts.set_index('板塊產業'))

    csv = top20.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下載完整 CSV 報表",
        data=csv,
        file_name=f"Radar_Top20_API_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
