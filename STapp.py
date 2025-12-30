import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === 1. 頁面初始設定 ===
st.set_page_config(page_title="GHSMC 廠務戰情中心", layout="wide", initial_sidebar_state="expanded")

# CSS 優化：戰情室風格
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric { background-color: #f9f9f9; padding: 10px; border-radius: 10px; border: 1px solid #eee;}
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; }
    .stTabs [aria-selected="true"] { background-color: #e6f3ff; border-bottom: 3px solid #007bff; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 GHSMC Scrubber 智慧監控戰情室")
st.markdown("**Client:** GHSMC (積海半導體) | **System:** Skybit-PI Fleet Monitor")

# === 2. 模擬全廠數據 (Fleet Simulation) ===
# 這裡模擬 10 台機台的即時狀態
def generate_fleet_data():
    np.random.seed(42) # 固定種子讓演示穩定，或移除以隨機
    data = []
    for i in range(1, 11):
        # 隨機生成每台機台的健康狀況
        health = np.random.randint(60, 100)
        ph_val = np.random.uniform(6.0, 9.0)
        cond_val = np.random.randint(150, 2500)
        
        status = "Normal"
        if health < 80: status = "Warning"
        if health < 70: status = "Critical"
        
        data.append({
            "Unit ID": f"SC-{i:02d}",
            "Location": f"Zone-{i%3+1}",
            "pH Reading": round(ph_val, 2),
            "Cond. (uS/cm)": int(cond_val),
            "Health Score": health,
            "Status": status
        })
    return pd.DataFrame(data)

df_fleet = generate_fleet_data()

# === 3. 頂部：全廠關鍵指標 (KPIs) ===
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("監控機台總數", "10 Units", "Full Load")
with col_kpi2:
    avg_health = df_fleet["Health Score"].mean()
    st.metric("平均健康度 (Avg Health)", f"{avg_health:.1f}%", "-1.2% vs Last Week")
with col_kpi3:
    warnings = df_fleet[df_fleet["Status"]=="Warning"].shape[0]
    st.metric("一級警報 (Warning)", f"{warnings} Units", delta="Attention Needed", delta_color="off")
with col_kpi4:
    criticals = df_fleet[df_fleet["Status"]=="Critical"].shape[0]
    st.metric("嚴重異常 (Critical)", f"{criticals} Units", delta="Action Required", delta_color="inverse")

st.markdown("---")

# === 4. 左側欄：互動控制台 ===
with st.sidebar:
    st.header("🎮 互動演示控制台")
    st.success("請選擇下方機台進行深入診斷")
    
    # 讓客戶選擇要看哪一台
    selected_unit = st.selectbox(
        "🔍 選擇檢測機台 (Select Unit)", 
        df_fleet["Unit ID"].tolist(),
        index=3 # 預設選一台比較有問題的讓客戶看
    )
    
    current_status = df_fleet[df_fleet["Unit ID"] == selected_unit].iloc[0]
    
    st.markdown("---")
    st.subheader(f"🛠️ {selected_unit} 物理參數模擬")
    st.caption("調整下方滑桿，模擬該機台的真實反應")
    
    # 這些滑桿只影響選中的那一台
    gas_load = st.slider("酸性廢氣流量 (Gas Load)", 0.0, 1.0, 0.5)
    fouling_factor = st.slider("感測器結垢程度 (Fouling)", 0.0, 1.0, 
                               0.8 if current_status['Status'] != 'Normal' else 0.2, 
                               help="模擬該機台電極髒污程度")
    noise_level = st.slider("訊號雜訊 (Noise)", 0.01, 0.1, 0.02)

# === 5. 主畫面：分層顯示 ===

# 分頁：全廠列表 vs 單機診斷
main_tab1, main_tab2 = st.tabs(["📋 全廠總覽 (Fleet Overview)", "🔬 單機深度診斷 (Deep Diagnostics)"])

with main_tab1:
    st.subheader("即時機台狀態列表")
    
    # 使用顏色標記 DataFrame
    def highlight_status(val):
        color = 'green' if val == 'Normal' else 'orange' if val == 'Warning' else 'red'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        df_fleet.style.applymap(highlight_status, subset=['Status']),
        use_container_width=True,
        height=400
    )
    st.info("💡 提示：點擊上方標籤頁「單機深度診斷」或側邊欄選單，查看特定機台的物理分析。")

with main_tab2:
    st.subheader(f"📍 機台 {selected_unit} 雙感測器物理分析")
    
    # --- 這裡放入原本的物理核心代碼 ---
    steps = 200
    t = np.linspace(0, 100, steps)
    
    # A. pH 模型
    pino_ph_target = 8.5 - (gas_load * 4.0) + 0.3 * np.sin(t/8)
    ph_drift = fouling_factor * 1.5 
    sensor_ph_reading = pino_ph_target + ph_drift + np.random.normal(0, noise_level, steps)

    # B. 導電度模型
    pino_cond_target = 200 + (gas_load * 1800) + 50 * np.sin(t/5)
    cond_attenuation = fouling_factor * 800 
    sensor_cond_reading = pino_cond_target - cond_attenuation + np.random.normal(0, noise_level*100, steps)
    sensor_cond_reading = np.maximum(sensor_cond_reading, 0)

    # 顯示圖表
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧪 pH 趨勢分析")
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(x=t, y=pino_ph_target, name='Skybit-PI 真實值', line=dict(color='#00CC96', width=3, dash='dash')))
        fig_ph.add_trace(go.Scatter(x=t, y=sensor_ph_reading, name='Sensor 讀值', line=dict(color='#EF553B', width=2)))
        fig_ph.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([pino_ph_target, sensor_ph_reading[::-1]]),
                                    fill='toself', fillcolor='rgba(239, 85, 59, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='漂移量'))
        fig_ph.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ph, use_container_width=True)
        
        ph_error = abs(sensor_ph_reading[-1] - pino_ph_target[-1])
        if ph_error > 1.0:
            st.error(f"⚠️ pH 偵測異常：漂移量 {ph_error:.2f}")
        else:
            st.success("✅ pH 運作正常")

    with col2:
        st.markdown("#### ⚡ 導電度趨勢分析")
        fig_cond = go.Figure()
        fig_cond.add_trace(go.Scatter(x=t, y=pino_cond_target, name='Skybit-PI 真實值', line=dict(color='#636EFA', width=3, dash='dash')))
        fig_cond.add_trace(go.Scatter(x=t, y=sensor_cond_reading, name='Sensor 讀值', line=dict(color='#FFA15A', width=2)))
        fig_cond.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([pino_cond_target, sensor_cond_reading[::-1]]),
                                      fill='toself', fillcolor='rgba(255, 161, 90, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='結垢誤差'))
        fig_cond.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cond, use_container_width=True)
        
        cond_real = pino_cond_target[-1]
        sensor_val = sensor_cond_reading[-1]
        if sensor_val < cond_real * 0.7:
             st.error(f"🚨 嚴重結垢警報：讀值衰退 {(1 - sensor_val/cond_real)*100:.1f}%")
        else:
             st.success("✅ 導電度運作正常")