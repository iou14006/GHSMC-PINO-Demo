import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# === 1. 頁面初始設定 ===
st.set_page_config(page_title="Skybit-PI 雙感測器診斷系統", layout="wide")

# CSS 優化：讓 Tabs 看起來更專業
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

st.title("GHSMC🏭 Scrubber 關鍵感測器雙軌診斷系統")
st.markdown("**Client:** GHSMC (積海半導體) | **Tech:** Skybit-PINOs")

# === 2. 側邊欄：全域物理參數模擬 ===
with st.sidebar:
    st.header("⚙️ 物理環境模擬 (Physics Simulation)")
    
    st.subheader("1. 製程負載")
    gas_load = st.slider("酸性廢氣流量 (Acid Gas Load)", 0.0, 1.0, 0.5, 
                         help="模擬 Tool 端排出的 SiH4/HCl 等氣體量")
    
    st.subheader("2. 設備健康狀況")
    fouling_factor = st.slider("感測器結垢程度 (Fouling Factor)", 0.0, 1.0, 0.2, 
                               help="模擬電極表面附著結晶的厚度")
    
    st.subheader("3. 環境干擾")
    noise_level = st.slider("訊號雜訊 (Noise)", 0.01, 0.1, 0.02)

    st.markdown("---")
    st.caption("Skybit-PI 模型會根據上述參數，分別計算 pH 與導電度的理論變化。")

# === 3. 後端數據生成 (核心物理邏輯) ===
# 產生時間序列
steps = 200
t = np.linspace(0, 100, steps)

# --- A. pH 物理模型 (Logarithmic) ---
# 邏輯：氣體多 -> H+濃度高 -> pH 值下降
# 基礎 pH 8.5 (弱鹼性洗滌水)
pino_ph_target = 8.5 - (gas_load * 4.0) + 0.3 * np.sin(t/8)
# pH 感測器衰退特性：反應變慢 (Lag) 且數值會飄向中性 (Drift to 7) 或偏移
ph_drift = fouling_factor * 1.5  # 結垢越嚴重，讀數越不準
sensor_ph_reading = pino_ph_target + ph_drift + np.random.normal(0, noise_level, steps)

# --- B. 導電度物理模型 (Linear/Saturation) ---
# 邏輯：氣體多 -> 鹽類(Salt)增加 -> 導電度(Cond) 上升
# 基礎 200 uS/cm (補水)
pino_cond_target = 200 + (gas_load * 1800) + 50 * np.sin(t/5)
# 導電度感測器衰退特性：電極被絕緣結晶包覆 -> 測到的電阻變大 -> 導電度讀值"低於"真實值
cond_attenuation = fouling_factor * 800 # 結垢越重，數值掉越多
sensor_cond_reading = pino_cond_target - cond_attenuation + np.random.normal(0, noise_level*100, steps)
# 確保不小於0
sensor_cond_reading = np.maximum(sensor_cond_reading, 0)

# === 4. 儀表板分頁顯示 ===

tab1, tab2 = st.tabs(["🧪 pH 酸鹼度監控", "⚡ Conductivity 導電度監控"])

# --- TAB 1: pH 感測器 ---
with tab1:
    col1, col2, col3 = st.columns(3)
    
    # 計算即時誤差
    ph_error = abs(sensor_ph_reading[-1] - pino_ph_target[-1])
    ph_health = max(0, 100 - ph_error * 30)
    
    with col1:
        st.metric("pH 實體讀值 (Sensor)", f"{sensor_ph_reading[-1]:.2f}")
    with col2:
        st.metric("pH Skybit-PI 理論值 (Truth)", f"{pino_ph_target[-1]:.2f}", delta_color="off")
    with col3:
        st.metric("pH 感測器健康度", f"{ph_health:.1f}%", 
                 delta=f"-{ph_error:.2f} Drift", delta_color="inverse")

    # pH 圖表
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=t, y=pino_ph_target, mode='lines', name='PINO 理論值 (Physics)', line=dict(color='#00CC96', width=3, dash='dash')))
    fig_ph.add_trace(go.Scatter(x=t, y=sensor_ph_reading, mode='lines', name='Sensor 實測值', line=dict(color='#EF553B', width=2)))
    # 填充誤差區
    fig_ph.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([pino_ph_target, sensor_ph_reading[::-1]]),
        fill='toself', fillcolor='rgba(239, 85, 59, 0.2)', line=dict(color='rgba(0,0,0,0)'),
        name='異常漂移區間'
    ))
    fig_ph.update_layout(title="pH 感測器：物理一致性分析", xaxis_title="Time", yaxis_title="pH Value", height=350, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_ph, use_container_width=True)
    
    if ph_error > 1.0:
        st.error(f"⚠️ **pH 異常警告：** 偵測到讀值嚴重偏離 ({ph_error:.2f})。可能原因：玻璃電極老化或參考電極阻塞。")
    else:
        st.success("✅ pH 系統運作正常。")

# --- TAB 2: 導電度 感測器 ---
with tab2:
    col1, col2, col3 = st.columns(3)
    
    # 計算即時誤差
    cond_error = abs(sensor_cond_reading[-1] - pino_cond_target[-1])
    cond_health = max(0, 100 - cond_error / 10) # 導電度容忍範圍較大
    
    with col1:
        st.metric("Cond. 實體讀值", f"{int(sensor_cond_reading[-1])} µS/cm")
    with col2:
        st.metric("Cond. Skybit-PI 理論值", f"{int(pino_cond_target[-1])} µS/cm", delta_color="off")
    with col3:
        st.metric("導電度計健康度", f"{cond_health:.1f}%", 
                 delta=f"-{int(cond_error)} µS/cm Error", delta_color="inverse")

    # 導電度 圖表
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Scatter(x=t, y=pino_cond_target, mode='lines', name='Skybit-PI 理論濃度 (Physics)', line=dict(color='#636EFA', width=3, dash='dash')))
    fig_cond.add_trace(go.Scatter(x=t, y=sensor_cond_reading, mode='lines', name='Sensor 實測值', line=dict(color='#FFA15A', width=2)))
    
    # 填充誤差區
    fig_cond.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([pino_cond_target, sensor_cond_reading[::-1]]),
        fill='toself', fillcolor='rgba(255, 161, 90, 0.2)', line=dict(color='rgba(0,0,0,0)'),
        name='結垢影響區間 (Scaling Impact)'
    ))
    
    fig_cond.update_layout(title="導電度 (Conductivity)：結晶結垢分析", xaxis_title="Time", yaxis_title="Conductivity (µS/cm)", height=350, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_cond, use_container_width=True)

    # 針對 GHSMC 的痛點分析
    st.markdown("#### 🔬 Skybit-PI 診斷報告")
    if pino_cond_target[-1] > 1800:
        st.warning("⚠️ **高飽和風險 (High Saturation Risk)：** 目前廢氣負載導致理論導電度過高，建議增加補水量以避免結晶生成。")
    
    if sensor_cond_reading[-1] < pino_cond_target[-1] * 0.7:
        st.error("🚨 **嚴重結垢警告 (Scaling Alert)：** 實測導電度遠低於物理理論值。這表示電極表面已被絕緣結晶覆蓋，系統正在**「假性安全」**狀態運行（以為乾淨，其實很髒）。")
    elif cond_health > 90:
        st.success("✅ 導電度計運作正常，無顯著結垢跡象。")