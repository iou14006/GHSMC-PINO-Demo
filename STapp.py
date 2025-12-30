import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# === 1. 頁面初始設定 ===
st.set_page_config(
    page_title="GHSMC 廠務戰情中心", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS 優化：科技感戰情室風格
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric { background-color: #f9f9f9; padding: 10px; border-radius: 10px; border: 1px solid #eee;}
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; }
    .stTabs [aria-selected="true"] { background-color: #e6f3ff; border-bottom: 3px solid #007bff; }
    h1 { color: #0f1116; }
    h3 { color: #007bff; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 GHSMC Scrubber 智慧監控戰情中心")
st.markdown("**Client:** GHSMC (積海半導體) | **System:** Skybit-PI Fleet Monitor & Predictor")

# === 2. 模擬全廠數據 (Fleet Simulation) ===
@st.cache_data
def generate_fleet_data():
    np.random.seed(42) # 固定種子讓演示穩定
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
    st.info("由此處控制全場參數，模擬不同情境。")
    
    # 全廠層級設定
    st.subheader("1. 全廠製程設定")
    global_load = st.slider("🏭 全廠產能稼動率 (Fab Loading)", 0, 100, 85)
    gas_type = st.selectbox("☠️ 主要製程氣體類型", ["SiH4 (Silane)", "WF6 (Tungsten)", "Cl2 (Chlorine)"])

    st.markdown("---")
    
    # 單機層級設定
    st.subheader("2. 單機診斷選擇")
    selected_unit = st.selectbox(
        "🔍 選擇檢測機台", 
        df_fleet["Unit ID"].tolist(),
        index=3 # 預設選一台比較有問題的
    )
    current_status = df_fleet[df_fleet["Unit ID"] == selected_unit].iloc[0]
    
    st.caption(f"模擬 {selected_unit} 的物理反應：")
    unit_gas_load = st.slider("單機廢氣負載", 0.0, 1.0, 0.5)
    fouling_factor = st.slider("感測器結垢程度", 0.0, 1.0, 
                               0.8 if current_status['Status'] != 'Normal' else 0.2)
    noise_level = st.slider("訊號雜訊 (Noise)", 0.01, 0.1, 0.02)

# === 5. 主畫面：三層式架構 ===
# 分頁定義
tab1, tab2, tab3 = st.tabs([
    "📋 全廠總覽 (Fleet Overview)", 
    "🔬 單機深度診斷 (Digital Twin)", 
    "🌐 Skybit-PI 全廠流場預測 (Field Prediction)"
])

# --- Tab 1: 全廠總覽 ---
with tab1:
    st.subheader("即時機台狀態列表")
    
    def highlight_status(val):
        if val == 'Normal': color = '#d4edda' # 淺綠
        elif val == 'Warning': color = '#fff3cd' # 淺黃
        else: color = '#f8d7da' # 淺紅
        return f'background-color: {color}; color: black'

    st.dataframe(
        df_fleet.style.applymap(highlight_status, subset=['Status']),
        use_container_width=True,
        height=400
    )
    st.info("💡 提示：紅色標記代表 Skybit-PI 模型偵測到物理異常，請切換至「單機深度診斷」查看詳情。")

# --- Tab 2: 單機深度診斷 ---
with tab2:
    st.subheader(f"📍 機台 {selected_unit} 雙感測器物理分析")
    st.markdown(f"目前狀態：**{current_status['Status']}** | 位置：{current_status['Location']}")
    
    # 物理模型計算
    steps = 100
    t = np.linspace(0, 100, steps)
    
    # A. pH 模型 (考慮氣體負載與結垢漂移)
    pino_ph_target = 8.5 - (unit_gas_load * 4.0) + 0.3 * np.sin(t/8)
    ph_drift = fouling_factor * 1.5 
    sensor_ph_reading = pino_ph_target + ph_drift + np.random.normal(0, noise_level, steps)

    # B. 導電度模型 (考慮結晶造成的衰減)
    pino_cond_target = 200 + (unit_gas_load * 1800) + 50 * np.sin(t/5)
    cond_attenuation = fouling_factor * 800 
    sensor_cond_reading = pino_cond_target - cond_attenuation + np.random.normal(0, noise_level*100, steps)
    sensor_cond_reading = np.maximum(sensor_cond_reading, 0)

    # 雙欄圖表佈局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧪 pH 感測器：物理一致性分析")
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(x=t, y=pino_ph_target, name='Skybit-PI 理論真值', line=dict(color='#00CC96', width=3, dash='dash')))
        fig_ph.add_trace(go.Scatter(x=t, y=sensor_ph_reading, name='Sensor 實際讀值', line=dict(color='#EF553B', width=2)))
        fig_ph.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([pino_ph_target, sensor_ph_reading[::-1]]),
                                    fill='toself', fillcolor='rgba(239, 85, 59, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='異常漂移量'))
        fig_ph.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ph, use_container_width=True)
        
        # 智慧診斷訊息
        ph_error = abs(sensor_ph_reading[-1] - pino_ph_target[-1])
        if ph_error > 1.0:
            st.error(f"⚠️ **校正警報**：檢測到 pH 電極漂移 {ph_error:.2f}，建議立即執行兩點校正。")
        else:
            st.success("✅ **系統正常**：pH 感測器讀值符合物理模型預期。")

    with col2:
        st.markdown("#### ⚡ 導電度感測器：結晶結垢分析")
        fig_cond = go.Figure()
        fig_cond.add_trace(go.Scatter(x=t, y=pino_cond_target, name='Skybit-PI 理論真值', line=dict(color='#636EFA', width=3, dash='dash')))
        fig_cond.add_trace(go.Scatter(x=t, y=sensor_cond_reading, name='Sensor 實際讀值', line=dict(color='#FFA15A', width=2)))
        fig_cond.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([pino_cond_target, sensor_cond_reading[::-1]]),
                                      fill='toself', fillcolor='rgba(255, 161, 90, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='結垢訊號衰減'))
        fig_cond.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cond, use_container_width=True)
        
        # 智慧診斷訊息
        cond_real = pino_cond_target[-1]
        sensor_val = sensor_cond_reading[-1]
        loss_pct = (1 - sensor_val/cond_real)*100 if cond_real > 0 else 0
        
        if loss_pct > 30:
             st.error(f"🚨 **嚴重結垢警報**：導電度訊號衰退 {loss_pct:.1f}%，管路可能已堵塞，請檢查清洗。")
        else:
             st.success("✅ **系統正常**：無明顯結晶沉積現象。")

# --- Tab 3: PINO 全場流場預測 ---
with tab3:
    st.subheader("🌐 Skybit-PI 全廠管路結晶沉積預測 (Skybit-PI Field Prediction)")
    st.markdown("""
    此模型利用 **Skybit-PI** 學習流體力學算子。
    即便是過去未發生過的製程氣體組合，Skybit-PI 也能即時推算出全廠管路的高風險堵塞區域。
    """)
    
    # 模擬運算提示
    st.caption(f"當前模擬參數：稼動率 **{global_load}%** | 製程氣體 **{gas_type}**")

    # PINO 模擬邏輯 (使用高斯分佈模擬流場熱點)
    x_grid = np.linspace(0, 10, 30)
    y_grid = np.linspace(0, 10, 30)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 根據不同氣體和負載，改變熱點位置 (模擬不同氣體的物理沉積特性)
    if gas_type == "SiH4 (Silane)":
        # 矽烷：容易在彎頭處堆積 (模擬邊角熱點)
        center1, center2 = (2, 8), (8, 2)
        spread = 2.5
        z_scale = 1.2
    elif gas_type == "WF6 (Tungsten)":
        # 鎢製程：重金屬氣體沉積極快且集中
        center1, center2 = (5, 5), (5, 5) # 集中在中央主管
        spread = 1.0
        z_scale = 1.8
    else:
        # 氯氣：腐蝕性為主，分佈較廣但強度較低
        center1, center2 = (3, 3), (7, 7)
        spread = 4.0
        z_scale = 0.8

    load_factor = global_load / 100.0
    
    # 產生熱力場數據 (Z軸：結晶風險指數)
    # 公式：負載 * (熱點1 + 熱點2) + 隨機擾動
    Z = (load_factor * z_scale) * (
        np.exp(-((X-center1[0])**2 + (Y-center1[1])**2) / spread) + 
        np.exp(-((X-center2[0])**2 + (Y-center2[1])**2) / spread)
    )
    # 加入隨機擾動模擬真實流體紊流
    Z += np.random.normal(0, 0.02, Z.shape)

    # 繪製 3D Surface Plot
    fig_pino = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
    
    fig_pino.update_layout(
        title=f"Skybit-PI 預測：{gas_type} 全廠結晶風險 3D 視圖",
        scene = dict(
            xaxis_title='Fab Zone X',
            yaxis_title='Fab Zone Y',
            zaxis_title='Clogging Risk Index',
            zaxis=dict(range=[0, 2.0]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    st.plotly_chart(fig_pino, use_container_width=True)

    # 商業價值解說 (Actionable Insight)
    st.info(f"""
    **🤖 AI 決策建議：**
    根據 Skybit-PI 運算，在 **{gas_type}** 製程且高負載環境下，
    風險熱點集中於 **座標 ({center1[0]}, {center1[1]})** 區域。
    
    👉 **建議措施：** 1. 建議提前 3 天對該區 Scrubber 進行清洗保養。
    2. 自動調高該區域加熱帶 (Heater) 溫度 5°C 以抑制結晶生成。
    """)