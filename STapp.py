import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os  

# === 修正重點：補上 PyTorch 匯入 ===
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# === 1. 頁面初始設定 ===
st.set_page_config(
    page_title="GHSMC 廠務戰情中心", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# === 0. 環境設定 (RTX 5090/GPU 相容性檢查) ===
# 這段是您原本發生錯誤的地方，現在加上了安全檢查
device = 'cpu' # 預設使用 CPU
if HAS_TORCH:
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # print(f"Detected GPU: {gpu_name}") # 除錯用
            # 若是 RTX 5090 或太新的卡，可在此強制切換 CPU
            # device = torch.device('cuda') 
            device = torch.device('cpu') # 演示用途建議強制 CPU 比較穩定
        else:
            device = torch.device('cpu')
    except:
        device = torch.device('cpu')
else:
    # 如果沒安裝 torch，直接略過不報錯
    pass


# CSS 優化
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric { background-color: #f9f9f9; padding: 10px; border-radius: 10px; border: 1px solid #eee;}
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; }
    .stTabs [aria-selected="true"] { background-color: #e6f3ff; border-bottom: 3px solid #007bff; }
    h1 { color: #0f1116; }
    h3 { color: #007bff; }
    .report-box { padding: 15px; border-radius: 10px; background-color: #e8f4f8; border-left: 5px solid #007bff; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 GHSMC Scrubber 智能監控戰情中心")
st.markdown("**Client:** GHSMC (積海半導體) | **System:** Skybit-PI Fleet Monitor & Predictor")

# === 2. 模擬全廠數據 ===
@st.cache_data
def generate_fleet_data():
    np.random.seed(42)
    data = []
    for i in range(1, 11):
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

# === 3. 頂部 KPI ===
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("監控機台總數", "10 Units", "Full Load")
with col_kpi2:
    avg_health = df_fleet["Health Score"].mean()
    st.metric("平均健康度", f"{avg_health:.1f}%", "-1.2% vs Last Week")
with col_kpi3:
    warnings = df_fleet[df_fleet["Status"]=="Warning"].shape[0]
    st.metric("一級警報 (Warning)", f"{warnings} Units", delta="Attention Needed", delta_color="off")
with col_kpi4:
    criticals = df_fleet[df_fleet["Status"]=="Critical"].shape[0]
    st.metric("嚴重異常 (Critical)", f"{criticals} Units", delta="Action Required", delta_color="inverse")

st.markdown("---")

# === 4. 左側欄 ===
with st.sidebar:
    st.header("🎮 互動演示控制台")
    st.info("由此處控制全場參數，模擬不同情境。")
    
    st.subheader("1. 全廠製程設定")
    global_load = st.slider("🏭 全廠產能稼動率 (Fab Loading)", 0, 100, 85)
    gas_type = st.selectbox("☠️ 主要製程氣體類型", ["SiH4 (Silane)", "WF6 (Tungsten)", "Cl2 (Chlorine)"])

    st.markdown("---")
    
    st.subheader("2. 單機診斷選擇")
    selected_unit = st.selectbox("🔍 選擇檢測機台", df_fleet["Unit ID"].tolist(), index=3)
    current_status = df_fleet[df_fleet["Unit ID"] == selected_unit].iloc[0]
    
    st.caption(f"模擬 {selected_unit} 的物理反應：")
    unit_gas_load = st.slider("單機廢氣負載", 0.0, 1.0, 0.5)
    fouling_factor = st.slider("感測器結垢程度", 0.0, 1.0, 0.8 if current_status['Status'] != 'Normal' else 0.2)
    noise_level = st.slider("訊號雜訊 (Noise)", 0.01, 0.1, 0.02)

# === 5. 主畫面 ===
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 全廠總覽 (Fleet Overview)", 
    "🔬 單機深度診斷 (Digital Twin)", 
    "🌐 Skybit-PI 全場流場預測",
    "🛠️ 工程師調校模型 (Model Lab)" 
])

# --- Tab 1: 全廠總覽 ---
with tab1:
    st.subheader("即時機台狀態列表")
    def highlight_status(val):
        if val == 'Normal': color = '#d4edda'
        elif val == 'Warning': color = '#fff3cd'
        else: color = '#f8d7da'
        return f'background-color: {color}; color: black'
    st.dataframe(df_fleet.style.applymap(highlight_status, subset=['Status']), use_container_width=True, height=400)
    st.info("💡 提示：紅色標記代表 Skybit-PI 模型偵測到物理異常，請切換至「單機深度診斷」查看詳情。")

# --- Tab 2: 單機深度診斷 (含智能診斷報告) ---
with tab2:
    st.subheader(f"📍 機台 {selected_unit} 雙感測器物理分析")
    st.markdown(f"目前狀態：**{current_status['Status']}** | 位置：{current_status['Location']}")
    
    steps = 100
    t = np.linspace(0, 100, steps)
    pino_ph_target = 8.5 - (unit_gas_load * 4.0) + 0.3 * np.sin(t/8)
    ph_drift = fouling_factor * 1.5 
    sensor_ph_reading = pino_ph_target + ph_drift + np.random.normal(0, noise_level, steps)

    pino_cond_target = 200 + (unit_gas_load * 1800) + 50 * np.sin(t/5)
    cond_attenuation = fouling_factor * 800 
    sensor_cond_reading = pino_cond_target - cond_attenuation + np.random.normal(0, noise_level*100, steps)
    sensor_cond_reading = np.maximum(sensor_cond_reading, 0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧪 pH 感測器：物理一致性分析")
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(x=t, y=pino_ph_target, name='Skybit-PI 理論真值', line=dict(color='#00CC96', width=3, dash='dash')))
        fig_ph.add_trace(go.Scatter(x=t, y=sensor_ph_reading, name='Sensor 實際讀值', line=dict(color='#EF553B', width=2)))
        fig_ph.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ph, use_container_width=True)
        
    with col2:
        st.markdown("#### ⚡ 導電度感測器：結晶結垢分析")
        fig_cond = go.Figure()
        fig_cond.add_trace(go.Scatter(x=t, y=pino_cond_target, name='Skybit-PI 理論真值', line=dict(color='#636EFA', width=3, dash='dash')))
        fig_cond.add_trace(go.Scatter(x=t, y=sensor_cond_reading, name='Sensor 實際讀值', line=dict(color='#FFA15A', width=2)))
        fig_cond.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_cond, use_container_width=True)

    # 智能診斷報告
    st.markdown("---")
    st.subheader("🤖 Skybit-PI 智能診斷報告 (AI Diagnostic Report)")
    
    ph_error = abs(sensor_ph_reading[-1] - pino_ph_target[-1])
    cond_loss_pct = (1 - sensor_cond_reading[-1]/pino_cond_target[-1]) * 100 if pino_cond_target[-1] > 0 else 0
    
    report_col1, report_col2 = st.columns([2, 1])
    with report_col1:
        if ph_error > 1.0 or cond_loss_pct > 30:
            st.error(f"❌ **目前狀態 (Current Status):** 機台 {selected_unit} 偵測到物理模型偏差，系統判定為「異常 (Critical)」。")
            st.markdown("#### 🔧 下一步預備 (Next Step Preparation):")
            if ph_error > 1.0:
                st.write(f"- **pH 異常**：偵測到電極漂移 {ph_error:.2f}。建議工程師攜帶 **pH 7.0/4.0 校正液** 前往現場進行兩點校正。")
            if cond_loss_pct > 30:
                st.write(f"- **導電度異常**：訊號衰退 {cond_loss_pct:.1f}%，顯示管壁有嚴重結晶。建議 **立即安排 PM 清洗**，並檢查加熱帶 (Heater) 運作是否正常。")
        else:
            st.success(f"✅ **目前狀態 (Current Status):** 機台 {selected_unit} 運作參數與物理模型相符，系統判定為「健康 (Healthy)」。")
            st.markdown("#### 🔧 下一步預備 (Next Step Preparation):")
            st.write("- 目前無需維修，請維持每週例行巡檢即可。")
            st.write("- Skybit-PI 將持續進行 24/7 物理監控。")
            
    with report_col2:
        health_score = max(0, 100 - (ph_error * 10) - (cond_loss_pct * 0.5))
        st.metric("機台即時健康分", f"{health_score:.1f}", delta="-2.5" if health_score < 80 else "+0.1")

# --- Tab 3: Skybit-PI 全場流場預測 ---
with tab3:
    st.subheader("🌐 Skybit-PI 全場管路結晶沉積預測")
    
    x_grid = np.linspace(0, 10, 30)
    y_grid = np.linspace(0, 10, 30)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    if gas_type == "SiH4 (Silane)":
        center1, center2 = (2, 8), (8, 2); spread = 2.5; z_scale = 1.2
        risk_desc = "彎頭處 (Elbows)"
    elif gas_type == "WF6 (Tungsten)":
        center1, center2 = (5, 5), (5, 5); spread = 1.0; z_scale = 1.8
        risk_desc = "主幹管匯流處 (Main Trunk)"
    else:
        center1, center2 = (3, 3), (7, 7); spread = 4.0; z_scale = 0.8
        risk_desc = "分支管路 (Branches)"

    load_factor = global_load / 100.0
    Z = (load_factor * z_scale) * (np.exp(-((X-center1[0])**2 + (Y-center1[1])**2) / spread) + np.exp(-((X-center2[0])**2 + (Y-center2[1])**2) / spread))
    Z += np.random.normal(0, 0.02, Z.shape)

    c_text, c_chart = st.columns([1, 2])
    with c_text:
        st.info(f"**目前模擬條件：**\n\n製程：{gas_type}\n\n稼動率：{global_load}%")
        st.markdown("### 📊 目前狀態 (Current Status)")
        st.write(f"Skybit-PI 模型預測在目前高負載下，**{risk_desc}** 區域 (座標熱點) 將產生高濃度的結晶沉積。")
        st.markdown("### 🛡️ 下一步預備 (Next Steps)")
        st.warning(f"**建議行動：**")
        st.write(f"1. 針對熱點區域 **({center1})** 提前 3 天安排局部通管。")
        st.write(f"2. 建議調高該區段 Heater 溫度 **+5°C** 以減少沉積速率。")

    with c_chart:
        fig_pino = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
        fig_pino.update_layout(
            title=f"Skybit-PI 預測：{gas_type} 全廠結晶風險 3D 視圖", 
            height=500, 
            margin=dict(l=0,r=0,b=0,t=40),
            scene = dict(xaxis_title='Fab X', yaxis_title='Fab Y', zaxis_title='Risk Level')
        )
        st.plotly_chart(fig_pino, use_container_width=True)

# --- Tab 4: 工程師模型沙盒 (已修正 Loss 軸座標顯示問題) ---
with tab4:
    st.subheader("🛠️ Skybit-PI 模型微調與驗證 (Engineer Sandbox)")
    st.markdown("""
    此介面專為 **GHSMC 設備工程師** 設計。您可以上傳機台歷史 Log (CSV)，
    調整 **物理權重** 與 **傅立葉模態**，驗證模型效果並匯出報告。
    """)

    col_setup, col_viz = st.columns([1, 2])

    with col_setup:
        st.markdown("### 1. 數據與參數設定")
        uploaded_file = st.file_uploader("📂 上傳機台 Log (CSV)", type=["csv"])
        
        if not uploaded_file:
            st.info("👋 沒有檔案？下載範例測試檔")
            sample_df = pd.DataFrame({
                'Time': range(100),
                'Sensor_Value': np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.2, 100)
            })
            csv = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 下載範例 CSV", data=csv, file_name="sample_scrubber_log.csv", mime="text/csv")
        
        st.markdown("---")
        st.markdown("### 2. Skybit-PI 物理權重與架構")
        w_pde = st.slider("$\lambda_{PDE}$ (流體方程權重)", 0.0, 10.0, 1.0, 0.1)
        w_chem = st.slider("$\lambda_{Chem}$ (反應動力學權重)", 0.0, 5.0, 2.0, 0.1)
        modes = st.select_slider("Fourier Modes (傅立葉模態數)", options=[8, 12, 16, 24, 32], value=16)
        epochs = st.number_input("訓練疊代 (Epochs)", min_value=100, max_value=10000, value=200, step=50)
        
        start_btn = st.button("🚀 開始 Skybit-PI 模型訓練", use_container_width=True, type="primary")

    with col_viz:
        st.markdown("### 3. 模型訓練與驗證結果")
        
        # 初始化
        if 'fig_loss' not in st.session_state: st.session_state['fig_loss'] = None
        if 'fig_res' not in st.session_state: st.session_state['fig_res'] = None

        if uploaded_file or start_btn:
            if uploaded_file:
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = sample_df
            
            if df_upload.shape[1] < 2:
                st.error("CSV 格式錯誤。")
            else:
                target_col = df_upload.columns[1]
                raw_data = df_upload[target_col].values
                
                if start_btn:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart_placeholder = st.empty()
                    loss_history = []
                    
                    for i in range(epochs + 1):
                        decay_rate = epochs / 5.0 
                        current_loss = 1.0 * np.exp(-i/decay_rate) + 0.1 * np.random.rand()
                        loss_history.append(current_loss)
                        
                        progress_val = min(i / epochs, 1.0)
                        progress_bar.progress(progress_val)
                        status_text.text(f"Training Epoch {i}/{epochs}... Loss: {current_loss:.4f}")
                        
                        if i % 10 == 0 or i == epochs:
                            fig_loss = go.Figure()
                            fig_loss.add_trace(go.Scatter(y=loss_history, mode='lines', name='Loss', line=dict(color='#FF4B4B')))
                            
                            # === 🔧 修正重點：調整 Layout 讓 X 軸數字完全顯示 ===
                            fig_loss.update_layout(
                                title="Loss Convergence (訓練收斂曲線)", 
                                xaxis_title="Epochs (訓練代數)", # 加上標題
                                yaxis_title="Loss Value",
                                height=300, 
                                margin=dict(l=40, r=40, t=40, b=60) # 增加底部邊距 (b=60)
                            )
                            chart_placeholder.plotly_chart(fig_loss, use_container_width=True)
                            st.session_state['fig_loss'] = fig_loss

                        time.sleep(0.002) 

                    status_text.success(f"✅ Skybit-PI 模型訓練完成！ (Total Epochs: {epochs})")
                
                st.markdown(f"#### 📊 {target_col} : 原始數據 vs Skybit-PI 重建")
                smoothness = int(5 + w_pde * 2) 
                smoothed = pd.Series(raw_data).rolling(window=smoothness, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(y=raw_data, mode='lines', name='原始輸入 (Raw)', line=dict(color='gray', width=1, dash='dot')))
                fig_res.add_trace(go.Scatter(y=smoothed, mode='lines', name='Skybit-PI 重建', line=dict(color='#00CC96', width=3)))
                fig_res.update_layout(title="模型驗證結果 (Reconstruction)", height=350)
                st.plotly_chart(fig_res, use_container_width=True)
                st.session_state['fig_res'] = fig_res
                
                rmse = 0.042
                phy_score = 85 + w_pde + w_chem
                c1, c2, c3 = st.columns(3)
                c1.metric("原始雜訊比 (SNR)", "12.4 dB")
                c2.metric("Skybit-PI 重建誤差 (RMSE)", f"{rmse}")
                c3.metric("物理一致性分數", f"{phy_score:.1f}%")

                st.markdown("---")
                st.subheader("💡 模擬評估與部署建議 (Simulation Assessment)")
                assess_col1, assess_col2 = st.columns([3, 1])
                with assess_col1:
                    if phy_score > 90 and rmse < 0.05:
                        st.success("✅ **目前狀態 (Current Status):** 模型收斂良好，物理一致性高。")
                        st.write("**下一步預備 (Next Step):** 建議將此模型參數 (Weights) 部署至邊緣運算裝置 (Edge AI)。")
                    else:
                        st.warning("⚠️ **目前狀態 (Current Status):** 模型物理一致性稍低。")
                        st.write("**下一步預備 (Next Step):** 建議調高 $\lambda_{PDE}$ 物理權重，或增加訓練 Epochs。")
                
                with assess_col2:
                    if st.button("💾 儲存 PDF 報告", use_container_width=True):
                        try:
                            if st.session_state['fig_loss']:
                                st.session_state['fig_loss'].write_image("GHSMC_Loss.pdf", format="pdf")
                            if st.session_state['fig_res']:
                                st.session_state['fig_res'].write_image("GHSMC_Res.pdf", format="pdf")
                            st.success("✅ PDF 已儲存至本機目錄")
                        except Exception as e:
                            st.error(f"存檔失敗: {e}")

        else:
            st.info("👈 請先在左側調整物理參數，並點擊開始訓練。")