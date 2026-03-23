# monitor_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import os

st.set_page_config(page_title="MetaD Real-time Monitor", layout="wide")
st.title("实时 CV 监控仪 (心电图模式)")

# 侧边栏配置
file_path = st.sidebar.text_input("数据文件路径", "HILLS")
refresh_rate = st.sidebar.slider("刷新频率 (秒)", 1, 10, 2)
show_points = st.sidebar.number_input("显示点数", 100, 5000, 500)

# 创建占位符
chart_placeholder = st.empty()
metrics_placeholder = st.columns(3)

while True:
    if os.path.exists(file_path):
        try:
            # 读取数据
            data = pd.read_csv(file_path, sep='\s+', header=None, comment='#')
            cv = data.iloc[:, 1].values
            
            # 计算均值和标准差 (针对最近的数据)
            recent_cv = cv[-show_points:] if len(cv) > show_points else cv
            current_mean = np.mean(recent_cv)
            current_std = np.std(recent_cv)

            # 更新上方指标
            metrics_placeholder[0].metric("当前步数", len(cv))
            metrics_placeholder[1].metric("最近均值", f"{current_mean:.4f}")
            metrics_placeholder[2].metric("最近标准差", f"{current_std:.4f}")

            # 绘制滚动图
            chart_placeholder.line_chart(recent_cv)
            
        except Exception as e:
            st.error(f"读取失败: {e}")
    else:
        st.warning("等待 HILLS 文件生成...")

    time.sleep(refresh_rate)