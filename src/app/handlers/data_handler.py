# -*- coding: utf-8 -*-
"""
数据处理模块
从 app.py 数据处理部分提取
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any


class DataHandler:
    """数据处理器"""
    
    @staticmethod
    def read_files(uploaded_files) -> List[Tuple[str, pd.DataFrame]]:
        """读取上传的文件"""
        file_dfs = []
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                file_dfs.append((uploaded_file.name, df))
            except Exception as e:
                st.error(f"读取 {uploaded_file.name} 失败: {e}")
                continue
        return file_dfs
    
    @staticmethod
    def process_files(file_dfs: List, column_mapping: Dict) -> Dict[str, Any]:
        """处理所有文件的数据"""
        all_precip = []
        all_evap = []
        all_flow = []
        all_file_events = []
        
        for file_idx, (file_name, raw_df) in enumerate(file_dfs):
            df = raw_df.copy()
            
            for std_name, orig_name in column_mapping.items():
                if orig_name and orig_name in df.columns:
                    df = df.rename(columns={orig_name: std_name})
            
            if 'precip' not in df.columns or 'flow' not in df.columns:
                st.error(f"缺少必要列。当前列: {list(df.columns)}")
                continue
            
            if 'evap' not in df.columns:
                df['evap'] = 0.0
            
            if 'date' not in df.columns:
                df['date'] = range(len(df))
            
            df = df.fillna(0)
            
            precip_arr = np.array(df['precip'].values)
            flow_arr = np.array(df['flow'].values)
            evap_arr = np.array(df['evap'].values)
            
            all_precip.extend(precip_arr.tolist())
            all_evap.extend(evap_arr.tolist())
            all_flow.extend(flow_arr.tolist())
        
        return {
            'precip': np.array(all_precip),
            'evap': np.array(all_evap),
            'flow': np.array(all_flow),
            'file_events': all_file_events,
        }
    
    @staticmethod
    def detect_timestep(dates) -> str:
        """检测时间尺度"""
        from src.data_agent import infer_timestep_by_llm
        from src.llm_api import call_minimax
        return infer_timestep_by_llm(dates, call_minimax)