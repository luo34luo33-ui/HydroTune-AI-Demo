# -*- coding: utf-8 -*-
"""
模型率定模块
从 app.py 率定部分提取
"""
import streamlit as st
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class CalibrationHandler:
    """模型率定处理器"""
    
    @staticmethod
    def calibrate_model(
        model_name: str,
        precip: np.ndarray,
        evap: np.ndarray,
        flow: np.ndarray,
        max_iter: int,
        spatial_data: Dict,
        timestep: str,
        algorithm: str,
        algo_params: Dict,
        upstream_flow: Optional[np.ndarray] = None,
        enable_routing: bool = False,
        manual_routing_params: Optional[Dict] = None,
        progress_callback=None,
    ) -> Tuple[Dict, float, np.ndarray]:
        """率定模型"""
        from src.hydro_calc import calibrate_model_fast
        
        return calibrate_model_fast(
            model_name=model_name,
            precip=precip,
            evap=evap,
            observed_flow=flow,
            max_iter=max_iter,
            spatial_data=spatial_data,
            timestep=timestep,
            algorithm=algorithm,
            algo_params=algo_params,
            upstream_flow=upstream_flow,
            enable_routing=enable_routing,
            manual_routing_params=manual_routing_params,
            progress_callback=progress_callback,
        )
    
    @staticmethod
    def run_all_models(
        models: List[str],
        calib_events: List[Dict],
        max_iter: int,
        spatial_data: Dict,
        timestep: str,
        algorithm: str,
        algo_params: Dict,
        enable_routing: bool,
        manual_routing_params: Optional[Dict] = None,
        warmup_steps: int = 0,
    ) -> Dict[str, Dict]:
        """运行所有模型的率定"""
        from src.hydro_calc import calibrate_model_fast
        
        results = {}
        progress_bar = st.progress(0)
        
        for model_idx, model_name in enumerate(models):
            st.write(f"  🔄 开始率定 {model_name}...")
            
            spatial_data_copy = spatial_data.copy()
            
            manual_routing = None
            if enable_routing:
                manual_routing = manual_routing_params
            
            try:
                result = calibrate_model_fast(
                    model_name,
                    calib_events[0]['precip'],
                    calib_events[0]['evap'],
                    calib_events[0]['flow'],
                    max_iter=max_iter,
                    spatial_data=spatial_data_copy,
                    timestep=timestep,
                    algorithm=algorithm,
                    algo_params=algo_params,
                    upstream_flow=calib_events[0].get('upstream'),
                    enable_routing=enable_routing,
                    calib_events=calib_events,
                    warmup_steps=warmup_steps,
                    progress_callback=lambda p: progress_bar.progress((model_idx + p) / len(models)),
                    manual_routing_params=manual_routing
                )
                
                params, nse, simulated = result
                results[model_name] = {
                    "model_name": model_name,
                    "params": params,
                    "nse": nse,
                    "simulated": simulated,
                }
                st.write(f"  ✅ {model_name}: NSE={nse:.4f}")
                
            except Exception as e:
                st.error(f"  ⚠️ {model_name} 率定异常: {e}")
                results[model_name] = None
            
            progress_bar.progress((model_idx + 1) / len(models))
        
        return results