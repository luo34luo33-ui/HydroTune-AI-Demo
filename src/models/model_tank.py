# -*- coding: utf-8 -*-
"""
Tank水箱模型适配器
接入 src/hydro/ 核心计算模块
"""
from typing import Dict, Tuple, Optional
import numpy as np

from .base_model import BaseModel
try:
    from tank_model_structured.src import tank_discharge, MODEL_PARAMS, TANK_PARAMETER_ORDER
except ImportError:
    from src.hydro import tank_discharge, MODEL_PARAMS, TANK_PARAMETER_ORDER


class TankModel(BaseModel):
    """
    Tank水箱模型 (Sugawara & Funiyuki, 1956)
    
    经典的概念性水文模型，通过多层水箱模拟水文过程。
    
    模型特点：
    - 四层串联水箱结构
    - 每层具有侧孔和底孔出流
    - 适用于湿润地区流域模拟
    """

    TANK_PARAMS = TANK_PARAMETER_ORDER

    @property
    def name(self) -> str:
        return "tank水箱模型"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            name: (info['min'], info['max'])
            for name, info in MODEL_PARAMS.items()
            if name in self.TANK_PARAMS
        }

    @property
    def default_params(self) -> Dict[str, float]:
        return {p: MODEL_PARAMS[p]['default'] for p in self.TANK_PARAMS}

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
        warmup_steps: int = 0,
    ) -> np.ndarray:
        """
        运行Tank水箱模型
        
        参数
        ----------
        precip : np.ndarray
            降水序列 (mm), shape: (n_timesteps,)
        evap : np.ndarray
            蒸发序列 (mm), shape: (n_timesteps,)
        params : Dict[str, float]
            模型参数字典
        spatial_data : Dict, optional
            空间数据，默认 {'area': 150.7944, 'del_t': 24.0}
            - area: 流域面积 (km²)
            - del_t: 时间步长 (小时)
            - timestep: 'hourly' 或 'daily'
        temperature : np.ndarray, optional
            温度序列 (°C)，Tank模型不使用此参数
        warmup_steps : int
            预热期步数
        
        返回
        -------
        np.ndarray
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        use_simple = False
        if spatial_data is not None:
            use_simple = spatial_data.get('use_simple_impl', False)
        
        if use_simple:
            from src.hydro import tank_simple
            if spatial_data is None:
                spatial_data = {'area': 150.7944, 'del_t': 24.0}
            area = spatial_data.get('area', 150.7944)
            del_t = spatial_data.get('del_t', 24.0)
            return tank_simple.run_tank_model(precip, evap, params, area, del_t)
        
        if spatial_data is None:
            spatial_data = {'area': 150.7944, 'del_t': 24.0}
        
        area = spatial_data.get('area')
        if area is None:
            raise ValueError("spatial_data 必须包含 'area' 参数（流域面积，单位km²）")
        
        if 'del_t' in spatial_data:
            del_t = spatial_data['del_t']
        else:
            timestep = spatial_data.get('timestep', 'daily')
            del_t = 1.0 if timestep == 'hourly' else 24.0
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        if del_t == 1.0:
            precip_input = precip * 24.0
            evap_input = evap * 24.0
        else:
            precip_input = precip
            evap_input = evap
        
        discharge, _ = tank_discharge(
            precipitation=precip_input,
            evapotranspiration=evap_input,
            del_t=del_t,
            area=area,
            t0_is=full_params['t0_is'],
            t0_boc=full_params['t0_boc'],
            t0_soc_uo=full_params['t0_soc_uo'],
            t0_soc_lo=full_params['t0_soc_lo'],
            t0_soh_uo=full_params['t0_soh_uo'],
            t0_soh_lo=full_params['t0_soh_lo'],
            t1_is=full_params['t1_is'],
            t1_boc=full_params['t1_boc'],
            t1_soc=full_params['t1_soc'],
            t1_soh=full_params['t1_soh'],
            t2_is=full_params['t2_is'],
            t2_boc=full_params['t2_boc'],
            t2_soc=full_params['t2_soc'],
            t2_soh=full_params['t2_soh'],
            t3_is=full_params['t3_is'],
            t3_soc=full_params['t3_soc'],
        )
        
        return discharge

    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            't0_is': 'Tank-0 初始蓄水量 (mm)',
            't0_boc': 'Tank-0 底孔出流系数',
            't0_soc_uo': 'Tank-0 侧孔出流系数（上层）',
            't0_soc_lo': 'Tank-0 侧孔出流系数（下层）',
            't0_soh_uo': 'Tank-0 侧孔高度（上层）(mm)',
            't0_soh_lo': 'Tank-0 侧孔高度（下层）(mm)',
            't1_is': 'Tank-1 初始蓄水量 (mm)',
            't1_boc': 'Tank-1 底孔出流系数',
            't1_soc': 'Tank-1 侧孔出流系数',
            't1_soh': 'Tank-1 侧孔高度 (mm)',
            't2_is': 'Tank-2 初始蓄水量 (mm)',
            't2_boc': 'Tank-2 底孔出流系数',
            't2_soc': 'Tank-2 侧孔出流系数',
            't2_soh': 'Tank-2 侧孔高度 (mm)',
            't3_is': 'Tank-3 初始蓄水量 (mm，基流)',
            't3_soc': 'Tank-3 侧孔出流系数（基流）',
        }