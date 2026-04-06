# -*- coding: utf-8 -*-
"""
Tank水箱模型产流计算模块

参考文献:
- A conceptual rainfall-runoff model considering seasonal variation
  (Kyungrock Paik, Joong H. Kim, Hung S. Kim and Dong R. Lee)
"""

import numpy as np
from typing import Dict, Tuple, Optional


def tank_discharge(
    precipitation: np.ndarray,
    evapotranspiration: np.ndarray,
    del_t: float,
    area: float,
    t0_is: float,
    t0_boc: float,
    t0_soc_uo: float,
    t0_soc_lo: float,
    t0_soh_uo: float,
    t0_soh_lo: float,
    t1_is: float,
    t1_boc: float,
    t1_soc: float,
    t1_soh: float,
    t2_is: float,
    t2_boc: float,
    t2_soc: float,
    t2_soh: float,
    t3_is: float,
    t3_soc: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    运行Tank模型模拟

    参数
    ----------
    precipitation : np.ndarray
        降水序列 (mm)
    evapotranspiration : np.ndarray
        蒸散发序列 (mm)
    del_t : float
        时间步长 (小时)
    area : float
        流域面积 (km²)
    t0_is 到 t3_soc : float
        Tank模型参数

    返回
    -------
    discharge : np.ndarray
        模拟流量 (m³/s)
    states : dict
        模型状态，包含:
        - tank_storage: array of shape (n_timesteps, 4)
        - side_outlet_flow: array of shape (n_timesteps, 4)
        - bottom_outlet_flow: array of shape (n_timesteps, 3)

    单位
    -----
    | area                 | km²         |
    | del_t                | hr          |
    | discharge            | m³/s        |
    | precipitation        | mm          |
    | evapotranspiration   | mm          |
    """
    if precipitation.shape != evapotranspiration.shape:
        raise ValueError(
            f'ERROR-TANK-01: 降水与蒸散发数据长度不匹配'
        )

    time_step = precipitation.shape[0]

    if t0_soh_uo < t0_soh_lo:
        print(
            'WARNING-TANK-01: 上层侧孔高度小于下层侧孔高度 (Tank 0)'
        )

    tank_storage = np.zeros((time_step, 4), dtype=np.float64)
    side_outlet_flow = np.zeros((time_step, 4), dtype=np.float64)
    bottom_outlet_flow = np.zeros((time_step, 3), dtype=np.float64)

    del_rf_et = precipitation - evapotranspiration

    tank_storage[0, 0] = max(t0_is, 0)
    tank_storage[0, 1] = max(t1_is, 0)
    tank_storage[0, 2] = max(t2_is, 0)
    tank_storage[0, 3] = max(t3_is, 0)

    for t in np.arange(time_step):
        side_outlet_flow[t, 0] = (
            t0_soc_lo * max(tank_storage[t, 0] - t0_soh_lo, 0) +
            t0_soc_uo * max(tank_storage[t, 0] - t0_soh_uo, 0)
        )

        side_outlet_flow[t, 1] = t1_soc * max(tank_storage[t, 1] - t1_soh, 0)

        side_outlet_flow[t, 2] = t2_soc * max(tank_storage[t, 2] - t2_soh, 0)

        side_outlet_flow[t, 3] = t3_soc * tank_storage[t, 3]

        bottom_outlet_flow[t, 0] = t0_boc * tank_storage[t, 0]
        bottom_outlet_flow[t, 1] = t1_boc * tank_storage[t, 1]
        bottom_outlet_flow[t, 2] = t2_boc * tank_storage[t, 2]

        if t < (time_step - 1):
            tank_storage[t+1, 0] = (
                tank_storage[t, 0] + del_rf_et[t+1] -
                (side_outlet_flow[t, 0] + bottom_outlet_flow[t, 0])
            )

            tank_storage[t+1, 1] = (
                tank_storage[t, 1] + bottom_outlet_flow[t, 0] -
                (side_outlet_flow[t, 1] + bottom_outlet_flow[t, 1])
            )

            tank_storage[t+1, 2] = (
                tank_storage[t, 2] + bottom_outlet_flow[t, 1] -
                (side_outlet_flow[t, 2] + bottom_outlet_flow[t, 2])
            )

            tank_storage[t+1, 3] = (
                tank_storage[t, 3] + bottom_outlet_flow[t, 2] -
                side_outlet_flow[t, 3]
            )

            tank_storage[t+1, 0] = max(tank_storage[t+1, 0], 0)
            tank_storage[t+1, 1] = max(tank_storage[t+1, 1], 0)
            tank_storage[t+1, 2] = max(tank_storage[t+1, 2], 0)
            tank_storage[t+1, 3] = max(tank_storage[t+1, 3], 0)

        for i in range(4):
            total_outflow = (
                bottom_outlet_flow[t, i] + side_outlet_flow[t, i]
                if i <= 2 else side_outlet_flow[t, i]
            )

            if total_outflow > tank_storage[t, i]:
                pass

    unit_conv_coeff = (area * 1000) / (del_t * 3600)

    discharge = unit_conv_coeff * side_outlet_flow.sum(axis=1)

    states = {
        'tank_storage': tank_storage,
        'side_outlet_flow': side_outlet_flow,
        'bottom_outlet_flow': bottom_outlet_flow
    }

    return discharge, states


class TankModel:
    """Tank模型类 - 面向对象接口"""
    
    def __init__(self, area: float):
        self.area = area
        self.params = {}
        self._set_default_params()

    def _set_default_params(self) -> None:
        from .tank_config import MODEL_PARAMS, TANK_PARAMETER_ORDER
        for param in TANK_PARAMETER_ORDER:
            self.params[param] = MODEL_PARAMS[param]['default']

    def set_params(self, params: Dict[str, float]) -> 'TankModel':
        from .tank_config import MODEL_PARAMS
        for key, value in params.items():
            if key in MODEL_PARAMS:
                min_val = MODEL_PARAMS[key]['min']
                max_val = MODEL_PARAMS[key]['max']
                self.params[key] = np.clip(value, min_val, max_val)
        return self

    def get_params(self) -> Dict[str, float]:
        return self.params.copy()

    def run(
        self,
        precipitation: np.ndarray,
        evapotranspiration: np.ndarray,
        del_t: float = 24.0,
        initial_storage: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        p = self.params

        if initial_storage:
            p = {**self.params, **{
                k.replace('is', 'is'): v
                for k, v in initial_storage.items()
            }}

        return tank_discharge(
            precipitation=precipitation,
            evapotranspiration=evapotranspiration,
            del_t=del_t,
            area=self.area,
            t0_is=p['t0_is'],
            t0_boc=p['t0_boc'],
            t0_soc_uo=p['t0_soc_uo'],
            t0_soc_lo=p['t0_soc_lo'],
            t0_soh_uo=p['t0_soh_uo'],
            t0_soh_lo=p['t0_soh_lo'],
            t1_is=p['t1_is'],
            t1_boc=p['t1_boc'],
            t1_soc=p['t1_soc'],
            t1_soh=p['t1_soh'],
            t2_is=p['t2_is'],
            t2_boc=p['t2_boc'],
            t2_soc=p['t2_soc'],
            t2_soh=p['t2_soh'],
            t3_is=p['t3_is'],
            t3_soc=p['t3_soc'],
        )

    def get_outlet_flows(
        self,
        precipitation: np.ndarray,
        evapotranspiration: np.ndarray,
        del_t: float = 24.0,
    ) -> Dict[str, np.ndarray]:
        _, states = self.run(precipitation, evapotranspiration, del_t)

        return {
            'surface_runoff': states['side_outlet_flow'][:, 0],
            'interflow': states['side_outlet_flow'][:, 1],
            'subbaseflow': states['side_outlet_flow'][:, 2],
            'baseflow': states['side_outlet_flow'][:, 3],
        }