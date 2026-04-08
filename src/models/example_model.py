"""
示例水文模型
用于Demo演示，你可以参考这些模板来添加你的真实模型

.. deprecated::
    简化模型已废弃，请使用完整模型代替：
    - 水箱模型 → Tank水箱模型 (Tank水箱模型)
    - HBV模型 → HBV模型
"""
import warnings
import numpy as np
from typing import Dict, Tuple, Optional
from .base_model import BaseModel


def runoff_depth_to_flow(runoff_mm: np.ndarray, area: float, timestep_hours: float = 24.0) -> np.ndarray:
    """
    将径流深(mm)转换为流量(m³/s)

    Args:
        runoff_mm: 径流深 (mm)
        area: 流域面积 (km²)
        timestep_hours: 时间步长(小时)，默认24即日尺度

    Returns:
        流量 (m³/s)
    """
    seconds_per_step = timestep_hours * 3600
    return runoff_mm * area * 1000 / seconds_per_step


class SimpleTankModel(BaseModel):
    """
    简单水箱模型（Single Tank Model）[已废弃]
    
    改进版：双层调蓄机制，模拟地表径流和基流，使输出更平滑。
    
    .. deprecated::
        请使用 Tank水箱模型 代替
        推荐调用方式: ModelRegistry.get_model("Tank水箱模型")
    """

    @property
    def name(self) -> str:
        return "水箱模型"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "k1": (0.01, 0.3),    # 快速流调蓄系数
            "k2": (0.001, 0.05),  # 慢速流调蓄系数（基流）
            "c": (0.01, 0.3),    # 产流系数
        }

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
        timestep_hours = self.get_timestep_hours(spatial_data)
        
        n = len(precip)
        Q = np.zeros(n)
        
        S1 = 0.0
        S2 = 0.0
        k1 = params["k1"]
        k2 = params["k2"]
        c = params["c"]

        for i in range(n):
            Pe = max(precip[i] - evap[i], 0)
            R = c * Pe
            S1 = S1 + R
            Q1 = k1 * S1
            S1 = S1 - Q1
            
            S2 = S2 + Q1
            Q2 = k2 * S2
            S2 = S2 - Q2
            
            Q[i] = Q1 + Q2

        return runoff_depth_to_flow(Q, area, timestep_hours)


class LinearReservoirModel(BaseModel):
    """
    线性水库模型
    经典的水文汇流模型
    """

    @property
    def name(self) -> str:
        return "线性水库模型"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "k": (0.1, 10.0),  # 蓄水常数 (days)
            "c": (0.01, 0.99),  # 产流系数
        }

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
        timestep_hours = self.get_timestep_hours(spatial_data)
        
        n = len(precip)
        Q = np.zeros(n)
        S = 0

        k = params["k"]
        c = params["c"]

        for i in range(n):
            P_eff = max(precip[i] - evap[i], 0)
            S = S + c * P_eff
            Q[i] = S / k
            S = S - Q[i]
            S = max(S, 0)

        return runoff_depth_to_flow(Q, area, timestep_hours)


class HBVLikeModel(BaseModel):
    """
    HBV模型（简化版）
    
    包含土壤水分和响应函数的简化版本，适合日尺度和小时尺度。
    支持时间尺度自动适配。
    """
    
    @property
    def name(self) -> str:
        return "HBV模型"
    
    @property
    def supports_hourly(self) -> bool:
        return True

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "fc": (50.0, 500.0),  # 田间持水量 (mm)
            "beta": (1.0, 5.0),  # 形状系数
            "k0": (0.01, 0.5),  # 快速出流系数
            "k1": (0.001, 0.1),  # 慢速出流系数
            "lp": (0.3, 1.0),  # 蒸散发限制
        }

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
        timestep_hours = self.get_timestep_hours(spatial_data)
        
        n = len(precip)
        Q = np.zeros(n)

        fc = params["fc"]
        beta = params["beta"]
        k0 = params["k0"]
        k1 = params["k1"]
        lp = params["lp"]

        sm = fc * 0.5
        su = 0
        sl = 0

        for i in range(n):
            ea = evap[i] * min(sm / (lp * fc), 1.0)
            sm = sm + precip[i] - ea

            if sm > fc:
                r = sm - fc
                sm = fc
            else:
                r = precip[i] * (sm / fc) ** beta

            sm = sm - r
            sm = np.clip(sm, 0, fc)

            su = su + r * 0.7
            sl = sl + r * 0.3

            q0 = k0 * su
            q1 = k1 * sl
            Q[i] = q0 + q1

            su = su - q0
            sl = sl - q1
            su = max(su, 0)
            sl = max(sl, 0)

        return runoff_depth_to_flow(Q, area, timestep_hours)
