"""
水文模型基类
符合ARCHITECTURE_GUIDE规范的统一接口
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional


class BaseModel(ABC):
    """
    水文模型基类
    支持集总式和分布式模型
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass

    @property
    def model_type(self) -> str:
        """模型类型: 'lumped' 或 'distributed'"""
        return "lumped"

    @property
    def supports_hourly(self) -> bool:
        """模型是否支持小时尺度"""
        return True

    def get_timestep_hours(self, spatial_data: Optional[Dict] = None) -> float:
        """
        获取时间步长(小时)

        Args:
            spatial_data: 空间数据，可包含 'timestep' 键

        Returns:
            时间步长(小时)，默认24即日尺度
        """
        if spatial_data is not None:
            timestep = spatial_data.get('timestep', 'daily')
            if timestep == 'hourly':
                return 1.0
        return 24.0

    @property
    @abstractmethod
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        参数取值范围
        返回: {'param_name': (min, max), ...}
        """
        pass

    @property
    def default_params(self) -> Dict[str, float]:
        """默认参数值（取范围中值）"""
        return {k: (v[0] + v[1]) / 2 for k, v in self.param_bounds.items()}

    @abstractmethod
    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        运行模型

        Args:
            precip: 降水序列 (mm)
                - 集总式: shape (n_timesteps,)
                - 分布式: shape (n_timesteps, n_cells)
            evap: 蒸发序列 (mm)，格式同precip
            params: 模型参数字典
            spatial_data: 空间数据（分布式模型需要）
                {
                    'area': float,           # 流域面积 (km²)
                    'dem': np.ndarray,       # 数字高程
                    'landuse': np.ndarray,   # 土地利用
                    'soil': np.ndarray,      # 土壤类型
                    'flow_direction': np.ndarray,  # 流方向
                    'subcatchments': Dict    # 子流域信息
                }
            temperature: 温度序列 (°C)，格式同precip
                - 用于需要温度输入的模型（如HBV融雪）
                - 可选，部分模型可能需要

        Returns:
            模拟流量序列 (m³/s)
                - 集总式: shape (n_timesteps,)
                - 分布式: shape (n_timesteps, n_outlets)
        """
        pass

    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数是否在有效范围内"""
        for name, (min_val, max_val) in self.param_bounds.items():
            if name not in params:
                return False
            if not (min_val <= params[name] <= max_val):
                return False
        return True

    def get_required_spatial_data(self) -> list:
        """
        获取模型需要的空间数据类型
        子类可重写此方法

        Returns:
            需要的空间数据类型列表，如 ['dem', 'landuse', 'soil']
        """
        return []
