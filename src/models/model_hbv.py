"""
HBV水文模型适配器
将 HBV_model_structured 项目适配到 HydroTune-AI 的统一接口
"""
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

def _get_base_path(folder_name):
    """根据当前文件位置获取项目根目录下的子模块路径"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    possible_roots = [
        os.path.dirname(os.path.dirname(current_dir)),
        os.path.dirname(current_dir),
        os.getcwd(),
        "/mount/src/hydrotune-ai-demo",
        "/app",
    ]
    for root in possible_roots:
        path = os.path.join(root, folder_name)
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

_hbv_base_path = _get_base_path("HBV_model_structured")

def _import_hbv_module():
    """动态导入HBV模型模块，临时添加sys.path并清理缓存"""
    global ModelConfig, MODEL_PARAMS_HBV, get_param_ranges, get_default_params
    global HBVModel, DailyInputData, MonthlyClimateData, HBV_AVAILABLE
    
    saved_modules = {}
    hbv_path_str = str(_hbv_base_path)
    for k in list(sys.modules.keys()):
        if k != __name__:
            mod = sys.modules.get(k)
            mod_file = getattr(mod, '__file__', None) or ''
            if hbv_path_str in mod_file or k.split('.')[0] in ['config', 'core', 'data', 'utils', 'evaluation', 'visualization']:
                saved_modules[k] = sys.modules.pop(k, None)
    
    if hbv_path_str in sys.path:
        sys.path.remove(hbv_path_str)
    sys.path.insert(0, hbv_path_str)
    
    try:
        import config.model_config as hbv_config
        ModelConfig = hbv_config.ModelConfig
        MODEL_PARAMS_HBV = hbv_config.MODEL_PARAMS
        get_param_ranges = hbv_config.get_param_ranges
        get_default_params = hbv_config.get_default_params
        
        import core.model as hbv_core
        HBVModel = hbv_core.HBVModel
        
        import data.loader as hbv_data
        DailyInputData = hbv_data.DailyInputData
        MonthlyClimateData = hbv_data.MonthlyClimateData
        
        HBV_AVAILABLE = True
        
        sys.path.remove(hbv_path_str)
        
        for k in list(sys.modules.keys()):
            mod = sys.modules.get(k)
            mod_file = getattr(mod, '__file__', None) or ''
            if hbv_path_str in mod_file:
                del sys.modules[k]
    except Exception as e:
        print(f"[WARN] HBV模型导入失败: {e}")
        HBV_AVAILABLE = False
        ModelConfig = None
        get_param_ranges = None
        get_default_params = None
        HBVModel = None
        DailyInputData = None
        MonthlyClimateData = None
        if hbv_path_str in sys.path:
            sys.path.remove(hbv_path_str)

import types
ModelConfig = None
MODEL_PARAMS_HBV = None
get_param_ranges = None
get_default_params = None
HBVModel = None
DailyInputData = None
MonthlyClimateData = None
HBV_AVAILABLE = False

from .base_model import BaseModel

_import_hbv_module()


class HBVModelAdapter(BaseModel):
    """
    HBV (Hydrologiska Byråns Vattenbalans) 水文模型
    
    经典的概念性水文模型，由瑞典水文气象局开发。
    
    模型特点：
    - 积雪模块（温度指数法）
    - 土壤水分模块
    - 响应函数模块（三个线性水库）
    - 汇流模块
    
    参数说明：
    - dd: 度日因子 (mm/°C/day)
    - fc: 田间持水量 (mm)
    - beta: 形状参数
    - c: 潜在蒸散发温度校正系数
    - k0, k1, k2: 各层出流系数
    - l: 表层储水阈值 (mm)
    - kp: 储水间交换系数
    - pwp: 永久萎蔫点 (mm)
    """
    
    DEFAULT_MONTHLY_TEMP = np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3])
    DEFAULT_MONTHLY_PET = np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])

    @property
    def name(self) -> str:
        return "HBV模型"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def supports_hourly(self) -> bool:
        return False

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        获取参数取值范围
        """
        if not HBV_AVAILABLE:
            return {
                'dd': (3.0, 7.0),
                'fc': (100.0, 200.0),
                'beta': (1.0, 7.0),
                'c': (0.01, 0.07),
                'k0': (0.05, 0.2),
                'l': (2.0, 5.0),
                'k1': (0.01, 0.1),
                'k2': (0.01, 0.05),
                'kp': (0.01, 0.05),
                'pwp': (90.0, 180.0),
            }
        
        return get_param_ranges()

    @property
    def default_params(self) -> Dict[str, float]:
        """获取默认参数值"""
        if not HBV_AVAILABLE:
            return {
                'dd': 6.10, 'fc': 195.0, 'beta': 2.6143, 'c': 0.07,
                'k0': 0.163, 'l': 4.87, 'k1': 0.029, 'k2': 0.049,
                'kp': 0.050, 'pwp': 106.0,
            }
        
        return get_default_params()

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
        运行HBV水文模型
        
        Args:
            precip: 降水序列 (mm/day), shape: (n_timesteps,)
            evap: 潜在蒸散发序列 (mm/day), shape: (n_timesteps,)
            params: 模型参数字典
            spatial_data: 空间数据，可选，默认使用典型温带流域参数
                {
                    'area': float,  # 流域面积 (km²)
                    'monthly_temp': np.ndarray,  # 月平均温度 (°C)，长度12
                    'monthly_pet': np.ndarray,   # 月平均PET (mm/day)，长度12
                }
            temperature: 温度序列 (°C), shape: (n_timesteps,)，可选
            warmup_steps: 预热期步数
            
        Returns:
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        use_simple = False
        if spatial_data is not None:
            use_simple = spatial_data.get('use_simple_impl', False)
        
        if use_simple:
            from src.hydro import hbv_simple
            area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
            return hbv_simple.run_hbv_model(precip, evap, params, area)
        
        if not HBV_AVAILABLE:
            raise RuntimeError(
                "HBV模型不可用，请确保 HBV_model_structured 目录存在且代码完整"
            )
        
        # 默认spatial_data（兼容简化调用）
        if spatial_data is None:
            spatial_data = {
                'area': 150.7944,
                'monthly_temp': self.DEFAULT_MONTHLY_TEMP,
                'monthly_pet': self.DEFAULT_MONTHLY_PET,
            }
        
        area = spatial_data.get('area')
        if area is None:
            area = 150.7944  # 默认值
        
        monthly_temp = spatial_data.get('monthly_temp')
        monthly_pet = spatial_data.get('monthly_pet')
        
        if monthly_temp is None:
            monthly_temp = self.DEFAULT_MONTHLY_TEMP
        if monthly_pet is None:
            monthly_pet = self.DEFAULT_MONTHLY_PET
        
        # 估算温度（如果未提供）
        if temperature is None:
            temperature = self._estimate_temperature(precip, evap)
        
        n_days = len(precip)
        
        time_arr = np.arange(n_days)
        month_arr = (time_arr % 365) // 30
        month_arr = np.clip(month_arr, 0, 11)
        
        daily_input = DailyInputData(
            time=time_arr,
            month=month_arr.astype(int),
            temperature=temperature,
            precipitation=precip,
            evaporation=evap,
            n_days=n_days
        )
        
        monthly_input = MonthlyClimateData(
            month_avg_temp=monthly_temp,
            monthly_pet_total=monthly_pet * 30,
            daily_pet_avg=monthly_pet
        )
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        try:
            config = ModelConfig(params=full_params)
            model = HBVModel(config=config, catchment_area=area)
            
            output = model.run(daily_input, monthly_input)
            
            return output.flow_cms
            
        except Exception as e:
            raise RuntimeError(f"HBV模型运行失败: {e}")

    def get_param_descriptions(self) -> Dict[str, str]:
        """获取参数描述"""
        return {
            'dd': '度日因子 - 雪融化速率 (mm/°C/day)',
            'fc': '田间持水量 - 土壤最大蓄水容量 (mm)',
            'beta': '形状参数 - 控制产流的非线性程度',
            'c': '潜在蒸散发温度校正系数',
            'k0': '表层储水出流系数 - 快速径流响应 (day⁻¹)',
            'l': '表层储水阈值 (mm)',
            'k1': '上层储水出流系数 - 慢速径流响应 (day⁻¹)',
            'k2': '下层储水出流系数 - 基流响应 (day⁻¹)',
            'kp': '储水间交换系数 - 上下层储水交换 (day⁻¹)',
            'pwp': '永久萎蔫点 - 土壤水分下限 (mm)',
        }
    
    def _estimate_temperature(self, precip: np.ndarray, evap: np.ndarray) -> np.ndarray:
        """
        从降水和蒸发估算温度序列
        
        这是一个简化的估算方法，用于在没有实测温度数据时使用。
        假设夏季温度高、冬季温度低，与月平均温度模式一致。
        
        Args:
            precip: 降水序列
            evap: 蒸发序列
            
        Returns:
            估算的温度序列 (°C)
        """
        n_days = len(precip)
        dates = np.arange(n_days)
        month_arr = (dates % 365) // 30
        month_arr = np.clip(month_arr, 0, 11)
        
        return np.array([self.DEFAULT_MONTHLY_TEMP[m] + np.random.randn() * 2 
                         for m in month_arr])
