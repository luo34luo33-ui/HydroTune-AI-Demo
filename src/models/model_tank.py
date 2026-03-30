"""
Tank水箱模型适配器
将 tank-model-structured 项目适配到 HydroTune-AI 的统一接口
"""
import importlib.util
import types
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

_tank_base_path = _get_base_path("tank-model-structured")

def _import_tank_module():
    """动态导入Tank模型模块，避免sys.path污染"""
    global tank_discharge, MODEL_PARAMS, TANK_PARAMETER_ORDER, TANK_AVAILABLE
    
    try:
        spec = importlib.util.spec_from_file_location(
            "tank_core", _tank_base_path / "core" / "generation.py"
        )
        if spec and spec.loader:
            tank_core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tank_core)
            tank_discharge = tank_core.tank_discharge
        else:
            raise ImportError("Cannot load core.generation")
        
        spec2 = importlib.util.spec_from_file_location(
            "tank_config", _tank_base_path / "config" / "model_config.py"
        )
        if spec2 and spec2.loader:
            tank_config = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(tank_config)
            MODEL_PARAMS = tank_config.MODEL_PARAMS
            TANK_PARAMETER_ORDER = tank_config.TANK_PARAMETER_ORDER
        else:
            raise ImportError("Cannot load config.model_config")
        
        TANK_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] Tank模型导入失败: {e}")
        TANK_AVAILABLE = False
        tank_discharge = None
        MODEL_PARAMS = None
        TANK_PARAMETER_ORDER = None

tank_discharge = None
MODEL_PARAMS = None
TANK_PARAMETER_ORDER = None
TANK_AVAILABLE = False

from .base_model import BaseModel

_import_tank_module()


class TankModel(BaseModel):
    """
    Tank水箱模型 (Sugawara & Funiyuki, 1956)
    
    经典的概念性水文模型，通过多层水箱模拟水文过程。
    
    模型特点：
    - 四层串联水箱结构
    - 每层具有侧孔和底孔出流
    - 适用于湿润地区流域模拟
    
    参数说明：
    - t0_* 到 t3_*: 各层水箱的初始蓄量和出流参数
    - t*_is: 初始蓄水量
    - t*_boc: 底孔出流系数
    - t*_soc: 侧孔出流系数
    - t*_soh: 侧孔高度
    """

    TANK_PARAMS = [
        't0_is', 't0_boc', 't0_soc_uo', 't0_soc_lo', 't0_soh_uo', 't0_soh_lo',
        't1_is', 't1_boc', 't1_soc', 't1_soh',
        't2_is', 't2_boc', 't2_soc', 't2_soh',
        't3_is', 't3_soc'
    ]

    @property
    def name(self) -> str:
        return "Tank水箱模型(完整版)"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        获取参数取值范围
        """
        if not TANK_AVAILABLE:
            return {
                't0_is': (0.01, 100.0),
                't0_boc': (0.1, 0.5),
                't0_soc_uo': (0.1, 0.5),
                't0_soc_lo': (0.1, 0.5),
                't0_soh_uo': (75.0, 100.0),
                't0_soh_lo': (0.0, 50.0),
                't1_is': (0.01, 100.0),
                't1_boc': (0.01, 0.5),
                't1_soc': (0.01, 0.5),
                't1_soh': (0.0, 100.0),
                't2_is': (0.01, 100.0),
                't2_boc': (0.01, 0.5),
                't2_soc': (0.01, 0.5),
                't2_soh': (0.0, 100.0),
                't3_is': (0.01, 100.0),
                't3_soc': (0.01, 0.5),
            }
        
        if MODEL_PARAMS:
            return {
                name: (info['min'], info['max'])
                for name, info in MODEL_PARAMS.items()
                if name in self.TANK_PARAMS
            }
        return {}

    @property
    def default_params(self) -> Dict[str, float]:
        """获取默认参数值"""
        if not TANK_AVAILABLE:
            return {
                't0_is': 50.0, 't0_boc': 0.3, 't0_soc_uo': 0.3, 't0_soc_lo': 0.3,
                't0_soh_uo': 87.5, 't0_soh_lo': 25.0,
                't1_is': 50.0, 't1_boc': 0.25, 't1_soc': 0.25, 't1_soh': 50.0,
                't2_is': 50.0, 't2_boc': 0.25, 't2_soc': 0.25, 't2_soh': 50.0,
                't3_is': 50.0, 't3_soc': 0.25,
            }
        
        return {p: MODEL_PARAMS[p]['default'] for p in self.TANK_PARAMS}

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        运行Tank水箱模型
        
        Args:
            precip: 降水序列 (mm), shape: (n_timesteps,)
            evap: 蒸发序列 (mm), shape: (n_timesteps,)
            params: 模型参数字典
            spatial_data: 空间数据，可选，默认使用 {'area': 150.7944, 'del_t': 24.0}
                {
                    'area': float,  # 流域面积 (km²)
                    'del_t': float,  # 时间步长 (小时)，默认24
                    'timestep': str  # 'hourly' 或 'daily'，会覆盖 del_t
                }
            temperature: 温度序列 (°C)，Tank模型不使用此参数
            
        Returns:
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        if not TANK_AVAILABLE:
            raise RuntimeError(
                "Tank模型不可用，请确保 tank-model-structured 目录存在且代码完整"
            )
        
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
        
        try:
            discharge, states = tank_discharge(
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
            
        except Exception as e:
            raise RuntimeError(f"Tank模型运行失败: {e}")

    def get_param_descriptions(self) -> Dict[str, str]:
        """获取参数描述"""
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
