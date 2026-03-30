"""
新安江模型 (XAJ) 适配器
将 XAJ-model-structured 项目适配到 HydroTune-AI 的统一接口
"""
import sys
import importlib.util
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

def _find_xaj_path():
    """查找 XAJ-model-structured 目录的可用路径
    
    使用绝对路径确保在各种环境下都能正确定位
    """
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
        xaj_path = os.path.join(root, "XAJ-model-structured")
        config_path = os.path.join(xaj_path, "config.py")
        if os.path.exists(xaj_path) and os.path.exists(config_path):
            return os.path.abspath(xaj_path)
    
    return None

_xaj_base_path = _find_xaj_path()

def _import_xaj_module():
    """动态导入XAJ模型模块，使用sys.path临时添加"""
    global DEFAULT_PARAMS, PARAM_RANGES, xaj_validate
    global run_new_xaj, prepare_parameters_for_model, XAJ_AVAILABLE
    
    if _xaj_base_path is None:
        print(f"[WARN] XAJ 模型路径不存在: {list(Path('.').glob('*'))}")
        XAJ_AVAILABLE = False
        return
    
    print(f"[INFO] XAJ 模型路径: {_xaj_base_path}")
    
    saved_modules = {}
    for k in list(sys.modules.keys()):
        if ('config' in k.lower() or 'xaj' in k.lower()) and k != __name__:
            if 'src' not in (getattr(sys.modules.get(k), '__file__', '') or ''):
                saved_modules[k] = sys.modules.pop(k, None)
    
    xaj_str = str(_xaj_base_path)
    if xaj_str in sys.path:
        sys.path.remove(xaj_str)
    sys.path.insert(0, xaj_str)
    
    try:
        from config import DEFAULT_PARAMS as _dp, PARAM_RANGES as _pr
        from config import validate_params as _vp
        from main import run_new_xaj as _rx
        from calibration import prepare_parameters_for_model as _pp
        DEFAULT_PARAMS = _dp
        PARAM_RANGES = _pr
        xaj_validate = _vp
        run_new_xaj = _rx
        prepare_parameters_for_model = _pp
        XAJ_AVAILABLE = True
        
        if xaj_str in sys.path:
            sys.path.remove(xaj_str)
        print("[INFO] XAJ 模型加载成功")
    except Exception as e:
        print(f"[WARN] XAJ 模型导入失败: {e}")
        import traceback
        traceback.print_exc()
        XAJ_AVAILABLE = False
        DEFAULT_PARAMS = None
        PARAM_RANGES = None
        xaj_validate = None
        run_new_xaj = None
        prepare_parameters_for_model = None
        if xaj_str in sys.path:
            sys.path.remove(xaj_str)
    finally:
        for k, v in saved_modules.items():
            if k not in sys.modules:
                sys.modules[k] = v

DEFAULT_PARAMS = None
PARAM_RANGES = None
xaj_validate = None
run_new_xaj = None
prepare_parameters_for_model = None
XAJ_AVAILABLE = False

from .base_model import BaseModel

_import_xaj_module()


class XAJModel(BaseModel):
    """
    新安江模型 (XinAnJiang Model)
    
    经典的概念性水文模型，广泛应用于中国湿润地区流域水文预报。
    
    模型特点：
    - 三层蒸散发（上层、下层、深层）
    - 蓄满产流机制
    - 自由水蓄量水源划分（地表、壤中、地下）
    - 河网汇流（CSL方法）
    
    参数说明：
    - k: 蒸散发系数
    - b: 蓄水容量曲线指数
    - im: 不透水面积比例
    - um/lm/dm: 上/下/深层土壤蓄水容量
    - c: 深层蒸散发系数
    - sm: 自由水蓄水容量
    - ex: 自由水容量曲线指数
    - ki/kg: 壤中流/地下水出流系数
    - cs: 河网蓄水常数
    - l: 滞时
    - ci/cg: 壤中流/地下水消退系数
    """

    @property
    def name(self) -> str:
        return "新安江模型"

    @property
    def model_type(self) -> str:
        return "lumped"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        获取参数取值范围
        
        从 XAJ 的 PARAM_RANGES 转换为统一的 (min, max) 元组格式
        """
        if not XAJ_AVAILABLE:
            # 返回默认参数范围（当XAJ不可用时）
            # 注意：ki + kg 必须 < 1，否则模型运行失败
            return {
                'k': (0.5, 1.5),
                'b': (0.1, 0.8),
                'im': (0.0, 0.15),
                'um': (10.0, 60.0),
                'lm': (50.0, 150.0),
                'dm': (40.0, 120.0),
                'c': (0.1, 0.5),
                'sm': (10.0, 80.0),
                'ex': (1.0, 3.0),
                'ki': (0.05, 0.45),      # 确保 ki + kg < 0.9
                'kg': (0.05, 0.45),     # 确保 ki + kg < 0.9
                'cs': (0.5, 0.98),
                'l': (0, 20),
                'ci': (0.5, 0.98),
                'cg': (0.9, 0.999),
            }
        
        # XAJ 可用时，修改 PARAM_RANGES 确保约束
        return {
            'k': (0.5, 1.5),
            'b': (0.1, 0.8),
            'im': (0.0, 0.15),
            'um': (10.0, 60.0),
            'lm': (50.0, 150.0),
            'dm': (40.0, 120.0),
            'c': (0.1, 0.5),
            'sm': (10.0, 80.0),
            'ex': (1.0, 3.0),
            'ki': (0.05, 0.45),         # 修改：确保 ki + kg < 0.9
            'kg': (0.05, 0.45),         # 修改：确保 ki + kg < 0.9
            'cs': (0.5, 0.98),
            'l': (0, 20),
            'ci': (0.5, 0.98),
            'cg': (0.9, 0.999),
        }

    @property
    def default_params(self) -> Dict[str, float]:
        """获取默认参数值"""
        if not XAJ_AVAILABLE:
            return {
                'k': 0.8, 'b': 0.3, 'im': 0.01,
                'um': 20.0, 'lm': 70.0, 'dm': 60.0, 'c': 0.15,
                'sm': 20.0, 'ex': 1.5, 'ki': 0.3, 'kg': 0.4,
                'cs': 0.8, 'l': 1, 'ci': 0.8, 'cg': 0.98,
            }
        return DEFAULT_PARAMS.copy()

    def run(
        self,
        precip: np.ndarray,
        evap: np.ndarray,
        params: Dict[str, float],
        spatial_data: Optional[Dict] = None,
        temperature: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        运行新安江模型
        
        Args:
            precip: 降水序列 (mm), shape: (n_timesteps,)
            evap: 蒸发序列 (mm), shape: (n_timesteps,)
            params: 模型参数字典，包含15个参数
            spatial_data: 空间数据，可选，默认 {'area': 150.7944}
                {
                    'area': float,  # 流域面积 (km²)
                    'timestep': str  # 'hourly' 或 'daily'
                }
            
        Returns:
            模拟流量序列 (m³/s), shape: (n_timesteps,)
        """
        if not XAJ_AVAILABLE:
            raise RuntimeError(
                "XAJ 模型不可用，请确保 XAJ-model-structured 目录存在且代码完整"
            )
        
        area = 150.7944 if spatial_data is None else spatial_data.get('area', 150.7944)
        timestep_hours = self.get_timestep_hours(spatial_data)
        
        full_params = self.default_params.copy()
        full_params.update(params)
        
        if full_params['ki'] + full_params['kg'] >= 1.0:
            scale = 0.99 / (full_params['ki'] + full_params['kg'])
            full_params['ki'] *= scale
            full_params['kg'] *= scale
        
        n = len(precip)
        p_and_e = np.zeros((n, 1, 2))
        p_and_e[:, 0, 0] = precip
        p_and_e[:, 0, 1] = evap
        
        try:
            q_sim, es = run_new_xaj(p_and_e, full_params, warmup_length=0, return_state=False)
            
            runoff_mm = q_sim[:, 0, 0]
            
            seconds_per_step = timestep_hours * 3600
            flow = runoff_mm * area * 1000 / seconds_per_step
            
            return flow
            
        except Exception as e:
            raise RuntimeError(f"XAJ 模型运行失败: {e}")

    def validate_params(self, params: Dict[str, float]) -> bool:
        """
        验证参数是否有效
        
        新安江模型特殊约束：
        - ki + kg < 1
        - w0 = 0.6 * (um + lm + dm) < (um + lm + dm) = wm
        - b > 0 (保证指数计算有效)
        """
        # 基础范围检查
        if not super().validate_params(params):
            return False
        
        # ki + kg 约束
        if 'ki' in params and 'kg' in params:
            if params['ki'] + params['kg'] >= 1.0:
                return False
        
        # 确保内部状态计算有效: w0 = 0.6*wm < wm
        # 这总是成立的，只要 um, lm, dm > 0
        
        return True

    def get_param_descriptions(self) -> Dict[str, str]:
        """获取参数描述"""
        return {
            'k': '蒸散发系数：潜在蒸散发与参考作物蒸发的比值',
            'b': '蓄水容量曲线指数：反映流域蓄水容量分布不均匀性',
            'im': '不透水面积比例：流域内不透水面积占总面积的比例',
            'um': '上层土壤蓄水容量：上层张力水最大蓄水容量(mm)',
            'lm': '下层土壤蓄水容量：下层张力水最大蓄水容量(mm)',
            'dm': '深层土壤蓄水容量：深层张力水最大蓄水容量(mm)',
            'c': '深层蒸散发系数：深层蒸散发与下层蒸散发的比值',
            'sm': '自由水蓄水容量：表层自由水平均蓄水容量(mm)',
            'ex': '自由水容量曲线指数：反映自由水容量分布不均匀性',
            'ki': '壤中流出流系数：自由水蓄量向壤中流的出流比例',
            'kg': '地下水出流系数：自由水蓄量向地下水的出流比例',
            'cs': '河网蓄水常数：河网汇流的蓄水常数',
            'l': '滞时：河网汇流的滞时(时间步长数)',
            'ci': '壤中流消退系数：壤中流的消退系数',
            'cg': '地下水消退系数：地下水的消退系数',
        }
