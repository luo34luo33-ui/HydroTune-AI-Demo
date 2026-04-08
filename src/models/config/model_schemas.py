# -*- coding: utf-8 -*-
"""
模型Schema定义 - 配置驱动的模型元数据
新模型接入只需配置此项，无需修改代码
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable, Any
from enum import Enum

class ModelCategory(Enum):
    """模型分类"""
    TANK = "tank"
    HBV = "hbv"
    XAJ = "xaj"
    CUSTOM = "custom"

@dataclass
class ModelSchema:
    """模型元数据Schema
    
    新模型接入时，只需创建此配置对象并注册，即可自动获得：
    - 参数边界验证
    - 默认参数提供
    - 率定算法支持
    - BMA集成支持
    - 马斯京根演算支持
    """
    
    # ===== 基本信息 =====
    id: str                           # 唯一标识，如 'tank_v1'
    name: str                         # 显示名称，如 'Tank水箱模型'
    category: ModelCategory           # 模型分类
    description: str = ""             # 模型描述
    
    # ===== 模型能力 =====
    supports_hourly: bool = False      # 是否支持小时尺度
    requires_temperature: bool = False # 是否需要温度输入
    requires_spatial: bool = False     # 是否需要空间数据
    supports_upstream_routing: bool = False  # 是否支持上游演算
    
    # ===== 参数配置 =====
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    param_defaults: Dict[str, float] = field(default_factory=dict)
    param_descriptions: Dict[str, str] = field(default_factory=dict)
    param_units: Dict[str, str] = field(default_factory=dict)
    
    # ===== 约束条件 =====
    param_constraints: Optional[Callable] = None  # 参数约束函数
    
    # ===== 数据配置 =====
    loader_config: Dict[str, Any] = field(default_factory=dict)
    
    # ===== 子模块信息 =====
    module_path: str = ""              # 子模块目录名
    core_function: str = ""            # 核心函数名
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数是否在有效范围内"""
        for name, value in params.items():
            if name not in self.param_bounds:
                continue
            min_val, max_val = self.param_bounds[name]
            if not (min_val <= value <= max_val):
                return False
        
        # 自定义约束
        if self.param_constraints:
            return self.param_constraints(params)
        
        return True
    
    def get_default_params(self) -> Dict[str, float]:
        """获取默认参数"""
        return self.param_defaults.copy()


# ===== Schema注册表 =====
MODEL_SCHEMAS: Dict[str, ModelSchema] = {}

def register_schema(schema: ModelSchema):
    """注册模型Schema"""
    MODEL_SCHEMAS[schema.id] = schema
    print(f"[Schema] Registered: {schema.id} - {schema.name}")

def get_schema(model_id: str) -> Optional[ModelSchema]:
    """获取模型Schema"""
    return MODEL_SCHEMAS.get(model_id)

def get_schema_by_name(name: str) -> Optional[ModelSchema]:
    """根据名称获取模型Schema"""
    for schema in MODEL_SCHEMAS.values():
        if schema.name == name:
            return schema
    return None

def list_schemas() -> List[str]:
    """列出所有已注册的模型Schema ID"""
    return list(MODEL_SCHEMAS.keys())


# ===== 预定义Schema注册 =====
def _register_all_schemas():
    """注册所有内置模型Schema"""
    from .param_bounds import (
        TANK_PARAM_BOUNDS, HBV_PARAM_BOUNDS, XAJ_PARAM_BOUNDS
    )
    from .default_params import (
        TANK_DEFAULT_PARAMS, HBV_DEFAULT_PARAMS, XAJ_DEFAULT_PARAMS
    )
    
    # Tank Schema
    register_schema(ModelSchema(
        id='tank_v1',
        name='tank水箱模型',
        category=ModelCategory.TANK,
        description='Tank水箱模型 (Sugawara & Funiyuki, 1956) - 四层串联水箱结构',
        supports_hourly=True,
        requires_temperature=False,
        requires_spatial=True,
        supports_upstream_routing=True,
        param_bounds=TANK_PARAM_BOUNDS,
        param_defaults=TANK_DEFAULT_PARAMS,
        param_descriptions=_get_tank_descriptions(),
        param_units=_get_tank_units(),
        module_path='tank-model-structured',
        core_function='tank_discharge',
        loader_config={'area_required': True, 'del_t_required': True}
    ))
    
    # HBV Schema
    register_schema(ModelSchema(
        id='hbv_v1',
        name='HBV模型',
        category=ModelCategory.HBV,
        description='HBV水文模型 (Swedish Meteorological and Hydrological Institute) - 包含积雪模块',
        supports_hourly=False,
        requires_temperature=True,
        requires_spatial=True,
        supports_upstream_routing=True,
        param_bounds=HBV_PARAM_BOUNDS,
        param_defaults=HBV_DEFAULT_PARAMS,
        param_descriptions=_get_hbv_descriptions(),
        param_units=_get_hbv_units(),
        module_path='HBV_model_structured',
        core_function='HBVModel',
        loader_config={'area_required': True, 'monthly_data_required': True}
    ))
    
    # XAJ Schema
    register_schema(ModelSchema(
        id='xaj_v1',
        name='新安江模型',
        category=ModelCategory.XAJ,
        description='新安江模型 - 三层蒸散发、蓄满产流、自由水水源划分',
        supports_hourly=True,
        requires_temperature=False,
        requires_spatial=True,
        supports_upstream_routing=True,
        param_bounds=XAJ_PARAM_BOUNDS,
        param_defaults=XAJ_DEFAULT_PARAMS,
        param_descriptions=_get_xaj_descriptions(),
        param_units=_get_xaj_units(),
        module_path='XAJ-model-structured',
        core_function='run_new_xaj',
        loader_config={'area_required': True}
    ))


def _get_tank_descriptions() -> Dict[str, str]:
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

def _get_tank_units() -> Dict[str, str]:
    return {k: '-' for k in _get_tank_descriptions().keys()}

def _get_hbv_descriptions() -> Dict[str, str]:
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

def _get_hbv_units() -> Dict[str, str]:
    return {
        'dd': 'mm/°C/day', 'fc': 'mm', 'beta': '-', 'c': '-',
        'k0': 'day⁻¹', 'l': 'mm', 'k1': 'day⁻¹', 'k2': 'day⁻¹',
        'kp': 'day⁻¹', 'pwp': 'mm'
    }

def _get_xaj_descriptions() -> Dict[str, str]:
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

def _get_xaj_units() -> Dict[str, str]:
    return {
        'k': '-', 'b': '-', 'im': '-', 'um': 'mm', 'lm': 'mm', 'dm': 'mm',
        'c': '-', 'sm': 'mm', 'ex': '-', 'ki': '-', 'kg': '-', 'cs': '-',
        'l': 'timestep', 'ci': '-', 'cg': '-'
    }

# 自动注册所有Schema
_register_all_schemas()