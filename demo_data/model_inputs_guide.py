"""
6个水文模型统一输入数据规格
================================

一、所有模型通用输入
--------------------------------
precip: np.ndarray        # 日降水序列 (mm/day), shape: (n_days,)
evap: np.ndarray           # 日蒸散发序列 (mm/day), shape: (n_days,)

二、模型分类
--------------------------------

【A类】仅需基本输入 (precip, evap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 水箱模型 (SimpleTankModel)
   - 参数: 3个 (h0, alpha, beta)
   
2. 线性水库模型 (LinearReservoirModel)
   - 参数: 2个 (k, c)
   
3. 新安江模型 (XAJModel)
   - 参数: 15个
   - 无需spatial_data

示例:
    model.run(precip, evap, params)


【B类】需要spatial_data (area)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
4. Tank水箱模型 (TankModel)
   - 参数: 16个 (t0_is, t0_boc, t0_soc_lo, ...)
   - spatial_data: {'area': float, 'del_t': float}

示例:
    spatial_data = {'area': 150.7944, 'del_t': 24.0}
    model.run(precip, evap, params, spatial_data)


【C类】需要spatial_data和temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
5. HBV简化模型 (HBVLikeModel)
   - 参数: 5个 (fc, beta, k0, k1, lp)
   
6. HBV模型 (HBVModelAdapter)
   - 参数: 10个 (dd, fc, beta, c, k0, l, k1, k2, kp, pwp)
   - spatial_data: {'area': float, 'monthly_temp': array, 'monthly_pet': array}
   - temperature: np.ndarray (日温度序列, °C)

示例:
    spatial_data = {
        'area': 150.7944,
        'monthly_temp': np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3]),
        'monthly_pet': np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])
    }
    temperature = np.array([...])  # 日温度序列
    model.run(precip, evap, params, spatial_data, temperature)


三、完整数据生成示例
--------------------------------
"""

import numpy as np
import pandas as pd


def create_unified_input_data(n_days: int = 365, area: float = 150.7944) -> dict:
    """
    创建满足所有6个模型运行的统一输入数据
    
    Returns:
        {
            'precip': 日降水 (mm/day),
            'evap': 日蒸散发 (mm/day),
            'temperature': 日温度 (°C),
            'observed_flow': 观测流量 (m³/s),
            'area': 流域面积 (km²),
            'monthly_temp': 月平均温度 (°C),
            'monthly_pet': 月平均PET (mm/day),
            'spatial_data_tank': Tank模型专用,
            'spatial_data_hbv': HBV模型专用,
        }
    """
    np.random.seed(42)
    
    # 月平均温度 (典型温带流域)
    monthly_temp = np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3])
    
    # 月平均PET
    monthly_pet = np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])
    
    # 日数据
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    months = np.array([d.month - 1 for d in dates])
    
    # 日温度
    temperature = np.array([monthly_temp[m] + np.random.randn() * 3 for m in months])
    
    # 日蒸散发
    evap = np.array([monthly_pet[m] * (0.8 + np.random.rand() * 0.4) for m in months])
    
    # 日降水 (随机事件)
    precip = np.where(np.random.rand(n_days) < 0.3, np.random.rand(n_days) * 20 + 1, 0)
    
    # 观测流量 (简化模型生成)
    observed_flow = np.random.rand(n_days) * 50 + 10
    
    return {
        'precip': precip,
        'evap': evap,
        'temperature': temperature,
        'observed_flow': observed_flow,
        'area': area,
        'monthly_temp': monthly_temp,
        'monthly_pet': monthly_pet,
        'spatial_data_tank': {'area': area, 'del_t': 24.0},
        'spatial_data_hbv': {
            'area': area,
            'monthly_temp': monthly_temp,
            'monthly_pet': monthly_pet,
        },
    }


def demo():
    """演示所有模型运行"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.models import ModelRegistry
    from src.hydro_calc import calibrate_model
    
    # 生成统一数据
    data = create_unified_input_data(n_days=100)
    
    precip = data['precip']
    evap = data['evap']
    observed = data['observed_flow']
    
    print("\n" + "=" * 60)
    print("6个水文模型统一输入数据规格")
    print("=" * 60)
    
    print("\n【数据概览】")
    print(f"  流域面积: {data['area']} km2")
    print(f"  时间步长: {len(precip)} 天")
    
    print("\n【模型分类】")
    print("  A类 - 仅需precip, evap:")
    print("    - 水箱模型, 线性水库模型, 新安江模型")
    print("  B类 - 需spatial_data (area):")
    print("    - Tank水箱模型")
    print("  C类 - 需spatial_data + temperature:")
    print("    - HBV简化模型, HBV模型")
    
    print("\n" + "=" * 60)
    print("测试所有模型运行...")
    print("=" * 60)
    
    test_cases = [
        ("水箱模型", {}, None),
        ("线性水库模型", {}, None),
        ("HBV简化模型", data['spatial_data_hbv'], data['temperature']),
        ("Tank水箱模型", data['spatial_data_tank'], None),
        ("HBV模型", data['spatial_data_hbv'], data['temperature']),
        ("新安江模型", {}, None),
    ]
    
    for name, spatial, temp in test_cases:
        try:
            model = ModelRegistry.get_model(name)
            flow = model.run(precip, evap, model.default_params, spatial, temp)
            print(f"  OK: {name} - 流量范围 [{flow.min():.1f}, {flow.max():.1f}]")
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()
