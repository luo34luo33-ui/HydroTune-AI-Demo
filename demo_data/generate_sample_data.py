"""
示例数据生成器
生成满足所有6个模型运行的统一输入数据
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_data(
    n_days: int = 365,
    area: float = 150.7944,
    seed: int = 42
) -> dict:
    """
    生成示例水文数据，满足所有模型的输入需求
    
    Args:
        n_days: 天数
        area: 流域面积 (km²)
        seed: 随机种子
        
    Returns:
        dict: 包含所有模型输入数据的字典
    """
    np.random.seed(seed)
    
    # 时间序列
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # 月份 (0-11)
    months = np.array([d.month - 1 for d in dates])
    
    # 月平均温度 (°C) - 典型温带流域
    monthly_temp = np.array([2, 5, 10, 15, 20, 25, 28, 27, 22, 15, 8, 3])
    
    # 日温度序列 (°C)
    temperature = np.array([monthly_temp[m] + np.random.randn() * 3 for m in months])
    
    # 月平均PET (mm/day)
    monthly_pet = np.array([1.0, 1.5, 2.5, 4.0, 5.5, 6.5, 7.0, 6.5, 5.0, 3.0, 1.5, 1.0])
    
    # 日蒸散发序列 (mm/day)
    evap = np.array([
        monthly_pet[months[i]] * (0.8 + np.random.rand() * 0.4)
        for i in range(n_days)
    ])
    
    # 日降水序列 (mm/day) - 随机生成
    precip = np.zeros(n_days)
    for i in range(n_days):
        if np.random.rand() < 0.3:  # 30%概率有降水
            precip[i] = np.random.rand() * 20 + 1
    
    # 展平连续降水事件
    for i in range(1, n_days):
        if precip[i] > 0 and precip[i-1] > 0:
            precip[i] = precip[i-1] * 0.3 + precip[i] * 0.7
    
    # 观测流量 (m³/s) - 使用简单线性模型生成
    # Q = k * P * area / 86400 + baseflow
    k_runoff = 0.05  # 径流系数
    baseflow = area * 0.01  # 基流
    observed_flow = np.zeros(n_days)
    
    for i in range(n_days):
        effective_precip = max(precip[i] - evap[i], 0)
        runoff = effective_precip * k_runoff * area * 1000 / 86400
        observed_flow[i] = runoff + baseflow + np.random.randn() * 5
        observed_flow[i] = max(observed_flow[i], 0)
    
    # 平滑处理
    from scipy.ndimage import uniform_filter1d
    observed_flow = uniform_filter1d(observed_flow, size=3)
    
    # 构建spatial_data
    spatial_data_tank = {
        'area': area,
        'del_t': 24.0,  # 日尺度
    }
    
    spatial_data_hbv = {
        'area': area,
        'monthly_temp': monthly_temp,
        'monthly_pet': monthly_pet,
    }
    
    # 返回统一格式
    return {
        'dates': dates,
        'n_days': n_days,
        'precip': precip,           # 日降水 (mm/day)
        'evap': evap,               # 日蒸散发 (mm/day)
        'temperature': temperature,  # 日温度 (°C)
        'observed_flow': observed_flow,  # 观测流量 (m³/s)
        'area': area,               # 流域面积 (km²)
        'monthly_temp': monthly_temp,    # 月平均温度 (°C)
        'monthly_pet': monthly_pet,     # 月平均PET (mm/day)
        'spatial_data_tank': spatial_data_tank,
        'spatial_data_hbv': spatial_data_hbv,
    }


def save_to_csv(data: dict, output_dir: str = './demo_data'):
    """保存数据到CSV文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    df = pd.DataFrame({
        'date': data['dates'],
        'precip': data['precip'],
        'evap': data['evap'],
        'temperature': data['temperature'],
        'observed_flow': data['observed_flow'],
    })
    
    df.to_csv(output_path / 'sample_data.csv', index=False)
    print(f"数据已保存到 {output_path / 'sample_data.csv'}")
    
    # 保存月数据
    monthly_df = pd.DataFrame({
        'month': range(1, 13),
        'monthly_temp': data['monthly_temp'],
        'monthly_pet': data['monthly_pet'],
    })
    monthly_df.to_csv(output_path / 'monthly_data.csv', index=False)
    print(f"月数据已保存到 {output_path / 'monthly_data.csv'}")


def demo_all_models():
    """演示所有6个模型的运行"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.models import ModelRegistry
    from src.hydro_calc import calibrate_model
    
    print("=" * 60)
    print("生成示例数据...")
    data = generate_sample_data(n_days=365, area=150.7944)
    
    precip = data['precip']
    evap = data['evap']
    observed_flow = data['observed_flow']
    
    print(f"\n数据概览:")
    print(f"  时间步长: {data['n_days']} 天")
    print(f"  流域面积: {data['area']} km²")
    print(f"  降水范围: [{precip.min():.1f}, {precip.max():.1f}] mm")
    print(f"  蒸散发范围: [{evap.min():.1f}, {evap.max():.1f}] mm")
    print(f"  温度范围: [{data['temperature'].min():.1f}, {data['temperature'].max():.1f}] °C")
    
    print("\n" + "=" * 60)
    print("测试所有模型运行...")
    print("=" * 60)
    
    models_info = [
        ("水箱模型", {}, None),
        ("线性水库模型", {}, None),
        ("HBV简化模型", {}, None),
        ("Tank水箱模型", data['spatial_data_tank'], None),
        ("HBV模型", data['spatial_data_hbv'], data['temperature']),
        ("新安江模型", {}, None),
    ]
    
    results = []
    for model_name, spatial_data, temp in models_info:
        try:
            # 使用默认参数运行
            model = ModelRegistry.get_model(model_name)
            params = model.default_params
            
            flow = model.run(precip, evap, params, spatial_data, temp)
            
            # 计算NSE（如果观测流量合理）
            from src.hydro_calc import calc_nse
            nse = calc_nse(observed_flow, flow)
            
            print(f"\n✓ {model_name}")
            print(f"  参数数量: {len(params)}")
            print(f"  流量范围: [{flow.min():.2f}, {flow.max():.2f}] m³/s")
            print(f"  与观测NSE: {nse:.4f}")
            
            results.append({
                'model': model_name,
                'status': 'success',
                'nse': nse,
                'flow_range': (flow.min(), flow.max()),
            })
            
        except Exception as e:
            print(f"\n✗ {model_name}: {e}")
            results.append({
                'model': model_name,
                'status': 'failed',
                'error': str(e),
            })
    
    print("\n" + "=" * 60)
    print("测试率定功能 (每个模型5次迭代)...")
    print("=" * 60)
    
    for model_name, spatial_data, temp in models_info:
        try:
            params, nse, flow = calibrate_model(
                model_name, precip, evap, observed_flow,
                max_iter=5, spatial_data=spatial_data, temperature=temp
            )
            print(f"\n✓ {model_name} 率定成功")
            print(f"  最优NSE: {nse:.4f}")
        except Exception as e:
            print(f"\n✗ {model_name} 率定失败: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    return data


if __name__ == '__main__':
    # 生成数据并保存
    data = generate_sample_data(n_days=365, area=150.7944)
    save_to_csv(data)
    
    # 演示所有模型
    demo_all_models()
