"""
数据沙盒执行器
核心功能：向LLM发送数据指纹 -> 生成清洗代码 -> exec()执行
"""
import pandas as pd
import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# 时间尺度检测
# ============================================================
def infer_timestep(dates: pd.Series) -> str:
    """
    从日期列推断时间尺度

    Args:
        dates: 日期序列 (pandas Series)

    Returns:
        'hourly' 或 'daily'
    """
    if len(dates) < 2:
        return 'daily'

    if not pd.api.types.is_datetime64_any_dtype(dates):
        try:
            dates = pd.to_datetime(dates)
        except Exception:
            return 'daily'

    diffs = dates.diff().dropna()
    if len(diffs) == 0:
        return 'daily'

    median_diff = diffs.median()
    if hasattr(median_diff, 'total_seconds'):
        median_diff_hours = median_diff.total_seconds() / 3600
    else:
        median_diff_hours = float(median_diff) / 3600 if pd.notna(median_diff) else 24.0

    if median_diff_hours <= 2:
        return 'hourly'
    else:
        return 'daily'


def get_timestep_info(timestep: str) -> dict:
    """
    获取时间尺度的相关信息

    Args:
        timestep: 'hourly' 或 'daily'

    Returns:
        包含时间步信息的字典
    """
    info = {
        'hourly': {
            'label': '小时尺度',
            'hours': 1,
            'seconds': 3600,
            'del_t': 1.0,
            'description': '1小时时间步'
        },
        'daily': {
            'label': '日尺度',
            'hours': 24,
            'seconds': 86400,
            'del_t': 24.0,
            'description': '24小时时间步'
        }
    }
    return info.get(timestep, info['daily'])


# ============================================================
# 洪水场次数据结构
# ============================================================
@dataclass
class FloodEvent:
    """洪水场次数据类"""
    name: str
    start_idx: int
    end_idx: int
    start_date: Any
    end_date: Any
    precip: np.ndarray
    evap: np.ndarray
    observed_flow: np.ndarray
    
    @property
    def duration(self) -> int:
        """场次持续时间步数"""
        return self.end_idx - self.start_idx + 1


# ============================================================
# 洪水场次自动识别
# ============================================================
def detect_flood_events(
    dates: pd.Series,
    precip: np.ndarray,
    flow: np.ndarray,
    evap: np.ndarray = None,
    **kwargs
) -> List[FloodEvent]:
    """
    基于降水和流量峰值自动识别洪水场次
    
    当前版本：同一文件的所有数据视为一场洪水
    后续版本将支持基于流量涨落特征的洪水场次识别
    
    Args:
        dates: 日期序列
        precip: 降水序列
        flow: 流量序列
        evap: 蒸发序列（可选）
        
    Returns:
        洪水场次列表（当前版本只有一场）
    """
    if evap is None:
        evap = np.zeros_like(precip)
    
    n = len(precip)
    
    if hasattr(dates, 'iloc'):
        start_date = dates.iloc[0]
        end_date = dates.iloc[-1]
    else:
        start_date = dates[0]
        end_date = dates[-1]
    
    return [FloodEvent(
        name="洪水场次1",
        start_idx=0,
        end_idx=n - 1,
        start_date=start_date,
        end_date=end_date,
        precip=precip,
        evap=evap,
        observed_flow=flow
    )]


def split_into_events(df: pd.DataFrame, event_col: str = None, n_events: int = None) -> List[pd.DataFrame]:
    """
    将数据按指定方式分割成多个洪水场次
    
    Args:
        df: 完整数据DataFrame
        event_col: 按哪一列分割（如年份列）
        n_events: 均分场次数
        
    Returns:
        分割后的DataFrame列表
    """
    if event_col and event_col in df.columns:
        return [group for _, group in df.groupby(event_col)]
    elif n_events:
        chunk_size = len(df) // n_events
        return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_events)]
    else:
        return [df]


# ============================================================
# 保底代码：当 LLM 生成的代码失败时，默默兜底
# ============================================================
def fallback_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    保底方案：基于关键词强行重命名列
    确保Demo永不崩溃
    """
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()

        # 降水相关
        if any(k in low for k in ["precip", "rain", "p", "降水", "降雨", "precipitation"]):
            if "precip" not in rename_map.values():
                rename_map[col] = "precip"

        # 蒸发相关
        elif any(k in low for k in ["evap", "et", "蒸发", "evapotranspiration"]):
            if "evap" not in rename_map.values():
                rename_map[col] = "evap"

        # 流量相关
        elif any(k in low for k in ["flow", "discharge", "q", "流量", "径流", "streamflow"]):
            if "flow" not in rename_map.values():
                rename_map[col] = "flow"

        # 时间相关
        elif any(k in low for k in ["date", "time", "datetime", "时间", "日期"]):
            if "date" not in rename_map.values():
                rename_map[col] = "date"

    df = df.rename(columns=rename_map)
    return df


# ============================================================
# 数据指纹提取
# ============================================================
def extract_fingerprint(df: pd.DataFrame) -> dict:
    """
    提取数据指纹，发送给LLM分析

    Args:
        df: 原始数据DataFrame

    Returns:
        数据指纹字典
    """
    # 处理日期类型，转换为字符串
    head_records = df.head(5).copy()
    for col in head_records.columns:
        if pd.api.types.is_datetime64_any_dtype(head_records[col]):
            head_records[col] = head_records[col].astype(str)

    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": head_records.to_dict(orient="records"),
        "shape": list(df.shape),
        "missing": df.isnull().sum().to_dict(),
    }


# ============================================================
# 沙盒执行器 Prompt 模板
# ============================================================
SANDBOX_PROMPT = """你是一个数据清洗专家。用户上传了一份水文数据，但列名不符合标准格式。

**数据指纹：**
{fingerprint}

**目标格式：**
最终 DataFrame 必须包含以下列（顺序不重要）：
- 'date' (datetime 类型)
- 'precip' (float，降水 mm)
- 'evap' (float，蒸发 mm，如果没有蒸发列，创建一个全0的列)
- 'flow' (float，流量 m³/s)

**要求：**
1. 生成一段 Python 代码，输入变量名为 `df`，输出变量名也为 `df`
2. 代码只需要做：列名重命名、类型转换、缺失值处理
3. 不要引入新库，只用 pandas 和 numpy
4. 只输出纯 Python 代码，不要任何解释，不要用 ```python ``` 包裹
5. 代码必须能直接用 `exec()` 执行
6. 如果缺少蒸发列，创建一个全0的evap列

**代码：**"""


# ============================================================
# 核心：沙盒执行器
# ============================================================
def clean_data_with_sandbox(df: pd.DataFrame, llm_caller: Callable) -> Tuple[pd.DataFrame, str]:
    """
    核心功能：向 LLM 发送数据指纹 → 生成清洗代码 → exec() 执行

    Args:
        df: 原始数据
        llm_caller: llm_api.py 中的 call_minimax 函数

    Returns:
        (清洗后的标准格式 DataFrame, 时间尺度 'hourly'/'daily')
    """
    original_df = df.copy()

    fingerprint = extract_fingerprint(original_df)

    prompt = SANDBOX_PROMPT.format(fingerprint=fingerprint)

    code = llm_caller(prompt)

    code = code.replace("```python", "").replace("```", "").strip()

    sandbox = {"df": original_df.copy(), "pd": pd, "np": np}

    try:
        exec(code, sandbox)
        cleaned_df = sandbox["df"]

        required = ["date", "precip", "flow"]
        missing = [col for col in required if col not in cleaned_df.columns]

        if missing:
            raise ValueError(f"LLM生成的代码缺少列: {missing}")

        if "evap" not in cleaned_df.columns:
            cleaned_df["evap"] = 0.0

        timestep = infer_timestep(cleaned_df['date'])
        cleaned_df.attrs['timestep'] = timestep

        return cleaned_df, timestep

    except Exception as e:
        print(f"[WARN] 沙盒执行失败，切换保底代码: {e}")
        fallback_df = fallback_rename(original_df)

        if 'evap' not in fallback_df.columns:
            fallback_df['evap'] = 0.0

        timestep = infer_timestep(fallback_df['date']) if 'date' in fallback_df.columns else 'daily'
        fallback_df.attrs['timestep'] = timestep

        return fallback_df, timestep
