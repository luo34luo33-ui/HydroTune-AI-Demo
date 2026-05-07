# -*- coding: utf-8 -*-
"""
穷举式日期解析模块
HydroTune-AI - 智能水文模型率定系统

功能：
1. 穷举所有常见日期时间格式
2. 自动识别DataFrame中的日期列
3. 推断时间尺度（小时/日）
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional


DATE_FORMATS = [
    # 年/月/日 时:分:秒 (最常见)
    '%Y/%m/%d %H:%M:%S',
    '%Y/%m/%d %H:%M',
    
    # 年-月-日 时:分:秒
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    
    # 年月日 时:分:秒 (无分隔)
    '%Y%m%d %H:%M:%S',
    '%Y%m%d%H%M%S',
    '%Y%m%d %H:%M',
    '%Y%m%d',
    
    # 年.月.日 时:分:秒
    '%Y.%m.%d %H:%M:%S',
    '%Y.%m.%d %H:%M',
    '%Y.%m.%d',
    
    # 日/月/年 时:分:秒 (欧式)
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%d/%m/%Y',
    
    # 月/日/年 时:分:秒 (美式)
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M',
    '%m/%d/%Y',
    
    # 纯日期 (各种分隔)
    '%Y/%m/%d',
    '%Y-%m-%d',
    '%Y.%m.%d',
    '%d-%m-%Y',
    '%m-%d-%Y',
    
    # ISO8601 格式
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M',
    'ISO8601',
    
    # 带上午/下午
    '%Y/%m/%d %I:%M:%S %p',
    '%Y/%m/%d %I:%M %p',
    '%Y-%m-%d %I:%M:%S %p',
    '%Y-%m-%d %I:%M %p',
    
    # 中文格式
    '%Y年%m月%d日 %H:%M:%S',
    '%Y年%m月%d日 %H:%M',
    '%Y年%m月%d日',
    
    # Unix时间戳 (数字)
    'unix_s',
    'unix_ms',
]


def parse_dates(dates_input) -> pd.Series:
    """穷举式日期解析
    
    Args:
        dates_input: pd.Series, list, np.ndarray, DataFrame, 或字符串
        
    Returns:
        pd.Series: 日期序列
    """
    if dates_input is None:
        return _default_dates(100)
    
    n = _get_length(dates_input)
    if n == 0:
        return _default_dates(100)
    
    # 处理DataFrame - 自动查找日期列
    if hasattr(dates_input, 'columns'):
        dates_input = _find_date_column(dates_input)
    
    # 已经是datetime64
    if isinstance(dates_input, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(dates_input):
            return dates_input.reset_index(drop=True)
        dates_array = dates_input.values
    elif isinstance(dates_input, pd.Index):
        if pd.api.types.is_datetime64_any_dtype(dates_input):
            return pd.Series(dates_input).reset_index(drop=True)
        dates_array = np.array(dates_input)
    else:
        dates_array = np.array(dates_input)
    
    # 方法1: infer_datetime_format
    try:
        parsed = pd.to_datetime(dates_array, infer_datetime_format=True, errors='coerce')
        valid_ratio = _valid_ratio(parsed, n)
        if valid_ratio > 0.5:
            return _clean_result(parsed)
    except:
        pass
    
    # 方法2: 逐个格式尝试
    for fmt in DATE_FORMATS:
        if fmt == 'ISO8601':
            try:
                parsed = pd.to_datetime(dates_array, format='ISO8601', errors='coerce')
                valid_ratio = _valid_ratio(parsed, n)
                if valid_ratio > 0.5:
                    return _clean_result(parsed)
            except:
                continue
        elif fmt == 'unix_s':
            try:
                parsed = pd.to_datetime(dates_array.astype(float), unit='s', errors='coerce')
                valid_ratio = _valid_ratio(parsed, n)
                if valid_ratio > 0.5:
                    return _clean_result(parsed)
            except:
                continue
        elif fmt == 'unix_ms':
            try:
                parsed = pd.to_datetime(dates_array.astype(float), unit='ms', errors='coerce')
                valid_ratio = _valid_ratio(parsed, n)
                if valid_ratio > 0.5:
                    return _clean_result(parsed)
            except:
                continue
        else:
            try:
                parsed = pd.to_datetime(dates_array, format=fmt, errors='coerce')
                valid_ratio = _valid_ratio(parsed, n)
                if valid_ratio > 0.5:
                    return _clean_result(parsed)
            except:
                continue
    
    # 方法3: 无格式自动推断
    try:
        parsed = pd.to_datetime(dates_array, errors='coerce')
        valid_ratio = _valid_ratio(parsed, n)
        if valid_ratio > 0.5:
            return _clean_result(parsed)
    except:
        pass
    
    # 全部失败，返回默认日期序列
    return _default_dates(n)


def infer_timestep(dates: pd.Series) -> str:
    """从日期序列推断时间尺度
    
    Args:
        dates: 日期序列
        
    Returns:
        str: 'hourly', 'daily', 或 'multi-daily'
    """
    if len(dates) < 2:
        return 'daily'
    
    try:
        diffs = dates.diff().dropna()
        if len(diffs) == 0:
            return 'daily'
        
        # 转换为小时
        hours_list = []
        for d in diffs:
            try:
                hours = d.total_seconds() / 3600
                hours_list.append(hours)
            except:
                hours_list.append(24)
        
        if not hours_list:
            return 'daily'
        
        median_hours = np.median(hours_list)
        
        if median_hours <= 1:
            return 'hourly'
        elif median_hours <= 6:
            return 'hourly'
        elif median_hours <= 24:
            return 'daily'
        else:
            return 'multi-daily'
    except:
        return 'daily'


def _find_date_column(df: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
    """从DataFrame中查找日期列"""
    date_keywords = [
        'date', 'datetime', 'time', '日期', '时间',
        'timestamp', 'dt', 'dato', 'fecha'
    ]
    
    for col in df.columns:
        col_lower = str(col).lower()
        for kw in date_keywords:
            if kw in col_lower:
                return df[col]
    
    # 如果找不到，返回第一列
    return df.iloc[:, 0]


def _get_length(dates_input) -> int:
    """获取输入数据的长度"""
    if hasattr(dates_input, '__len__'):
        return len(dates_input)
    return 0


def _valid_ratio(parsed: pd.Series, n: int) -> float:
    """计算有效解析比例"""
    if n == 0:
        return 0
    try:
        return parsed.notna().sum() / n
    except:
        return 0


def _clean_result(parsed) -> pd.Series:
    """清理解析结果"""
    try:
        return parsed.reset_index(drop=True)
    except:
        return pd.Series(parsed)


def _default_dates(n: int) -> pd.Series:
    """生成默认日期序列"""
    return pd.Series(pd.date_range(start='2020-01-01', periods=n, freq='D'))


def parse_dates_with_fallback(
    dates_input,
    fallback_start: str = '2020-01-01'
) -> pd.Series:
    """带回退的日期解析
    
    Args:
        dates_input: 输入数据
        fallback_start: 回退起始日期
        
    Returns:
        pd.Series: 日期序列
    """
    result = parse_dates(dates_input)
    
    # 检查是否大部分为NaT
    n = len(result)
    na_ratio = result.isna().sum() / n if n > 0 else 1
    
    if na_ratio > 0.5:
        # 回退到默认日期序列
        return _default_dates(n)
    
    return result