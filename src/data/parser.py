"""
数据解析器
支持CSV、Excel等多种格式，自动识别列名
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class DataParser:
    """
    智能数据解析器
    自动识别时间列、降水、蒸发、流量等变量
    """

    # 标准列名映射（支持中英文）
    COLUMN_PATTERNS = {
        "date": ["date", "time", "datetime", "时间", "日期", "Date", "Time"],
        "precip": [
            "precip",
            "rainfall",
            "p",
            "降水",
            "降雨",
            "Precip",
            "P",
            "Precipitation",
            "Rainfall",
        ],
        "evap": [
            "evap",
            "et",
            "evapotranspiration",
            "蒸发",
            "Evap",
            "ET",
            "Evapotranspiration",
        ],
        "flow": [
            "flow",
            "discharge",
            "q",
            "流量",
            "径流",
            "Flow",
            "Q",
            "Discharge",
            "Streamflow",
        ],
    }

    def parse(self, file_path: str) -> pd.DataFrame:
        """
        解析文件为标准格式DataFrame

        Args:
            file_path: 文件路径

        Returns:
            标准格式DataFrame，包含 ['date', 'precip', 'evap', 'flow'] 列
        """
        path = Path(file_path)

        if path.suffix.lower() == ".csv":
            df = self._parse_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = self._parse_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        return self._standardize_columns(df)

    def parse_from_buffer(self, file_buffer, file_name: str) -> pd.DataFrame:
        """
        从文件缓冲区解析（用于Streamlit上传）

        Args:
            file_buffer: 文件缓冲区
            file_name: 文件名

        Returns:
            标准格式DataFrame
        """
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_buffer)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_buffer)
        else:
            raise ValueError(f"不支持的文件格式")

        return self._standardize_columns(df)

    def _parse_csv(self, path: Path) -> pd.DataFrame:
        """解析CSV文件"""
        # 尝试不同编码
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]

        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法读取CSV文件，请检查编码")

    def _parse_excel(self, path: Path) -> pd.DataFrame:
        """解析Excel文件"""
        xls = pd.ExcelFile(path)

        # 智能选择Sheet
        sheet = self._detect_data_sheet(xls.sheet_names)
        df = pd.read_excel(path, sheet_name=sheet)

        return df

    def _detect_data_sheet(self, sheet_names: List[str]) -> str:
        """智能识别数据Sheet"""
        # 优先选择包含数据关键词的Sheet
        data_keywords = ["data", "数据", "观测", "obs", "hydro", "水文"]

        for name in sheet_names:
            if any(kw in name.lower() for kw in data_keywords):
                return name

        # 默认选择第一个Sheet
        return sheet_names[0]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将列名映射为标准名称
        自动识别中英文列名
        """
        rename_map = {}

        for std_name, patterns in self.COLUMN_PATTERNS.items():
            for col in df.columns:
                col_str = str(col).strip()
                if col_str in patterns or col_str.lower() in [
                    p.lower() for p in patterns
                ]:
                    if std_name not in rename_map.values():
                        rename_map[col] = std_name
                        break

        df = df.rename(columns=rename_map)

        # 处理日期列
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 确保数值列为float
        for col in ["precip", "evap", "flow"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_fingerprint(self, df: pd.DataFrame) -> Dict:
        """
        提取数据指纹，用于发送给LLM分析

        Returns:
            数据指纹字典
        """
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "shape": list(df.shape),
            "missing": df.isnull().sum().to_dict(),
        }

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        验证数据是否符合标准格式

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        required_cols = ["precip", "flow"]

        # 检查必要列
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"缺少必要列: {col}")

        # 检查数据量
        if len(df) < 10:
            errors.append(f"数据量太少: {len(df)} 行")

        # 检查缺失值
        for col in ["precip", "flow"]:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.5:
                    errors.append(f"列 {col} 缺失值过多: {missing_pct:.1%}")

        return len(errors) == 0, errors
