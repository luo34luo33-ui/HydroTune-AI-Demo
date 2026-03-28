"""
水文模型智能Agent工作流
自动处理数据 → 调用模型 → 率定 → 比较 → 生成报告
"""
from typing import Dict, List, Callable
import pandas as pd
import numpy as np

from ..models.registry import ModelRegistry
from ..hydro_calc import compare_all_models, calc_nse


class HydroAgent:
    """
    水文模型智能Agent
    自动处理数据 → 调用模型 → 率定 → 比较
    """

    def __init__(self):
        self.models = ModelRegistry.list_models()
        self.results = []

    def run_full_workflow(
        self, data: pd.DataFrame, status_callback: Callable = None, max_iter: int = 30
    ) -> Dict:
        """
        执行完整的分析工作流

        Args:
            data: 标准格式数据 ['date', 'precip', 'evap', 'flow']
            status_callback: 状态更新回调函数
            max_iter: 率定最大迭代次数

        Returns:
            包含所有模型结果的字典
        """
        # 提取数据
        precip = data["precip"].values
        evap = data["evap"].values
        observed = data["flow"].values

        if status_callback:
            status_callback(f"正在率定 {len(self.models)} 个模型...")

        # 执行多模型比较
        results = compare_all_models(precip, evap, observed, max_iter=max_iter)

        self.results = results

        return {
            "results": results,
            "best_model": results[0]["model_name"] if results else None,
            "best_nse": results[0]["nse"] if results else None,
            "data_summary": {
                "length": len(data),
                "precip_mean": float(np.nanmean(precip)),
                "flow_mean": float(np.nanmean(observed)),
                "date_range": f"{data['date'].min()} ~ {data['date'].max()}"
                if "date" in data.columns
                else "N/A",
            },
        }

    def get_results_for_report(self) -> str:
        """获取格式化的结果字符串，用于生成报告"""
        if not self.results:
            return "无结果"

        lines = []
        for i, r in enumerate(self.results):
            lines.append(f"**第{i+1}名: {r['model_name']}**")
            lines.append(f"  - NSE: {r['nse']:.4f}")
            lines.append(f"  - RMSE: {r['rmse']:.4f}")
            lines.append(f"  - MAE: {r['mae']:.4f}")
            lines.append(f"  - PBIAS: {r['pbias']:.2f}%")
            lines.append(f"  - 最优参数: {r['params']}")
            lines.append("")

        return "\n".join(lines)
