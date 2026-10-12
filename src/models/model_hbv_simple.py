"""Expose the existing five-parameter HBV implementation to model workflows."""
from typing import Dict, Optional

import numpy as np

from .example_model import HBVLikeModel


class HBVSimpleModel(HBVLikeModel):
    """HBV 水文模型：概念性降雨径流模型，包含土壤水分、蒸散发与快慢径流响应。"""

    @property
    def name(self) -> str:
        return 'HBV Model'

    def get_param_descriptions(self) -> Dict[str, str]:
        return {
            'fc': '田间持水量 (mm)',
            'beta': '产流形状系数',
            'k0': '快速出流比例 (每计算步)',
            'k1': '慢速出流比例 (每计算步)',
            'lp': '蒸散发限制系数',
        }

    def run(self, precip: np.ndarray, evap: np.ndarray, params: Dict[str, float],
            spatial_data: Optional[Dict] = None,
            temperature: Optional[np.ndarray] = None, warmup_steps: int = 0) -> np.ndarray:
        # Keep the full series: calibration and evaluation exclude warm-up steps.
        return super().run(precip, evap, params, spatial_data, temperature)
