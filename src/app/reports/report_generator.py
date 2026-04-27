# -*- coding: utf-8 -*-
"""
报告生成模块
从 app.py 报告生成部分提取
"""
import streamlit as st
from typing import Dict, Any, List


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_calibration_report(
        calibration_results: Dict[str, Dict],
        catchment_area: float = 150.7944,
    ) -> str:
        """生率定报告"""
        from src.llm_reporter import generate_calibration_report
        
        return generate_calibration_report(
            calibration_results=calibration_results,
            all_results={},
            catchment_area=catchment_area,
        )
    
    @staticmethod
    def generate_comprehensive_report(
        calibration_results: Dict[str, Dict],
        catchment_area: float = 150.7944,
    ) -> str:
        """生成综合报告"""
        from src.llm_reporter import generate_calibration_report
        
        return generate_calibration_report(
            calibration_results=calibration_results,
            all_results={},
            catchment_area=catchment_area,
        )
    
    @staticmethod
    def generate_multifile_report(
        file_simulation_results: Dict[str, Dict],
        catchment_area: float = 150.7944,
    ) -> str:
        """生成多文件报告"""
        from src.llm_reporter import generate_calibration_report
        
        return generate_calibration_report(
            calibration_results=file_simulation_results,
            all_results={},
            catchment_area=catchment_area,
        )
    
    @staticmethod
    def download_report(report_content: str, filename: str = "report.md"):
        """提供报告下载"""
        st.download_button(
            label="📥 下载报告",
            data=report_content,
            file_name=filename,
            mime="text/markdown",
        )