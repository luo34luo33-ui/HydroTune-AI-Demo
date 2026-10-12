"""Regression checks for bounded rendering and complete event summaries."""
import ast
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.app.plotting import (
    EVENTS_PER_FIGURE, MAX_IMAGE_PIXELS, create_event_figure,
    figure_png, render_figure,
)
from src.bma_ensemble import calc_bma_weights, apply_bma_ensemble, format_weights_string
from src.hydro_calc import calc_nse, calc_kge, calc_rmse, calc_pbias


class CloudPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_oversized_figure_is_capped_and_closed(self):
        fig, ax = plt.subplots(figsize=(18, 58))
        ax.plot([0, 1], [0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(figure_png(fig))) as image:
                self.assertLessEqual(image.width * image.height, MAX_IMAGE_PIXELS)
        self.assertFalse(plt.fignum_exists(fig.number))

    def test_failed_encoding_still_closes_figure(self):
        fig, _ = plt.subplots()
        with patch.object(fig, "savefig", side_effect=RuntimeError("encoding failed")):
            with self.assertRaises(RuntimeError):
                figure_png(fig)
        self.assertFalse(plt.fignum_exists(fig.number))

    def test_display_and_download_reuse_the_same_png(self):
        fig, _ = plt.subplots()
        with patch("src.app.plotting.st.image") as display, patch(
            "src.app.plotting.st.download_button"
        ) as download:
            render_figure(fig, label="Download", file_name="chart.png", key="chart")
        self.assertIs(display.call_args.args[0], download.call_args.kwargs["data"])
        self.assertEqual(download.call_args.kwargs["on_click"], "ignore")
        self.assertFalse(plt.get_fignums())

    def test_entrypoint_event_loops_keep_all_results_across_batches(self):
        # Exercise the actual result loops without running calibration or calling an API.
        for entrypoint in ("app.py", "app_EN.py"):
            for count in (1, 9):
                with self.subTest(entrypoint=entrypoint, events=count):
                    tree = ast.parse(Path(entrypoint).read_text(encoding="utf-8"))
                    loop = next(
                        node for node in ast.walk(tree)
                        if isinstance(node, ast.For)
                        and any(isinstance(child, ast.Name) and child.id == "create_event_figure"
                                for child in ast.walk(node))
                    )
                    observed = np.linspace(1, 10, 12)
                    files = [(f"event_{i}.csv", pd.DataFrame({"flow": observed})) for i in range(count)]
                    results = {
                        model: {name: dict(observed=observed, precip=np.ones(12),
                                          simulated=observed * 0.95, is_calib=True,
                                          nse=0.98, rmse=0.3, pbias=-5)
                                for name, _ in files}
                        for model in ("Tank Model", "HBV Model")
                    }
                    namespace = dict(
                        np=np, pd=pd, plt=plt, EVENTS_PER_FIGURE=EVENTS_PER_FIGURE,
                        create_event_figure=create_event_figure, render_figure=render_figure,
                        file_dfs=files, n_files=count, file_simulation_results=results,
                        calibration_results={}, xgb_correction_data={}, warmup_steps=0,
                        model_colors={}, user_timestep="daily", summary_data=[], all_results={},
                        calc_bma_weights=calc_bma_weights, apply_bma_ensemble=apply_bma_ensemble,
                        format_weights_string=format_weights_string,
                        calc_nse=calc_nse, calc_kge=calc_kge, calc_rmse=calc_rmse, calc_pbias=calc_pbias,
                    )
                    with warnings.catch_warnings(), patch("src.app.plotting.st.image") as display, patch(
                        "src.app.plotting.st.download_button"
                    ) as download:
                        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                        exec(compile(ast.Module(body=[loop], type_ignores=[]), entrypoint, "exec"), namespace)
                    self.assertEqual(len(namespace["summary_data"]), count * 3)
                    self.assertEqual(len(namespace["all_results"]), count)
                    self.assertEqual(display.call_count, (count + 3) // 4)
                    self.assertEqual(download.call_count, display.call_count)
                    self.assertEqual(len({c.kwargs["key"] for c in download.call_args_list}), display.call_count)
                    self.assertFalse(plt.get_fignums())
                    for call in display.call_args_list:
                        with Image.open(BytesIO(call.args[0])) as image:
                            self.assertLessEqual(image.width * image.height, MAX_IMAGE_PIXELS)


if __name__ == "__main__":
    unittest.main()
