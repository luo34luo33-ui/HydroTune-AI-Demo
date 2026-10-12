"""Muskingum mode, calibration and UI regression tests."""
import ast
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src import hydro_calc as hydro
from src.app.routing import simulate_imported_model, validate_upstream_columns


class ControlledModel:
    param_bounds = {'a': (0.1, 2.0), 'b': (0.1, 2.0), 'c': (0.1, 2.0)}

    def validate_params(self, params):
        return set(params) == set(self.param_bounds)

    def run(self, precip, evap, params, spatial_data=None, temperature=None, warmup_steps=0):
        assert set(params) == set(self.param_bounds), params
        return precip * params['a'] + evap * params['b'] + params['c']


class RoutingCoreTests(unittest.TestCase):
    def setUp(self):
        self.model = ControlledModel()
        self.precip = np.array([1., 2., 4., 8., 6., 3., 2., 1.])
        self.evap = np.ones(8)
        self.upstream = np.array([2., 3., 7., 11., 8., 6., 3., 2.])
        self.model_params = dict(a=1.2, b=0.8, c=0.4)
        self.base = self.model.run(self.precip, self.evap, self.model_params)

    def calibrate(self, algorithm='two_stage', fixed=None, enabled=True, events=None, warmup=0):
        active = enabled and (events is None or any(e.get('upstream') is not None for e in events))
        routing = fixed if fixed is not None else dict(k_routing=3.2, x_routing=0.15)
        observed = self.base + hydro.muskingum_routing(self.upstream, **{
            'k': routing['k_routing'], 'x': routing['x_routing']
        }) if active else self.base
        expected_count = 5 if active and fixed is None else 3

        def optimizer(objective, bounds, max_iter, n_params, *args):
            self.assertEqual(n_params, expected_count)
            self.assertEqual(len(bounds), expected_count)
            if expected_count == 5:
                self.assertEqual(bounds[-2:], [(0.5, 5.0), (0.0, 0.5)])
            candidate = [1.2, 0.8, 0.4] + ([3.2, 0.15] if expected_count == 5 else [])
            score = objective(np.asarray(candidate))
            self.assertAlmostEqual(score, -1.0)
            return np.asarray(candidate), score

        with patch.object(hydro.ModelRegistry, 'get_model', return_value=self.model), patch.object(
            hydro, 'optimize_' + algorithm, side_effect=optimizer
        ):
            result = hydro.calibrate_model_fast(
                'controlled', self.precip, self.evap, observed,
                upstream_flow=self.upstream if events is None else events[0].get('upstream'),
                enable_routing=enabled, manual_routing_params=fixed,
                calib_events=events, algorithm=algorithm, warmup_steps=warmup,
            )
        params, score, simulated = result
        self.assertEqual({key: params[key] for key in self.model_params}, self.model_params)
        self.assertAlmostEqual(score, 1.0)
        if active:
            self.assertEqual({key: params[key] for key in routing}, routing)
        else:
            self.assertEqual(params, self.model_params)
        expected_first = events[0]['flow'] if events else observed
        np.testing.assert_allclose(simulated, expected_first)
        return params

    def test_all_optimizers_joint_fixed_and_disabled(self):
        for algorithm in ('two_stage', 'pso', 'ga', 'sce', 'de'):
            for mode in ('joint', 'fixed', 'disabled'):
                with self.subTest(algorithm=algorithm, mode=mode):
                    fixed = dict(k_routing=7.0, x_routing=0.05) if mode != 'joint' else None
                    self.calibrate(algorithm, fixed, mode != 'disabled', warmup=2)

    def test_later_event_with_upstream_activates_joint_and_fixed_routing(self):
        for fixed in (None, dict(k_routing=7., x_routing=0.05)):
            routing = fixed or dict(k_routing=3.2, x_routing=0.15)
            routed = hydro.apply_configured_routing(self.base, self.upstream, routing)
            events = [
                dict(precip=self.precip, evap=self.evap, flow=self.base, upstream=None),
                dict(precip=self.precip, evap=self.evap, flow=routed, upstream=self.upstream),
            ]
            self.calibrate(fixed=fixed, events=events)

    def test_enabled_without_upstream_does_not_add_variables(self):
        events = [dict(precip=self.precip, evap=self.evap, flow=self.base, upstream=None)]
        self.calibrate(events=events)

    def test_result_parameters_route_validation_and_correction_consistently(self):
        first = self.calibrate()
        second = self.calibrate(fixed=dict(k_routing=7., x_routing=0.05))
        for params in (first, second):
            simulated = self.model.run(self.precip, self.evap, self.model_params)
            validation = hydro.apply_configured_routing(simulated, self.upstream, params)
            correction_input = hydro.apply_configured_routing(simulated, self.upstream, params)
            expected = simulated + hydro.muskingum_routing(self.upstream, params['k_routing'], params['x_routing'])
            np.testing.assert_allclose(validation, expected)
            np.testing.assert_allclose(correction_input, expected)
        self.assertFalse(np.allclose(
            hydro.apply_configured_routing(self.base, self.upstream, first),
            hydro.apply_configured_routing(self.base, self.upstream, second),
        ))

    def test_imported_model_never_calls_optimizer_and_uses_manual_override(self):
        fixed = dict(k_routing=6.5, x_routing=0.05)
        expected = hydro.apply_configured_routing(self.base, self.upstream, fixed)
        imported = dict(self.model_params, k_routing=2.5, x_routing=0.25)
        with patch.object(hydro.ModelRegistry, 'get_model', return_value=self.model), patch.object(
            hydro, 'calibrate_model_fast', side_effect=AssertionError('Unexpected calibration')
        ):
            params, nse, simulated = simulate_imported_model(
                'controlled', self.precip, self.evap, expected, params=imported,
                upstream_flow=self.upstream, enable_routing=True, manual_routing_params=fixed,
            )
        self.assertEqual(params, dict(self.model_params, **fixed))
        self.assertAlmostEqual(nse, 1.)
        np.testing.assert_allclose(simulated, expected)
        self.assertEqual(imported['k_routing'], 2.5)

    def test_routing_requires_complete_parameters_and_aligned_series(self):
        with self.assertRaises(KeyError):
            hydro.apply_configured_routing(self.base, self.upstream, {})
        with self.assertRaises(ValueError):
            hydro.apply_configured_routing(self.base, self.upstream[:-1], self.model_params)

    def test_upstream_mapping_validation(self):
        valid = [('event.csv', pd.DataFrame({'upstream': [0., 2., 3.]}))]
        validate_upstream_columns(valid, 'upstream')
        for files, column in ((valid, ''), (valid, 'missing'),
                              ([('bad.csv', pd.DataFrame({'q': ['bad', 2]}))], 'q'),
                              ([('bad.csv', pd.DataFrame({'q': [np.nan, 2]}))], 'q')):
            with self.assertRaises(ValueError):
                validate_upstream_columns(files, column)


class RoutingUITests(unittest.TestCase):
    def test_continuous_results_plot_full_routed_series_with_warmup(self):
        import matplotlib.pyplot as plt
        from src.app.plotting import PLOT_DPI, render_figure
        from src.bma_ensemble import calc_bma_weights, apply_bma_ensemble, format_weights_string

        observed = np.arange(1., 13.)
        for filename in ('app.py', 'app_EN.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Compare)
                        and isinstance(n.test.left, ast.Name) and n.test.left.id == 'upload_mode'
                        and any(isinstance(child, ast.Constant) and child.value == 'simulated_full'
                                for statement in n.body for child in ast.walk(statement)))
            results = {name: dict(simulated=observed[:4], simulated_full=observed,
                                  nse=1., rmse=0., pbias=0.)
                       for name in ('HBV Model', 'Tank Model')}
            namespace = dict(
                plt=plt, np=np, st=Mock(), PLOT_DPI=PLOT_DPI, render_figure=render_figure,
                all_flow_arr=observed, all_precip_arr=np.ones(12), warmup_steps=2,
                calibration_results=results, model_colors={}, user_timestep='hourly',
                summary_data=[], all_results={}, preanalysis_result=None,
                calc_bma_weights=calc_bma_weights, apply_bma_ensemble=apply_bma_ensemble,
                format_weights_string=format_weights_string, calc_nse=hydro.calc_nse,
                calc_kge=hydro.calc_kge, calc_rmse=hydro.calc_rmse, calc_pbias=hydro.calc_pbias,
            )
            with patch('src.app.plotting.st.image') as display:
                exec(compile(ast.Module(body=node.body, type_ignores=[]), filename, 'exec'), namespace)
            self.assertEqual(display.call_count, 1)
            self.assertAlmostEqual(namespace['bma_nse'], 1.)
            self.assertEqual(len(namespace['summary_data']), 3)
            self.assertFalse(plt.get_fignums())

    def test_both_entrypoints_mode_switching_and_persistent_values(self):
        for filename, english in (('app.py', False), ('app_EN.py', True)):
            with self.subTest(filename=filename):
                at = AppTest.from_file(filename, default_timeout=40).run()
                self.assertFalse(at.exception)
                checkbox = next(c for c in at.checkbox if c.label in
                                ('启用上游出库汇流演算', 'Enable Upstream Routing'))
                checkbox.check().run()
                self.assertFalse(at.exception)
                choice = at.radio(key='routing_calibration_choice')
                self.assertEqual(choice.value, 'Yes' if english else '是')
                self.assertNotIn('routing_k_input', [n.key for n in at.number_input])
                choice.set_value('No' if english else '否').run()
                self.assertFalse(at.exception)
                self.assertEqual(at.number_input(key='routing_k_input').value, 2.5)
                at.number_input(key='routing_k_input').set_value(6.5).run()
                at.number_input(key='routing_x_input').set_value(0.1).run()
                at.radio(key='routing_calibration_choice').set_value('Yes' if english else '是').run()
                self.assertNotIn('routing_k_input', [n.key for n in at.number_input])
                at.radio(key='routing_calibration_choice').set_value('No' if english else '否').run()
                self.assertEqual(at.number_input(key='routing_k_input').value, 6.5)
                self.assertEqual(at.number_input(key='routing_x_input').value, 0.1)
                mode = next(r for r in at.radio if r.label in ('参数模式', 'Parameter Mode'))
                mode.set_value('Import Parameters' if english else '导入参数').run()
                self.assertNotIn('routing_calibration_choice', [r.key for r in at.radio])
                self.assertEqual(at.number_input(key='routing_k_input').value, 6.5)
                self.assertFalse(at.exception)

    def test_imported_values_prefill_once_and_allow_edits(self):
        script = '''import streamlit as st
from src.app.routing import render_routing_settings
st.session_state['imported_k_routing'] = st.session_state.get('test_import_k', 4.0)
st.session_state['imported_x_routing'] = 0.2
render_routing_settings(import_mode=True)
'''
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fixture = Path(tmp) / 'settings.py'
            fixture.write_text(script, encoding='utf-8')
            at = AppTest.from_file(str(fixture), default_timeout=40).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.number_input(key='routing_k_input').value, 4.0)
            at.number_input(key='routing_k_input').set_value(7.0).run()
            self.assertEqual(at.number_input(key='routing_k_input').value, 7.0)
            at.session_state['test_import_k'] = 5.0
            at.run()
            self.assertEqual(at.number_input(key='routing_k_input').value, 5.0)
            self.assertFalse(at.exception)

    def test_downstream_calls_use_model_params_in_both_entrypoints(self):
        for filename in ('app.py', 'app_EN.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'apply_configured_routing']
            self.assertTrue(calls)
            base = np.arange(8, dtype=float)
            upstream = np.array([1., 2., 6., 9., 5., 3., 2., 1.])
            for params in (dict(k_routing=3.2, x_routing=0.15), dict(k_routing=7., x_routing=0.05)):
                namespace = dict(simulated=base, full_simulated=base, sim=base,
                                 upstream_arr=upstream, all_upstream_arr=upstream,
                                 f={'upstream': upstream}, sf={'upstream': upstream},
                                 safe_params=params, params=params,
                                 apply_configured_routing=hydro.apply_configured_routing)
                expected = base + hydro.muskingum_routing(upstream, params['k_routing'], params['x_routing'])
                for call in calls:
                    actual = eval(compile(ast.Expression(body=call), filename, 'eval'), namespace)
                    np.testing.assert_allclose(actual, expected)

    def test_invalid_mapping_stops_entrypoints_before_api_or_calibration(self):
        for filename in ('app.py', 'app_EN.py'):
            upload = BytesIO(b'date,precip,evap,flow,upstream\n2020-01-01,1,0,2,3\n2020-01-02,2,0,3,4\n')
            upload.name = 'event.csv'

            def file_uploader(*args, **kwargs):
                if kwargs.get('accept_multiple_files'):
                    upload.seek(0)
                    return [upload]
                return None

            with patch('streamlit.file_uploader', side_effect=file_uploader), patch(
                'src.llm_api.call_minimax', side_effect=AssertionError('Unexpected API call')
            ), patch.object(hydro, 'calibrate_model_fast', side_effect=AssertionError('Unexpected calibration')):
                at = AppTest.from_file(filename, default_timeout=40).run()
                checkbox = next(c for c in at.checkbox if c.label in
                                ('启用上游出库汇流演算', 'Enable Upstream Routing'))
                checkbox.check().run()
                button = next(b for b in at.button if b.label in ('🚀 开始分析', '🚀 Start Analysis'))
                button.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any('上游出库列名' in e.value or 'upstream discharge column' in e.value for e in at.error))


if __name__ == '__main__':
    unittest.main()
