"""Five-model reporting must use actual, comparable evaluation results."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
from streamlit.testing.v1 import AppTest
from src.model_comparison import normalize_comparison, summarize
from src.llm_reporter import generate_multifile_report
from src.hydro_calc import calc_nse, calc_kge


def fixture():
    rows = []
    for event, calib in [('train', True), ('valid1', False), ('valid2', False)]:
        for model, nse in [('XAJ Model', .6), ('HBV Model', .7), ('Tank Model', .5), ('BMA Ensemble', .8), ('HBV Model-XGB', .9)]:
            rows.append(dict(model_name=model, file_name=event, is_calib=calib,
                             nse=nse, kge=nse-.1, rmse=10*(1-nse), pbias=10*(1-nse)))
    return rows


class ModelComparisonTests(unittest.TestCase):
    def report(self, rows, llm=None, lang='zh'):
        with patch('src.llm_reporter.get_language', return_value=lang):
            return generate_multifile_report([{}, {}, {}], {}, {}, call_llm=llm,
                                             comparison_results=rows)

    def test_all_five_models_reach_report_and_llm_with_exact_chart_metrics(self):
        rows = fixture()
        llm = Mock(return_value='analysis')
        for lang in ('zh', 'en'):
            report = self.report(rows, llm, lang)
            prompt = llm.call_args.args[0]
            for name in ('XAJ Model', 'HBV Model', 'Tank Model', 'BMA Ensemble', 'HBV Model-XGB'):
                self.assertIn(name, report)
                self.assertIn(name, prompt)
            self.assertIn('| HBV Model-XGB |', report)
            self.assertIn('0.9000 | 0.8000 | 1.0000 | 1.0000', report)
            self.assertIn('ΔNSE=+0.2000', report)
            self.assertIn('ΔRMSE=-2.0000', report)
            self.assertIn('analysis', report)
            self.assertIn('BMA', prompt)
            self.assertIn('true error lags' if lang == 'en' else '真实误差滞后', prompt)

    def test_comparison_uses_intersection_and_equal_event_weights(self):
        rows = fixture()
        rows = [r for r in rows if not (r['model_name'].endswith('XGB') and r['file_name'] == 'valid2')]
        next(r for r in rows if r['model_name'] == 'HBV Model' and r['file_name'] == 'valid2')['nse'] = -.8
        report = self.report(rows, lang='en')
        section = report.split('## 2.')[1].split('## 3.')[0]
        self.assertIn('valid1', section)
        self.assertNotIn('valid2', section)
        self.assertIn('| HBV | 0.7000', section)
        stats = summarize([r for r in normalize_comparison(rows, {}) if r['model'] == 'HBV' and not r['is_calib']])
        self.assertAlmostEqual(stats['nse'], -.05)

    def test_missing_model_or_disjoint_validation_prevents_unified_ranking(self):
        cases = [[r for r in fixture() if not r['model_name'].endswith('XGB')],
                 [dict(r, file_name='other') if r['model_name'].endswith('XGB') else r for r in fixture()]]
        for rows in cases:
            report = self.report(rows, lang='en')
            self.assertIn('No unified ranking', report)
            self.assertNotIn('Highest mean NSE', report)
        self.assertIn('Not generated / insufficient data', self.report(cases[0], lang='en'))

    def test_no_validation_does_not_infer_generalization(self):
        rows = [r for r in fixture() if r['is_calib']]
        llm = Mock(return_value='test')
        report = self.report(rows, llm, 'en')
        self.assertIn('No validation data', report)
        self.assertNotIn('Highest mean NSE', report)
        self.assertIn('Do not infer generalization without validation', llm.call_args.args[0])

    def test_nonfinite_missing_and_sentinel_values_are_not_zero(self):
        rows = fixture()
        for r in rows:
            if r['model_name'].endswith('XGB'):
                r['kge'] = np.inf
            if r['model_name'].startswith('Tank'):
                r['nse'] = -9999
        rows[0]['rmse'] = None
        report = self.report(rows, lang='en')
        self.assertIn('N/A', report)
        self.assertIn('No unified ranking', report)
        self.assertNotIn('-9999', report)
        self.assertFalse(any(r['valid'] for r in normalize_comparison(rows, {}) if r['model'] in ('Tank', 'XGB')))

    def test_correction_can_degrade_and_deltas_use_matching_base_events(self):
        rows = fixture()
        for r in rows:
            if r['model_name'].endswith('XGB'):
                r.update(nse=.4, kge=.3, rmse=6, pbias=-6)
        report = self.report(rows, lang='en')
        self.assertIn('ΔNSE=-0.3000 (degraded)', report)
        self.assertIn('Δ|PBIAS|=+3.0000', report)

    def test_llm_error_exception_disabled_and_empty_keep_comparison(self):
        for llm in (None, Mock(return_value='[ERROR] failed'), Mock(return_value=''), Mock(side_effect=RuntimeError('private response'))):
            report = self.report(fixture(), llm, 'en')
            self.assertIn('LLM unavailable', report)
            self.assertIn('Highest mean NSE', report)
            self.assertIn('ΔNSE=+0.2000', report)
            self.assertNotIn('private response', report)

    def test_bilingual_chart_rows_and_legacy_signature(self):
        for keys, label in [(('模型', '文件', '类型'), '验证'), (('Model', 'File', 'Type'), 'Validation')]:
            row = {keys[0]: 'HBV Model-XGB', keys[1]: 'event', keys[2]: label,
                   'NSE': .8, 'KGE': .6, 'RMSE': 2., 'PBIAS': -3., 'base_model': 'HBV Model'}
            normalized = normalize_comparison([row], {})[0]
            self.assertFalse(normalized['is_calib'])
            self.assertEqual(normalized['base_model'], 'HBV Model')
            self.assertEqual(normalized['kge'], .6)
        legacy = {'HBV Model': {'event': dict(nse=.7, kge=.6, rmse=2., pbias=3., is_calib=False)}}
        with patch('src.llm_reporter.get_language', return_value='en'):
            report = generate_multifile_report([{}], {}, legacy, None, 0, [{'nse_after': .9}])
        self.assertIn('0.7000', report)
        self.assertIn('Not generated / insufficient data', report)
        self.assertNotIn('Highest mean NSE', report)

    def test_entrypoints_pass_chart_summary_and_compute_true_correction_kge(self):
        observed = np.array([1., 4., 2., 8., 3.])
        corrected = observed + np.array([1., -1., .5, 2., -1.])
        self.assertNotAlmostEqual(calc_nse(observed, corrected), calc_kge(observed, corrected))
        for filename in ('app.py', 'app_EN.py'):
            tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'generate_multifile_report']
            self.assertTrue(calls)
            for call in calls:
                kw = next(k for k in call.keywords if k.arg == 'comparison_results')
                self.assertEqual(kw.value.id, 'summary_data')
            correction_dict = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict) and
                                   any(isinstance(k, ast.Constant) and k.value == 'corrected' for k in n.keys) and
                                   any(isinstance(k, ast.Constant) and k.value == 'best_model' for k in n.keys))
            kge_expr = next(v for k, v in zip(correction_dict.keys, correction_dict.values) if isinstance(k, ast.Constant) and k.value == 'kge')
            namespace = dict(calc_kge=calc_kge, flow_plot=observed, corrected_plot=corrected, kge_after=calc_kge(observed, corrected))
            self.assertAlmostEqual(eval(compile(ast.Expression(kge_expr), filename, 'eval'), namespace), calc_kge(observed, corrected))

    def test_chinese_homepage_title(self):
        app = AppTest.from_file('app.py', default_timeout=40).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, 'HydroTune：大语言模型智能体水文平台')


if __name__ == '__main__':
    unittest.main()
