"""Regression checks for session credentials, dispatch and safe diagnostics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import requests
from streamlit.testing.v1 import AppTest
from streamlit.proto.TextInput_pb2 import TextInput
from src import llm_api as api


def response(status=200, data=None):
    return Mock(status_code=status, json=Mock(return_value=data if data is not None else {
        "choices": [{"message": {"content": "OK"}}], "base_resp": {"status_code": 0},
    }))


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.state = {}
        self.secrets = {"MINIMAX_API_KEY": "server-mini", "DEEPSEEK_API_KEY": "server-deep"}
        self.session = patch.object(api, "_session_state", return_value=self.state)
        self.secret = patch.object(api, "_get_secret", side_effect=lambda k, default="": self.secrets.get(k, default))
        self.session.start()
        self.secret.start()
        api.set_language('zh')

    def tearDown(self):
        self.session.stop()
        self.secret.stop()
        api.set_language('zh')

    def test_minimax_session_override_beats_deepseek_and_uses_modern_request(self):
        self.state['llm_settings'] = dict(provider='minimax', api_key='visitor-key', region='intl', model='MiniMax-M2.7')
        with patch.object(api.requests, 'post', return_value=response()) as post:
            self.assertEqual(api.call_minimax('rain', 'system'), 'OK')
        args, kw = post.call_args
        self.assertEqual(args[0], api.MINIMAX_ENDPOINTS['intl'])
        self.assertEqual(kw['headers']['Authorization'], 'Bearer visitor-key')
        self.assertEqual(kw['json']['messages'], [dict(role='system', content='system'), dict(role='user', content='rain')])
        self.assertTrue(kw['json']['reasoning_split'])
        self.assertNotIn('tokens_to_generate', kw['json'])
        self.assertEqual(self.secrets['MINIMAX_API_KEY'], 'server-mini')

    def test_auto_preserves_deepseek_and_blank_override_uses_server_minimax(self):
        with patch.object(api.requests, 'post', return_value=response()) as post:
            api.call_minimax('test')
            self.assertEqual(post.call_args.args[0], api.DEEPSEEK_API_URL)
            self.state['llm_settings'] = dict(provider='minimax', api_key='', region='cn')
            api.call_minimax('test')
            self.assertEqual(post.call_args.args[0], api.MINIMAX_ENDPOINTS['cn'])
            self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer server-mini')

    def test_sessions_do_not_share_keys(self):
        self.state['llm_settings'] = dict(provider='minimax', api_key='visitor-one')
        self.assertEqual(api.get_llm_config()['api_key'], 'visitor-one')
        with patch.object(api, '_session_state', return_value={}):
            self.assertEqual(api.get_llm_config()['api_key'], 'server-mini')

    def test_failures_are_actionable_without_echoing_secrets(self):
        self.state['llm_settings'] = dict(provider='minimax', api_key='visitor-key')
        cases = [response(401), response(429), response(data={'base_resp': {
            'status_code': 2049, 'status_msg': 'visitor-key secret server response'}}),
            response(data={'choices': []}), response(data={'error': 'visitor-key'})]
        for resp in cases:
            with patch.object(api.requests, 'post', return_value=resp):
                result = api.call_minimax('test')
                self.assertTrue(result.startswith('[ERROR]'))
                self.assertNotIn('visitor-key', result)
                self.assertEqual(self.state['llm_last_error'], result)
        for error in [requests.Timeout('visitor-key'), requests.ConnectionError('visitor-key')]:
            with patch.object(api.requests, 'post', side_effect=error):
                self.assertNotIn('visitor-key', api.call_minimax('test'))
        with patch.object(api.requests, 'post', return_value=response()):
            self.assertEqual(api.call_minimax('test'), 'OK')
            self.assertNotIn('llm_last_error', self.state)

    def test_missing_credentials_invalid_region_and_truncated_answer(self):
        self.secrets.clear()
        with patch.object(api.requests, 'post') as post:
            self.assertIn('MINIMAX_API_KEY', api.call_minimax('test'))
            post.assert_not_called()
        self.state['llm_settings'] = dict(provider='minimax', api_key='visitor-key', region='invalid')
        self.assertIn('MINIMAX_REGION', api.call_minimax('test'))
        self.state['llm_settings']['region'] = 'cn'
        with patch.object(api.requests, 'post', return_value=response(data={'choices': [
            {'message': {'content': ''}, 'finish_reason': 'length'}]})):
            self.assertIn('长度上限', api.call_minimax('test'))


class LLMSettingsUITests(unittest.TestCase):
    def test_entrypoints_start_with_collapsed_password_settings(self):
        for name in ('app.py', 'app_EN.py'):
            at = AppTest.from_file(name, default_timeout=40).run()
            self.assertFalse(at.exception)
            expander = next(e for e in at.expander if 'MiniMax API' in e.label)
            self.assertFalse(expander.proto.expanded)
            self.assertEqual(at.text_input(key='minimax_key_input').value, '')
            self.assertEqual(at.text_input(key='minimax_key_input').proto.type, TextInput.PASSWORD)

    def test_apply_test_clear_and_session_isolation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            fixture = Path(tmp) / 'settings_app.py'
            fixture.write_text('from src.app.llm_settings import render_llm_settings\nrender_llm_settings()\n', encoding='utf-8')
            with patch.object(api, '_get_secret', side_effect=lambda k, default='': default), patch.object(
                api.requests, 'post', return_value=response()) as post:
                at = AppTest.from_file(str(fixture), default_timeout=40).run()
                at.text_input(key='minimax_key_input').set_value('visitor-key')
                at.selectbox(key='minimax_region_input').set_value('intl')
                next(b for b in at.button if b.label == '应用并测试连接').click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state['llm_settings']['provider'], 'minimax')
                self.assertEqual(post.call_args.args[0], api.MINIMAX_ENDPOINTS['intl'])
                self.assertTrue(any('连接成功' in s.value for s in at.success))
                second = AppTest.from_file(str(fixture), default_timeout=40).run()
                self.assertEqual(second.text_input(key='minimax_key_input').value, '')
                at.button(key='clear_llm_settings').click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.text_input(key='minimax_key_input').value, '')
                self.assertNotIn('llm_settings', at.session_state)


if __name__ == '__main__':
    unittest.main()
