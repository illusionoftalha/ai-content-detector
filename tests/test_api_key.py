import os
import unittest
from unittest.mock import patch

import ai_detector


class ApiKeyTests(unittest.TestCase):
    def check_key(self, configured, environment, expected):
        with patch.object(ai_detector, 'GROQ_API_KEY', configured), \
                patch.dict(os.environ, {'GROQ_API_KEY': environment}), \
                patch.object(ai_detector, 'Groq') as groq:
            groq.return_value.chat.completions.create.return_value.choices[0].message.content = '{}'
            self.assertEqual(ai_detector.analyze('example text'), {})
            groq.assert_called_once_with(api_key=expected)

    def test_default_empty_key_uses_environment(self):
        self.check_key('', 'environment-key', 'environment-key')

    def test_placeholder_uses_environment(self):
        self.check_key('YOUR_GROQ_API_KEY_HERE', 'environment-key', 'environment-key')

    def test_configured_key_has_precedence(self):
        self.check_key('configured-key', 'environment-key', 'configured-key')

    def test_missing_key_fails_before_creating_client(self):
        with patch.object(ai_detector, 'GROQ_API_KEY', ''), \
                patch.dict(os.environ, {'GROQ_API_KEY': ''}), \
                patch.object(ai_detector, 'Groq') as groq:
            with self.assertRaisesRegex(SystemExit, 'No API key found'):
                ai_detector.analyze('example text')
            groq.assert_not_called()


if __name__ == '__main__':
    unittest.main()
