from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tokenizer_runtime import (
    ResolvedTokenCounter,
    TokenCounterInfo,
    clear_token_counter_cache,
    count_tokens,
    describe_token_counter,
    resolve_token_counter,
)

class TokenizerRuntimeTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_token_counter_cache()

    def test_gpt_models_prefer_tiktoken_backend_when_available(self) -> None:
        fake_counter = ResolvedTokenCounter(
            info=TokenCounterInfo(
                backend='tiktoken',
                source='o200k_base',
                accurate=True,
            ),
            count_text=lambda text: len(text.split()),
        )
        with patch('src.tokenizer_runtime._try_build_tiktoken_counter', return_value=fake_counter):
            with patch('src.tokenizer_runtime._try_build_transformers_counter', return_value=None):
                info = describe_token_counter('gpt-4o-mini')
                token_count = count_tokens('hello world from sigil code', 'gpt-4o-mini')

        self.assertEqual(info.backend, 'tiktoken')
        self.assertTrue(info.accurate)
        self.assertEqual(token_count, 5)

    def test_transformers_backend_can_be_selected_with_env_override(self) -> None:
        fake_counter = ResolvedTokenCounter(
            info=TokenCounterInfo(
                backend='transformers',
                source='/tmp/fake-tokenizer (local_files_only)',
                accurate=True,
            ),
            count_text=lambda text: len(text.split()),
        )
        with patch.dict(
            'os.environ',
            {'SIGIL_TOKENIZER_PATH': '/tmp/fake-tokenizer'},
            clear=False,
        ):
            with patch('src.tokenizer_runtime._try_build_transformers_counter', return_value=fake_counter):
                info = describe_token_counter('Qwen/Qwen3-Coder-30B-A3B-Instruct')
                token_count = count_tokens('one two three', 'Qwen/Qwen3-Coder-30B-A3B-Instruct')

        self.assertEqual(info.backend, 'transformers')
        self.assertTrue(info.accurate)
        self.assertEqual(token_count, 3)

    def test_fallback_backend_is_used_when_all_tokenizers_fail(self) -> None:
        with patch('src.tokenizer_runtime._try_build_tiktoken_counter', return_value=None):
            with patch('src.tokenizer_runtime._try_build_transformers_counter', return_value=None):
                counter = resolve_token_counter('unknown-model')
                token_count = count_tokens('abcd' * 5, 'unknown-model')

        self.assertEqual(counter.info.backend, 'heuristic')
        self.assertFalse(counter.info.accurate)
        self.assertGreater(token_count, 0)
