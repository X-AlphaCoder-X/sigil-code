"""skill_get must load skills from the user folder (~/.sigil/skills), like the prompt list."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig


class UserSkillLoadTests(unittest.TestCase):
    def test_user_skill_is_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
            skill = Path(home) / '.sigil' / 'skills' / 'demo-user-skill'
            skill.mkdir(parents=True)
            (skill / 'SKILL.md').write_text(
                '---\nname: demo-user-skill\ndescription: test skill\n---\n\nBody text.\n',
                encoding='utf-8',
            )
            env = {'SIGIL_DISABLE_USER_SKILLS': ''}
            with mock.patch('src.skills_runtime._home', return_value=Path(home)), \
                    mock.patch.dict(os.environ, env):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='x'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(cwd)),
                )
                self.assertIsNotNone(agent.skills_runtime.get_skill('demo-user-skill'))
                agent.reload_skills()
                self.assertIsNotNone(agent.skills_runtime.get_skill('demo-user-skill'))


if __name__ == '__main__':
    unittest.main()
