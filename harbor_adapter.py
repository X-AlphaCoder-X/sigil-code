from __future__ import annotations

import os
import shlex

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, EnvVar, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SigilCodeInstalledAgent(BaseInstalledAgent):
    """
    Harbor installed-agent adapter for Sigil Code.

    This installs the published GitHub repo inside the Harbor environment and
    runs the CLI in one-shot mode against the task working directory.
    """

    CLI_FLAGS = [
        CliFlag("max_turns", cli="--max-turns", type="int"),
        CliFlag("append_system_prompt", cli="--append-system-prompt", type="str"),
        CliFlag("system_prompt", cli="--system-prompt", type="str"),
        CliFlag("stream", cli="--stream", type="bool"),
        CliFlag("allow_write", cli="--allow-write", type="bool", default=True),
        CliFlag("allow_shell", cli="--allow-shell", type="bool", default=True),
    ]

    ENV_VARS = [
        EnvVar("openai_api_key", env="OPENAI_API_KEY", type="str", env_fallback="OPENAI_API_KEY"),
        EnvVar("openai_base_url", env="OPENAI_BASE_URL", type="str", env_fallback="OPENAI_BASE_URL"),
        EnvVar("openai_model", env="OPENAI_MODEL", type="str", env_fallback="OPENAI_MODEL"),
    ]

    @staticmethod
    def name() -> str:
        return "sigilcode"

    def get_version_command(self) -> str | None:
        return "sigil --help >/dev/null 2>&1 && echo installed"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache python3 py3-pip bash curl git; "
                "elif command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && apt-get install -y python3 python3-pip bash curl git; "
                "elif command -v yum >/dev/null 2>&1; then "
                "yum install -y python3 python3-pip bash curl git; "
                "fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        version_spec = f"@{self._version}" if self._version else ""
        repo_spec = (
            f"git+https://github.com/X-AlphaCoder-X/sigil-code.git{version_spec}"
            if version_spec
            else "git+https://github.com/X-AlphaCoder-X/sigil-code.git"
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"python3 -m pip install --upgrade pip && python3 -m pip install {shlex.quote(repo_spec)} && "
                "sigil --help >/dev/null"
            ),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        del context

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        env = self.resolve_env_vars()
        if self.model_name and "OPENAI_MODEL" not in env:
            env["OPENAI_MODEL"] = self.model_name

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        escaped_instruction = shlex.quote(instruction)

        # Harbor executes in the task working directory; keep cwd anchored there.
        command = (
            f"sigil agent {escaped_instruction} "
            f"--cwd $(pwd) {extra_flags}"
            "2>&1 | stdbuf -oL tee /logs/agent/sigilcode.txt"
        )
        await self.exec_as_agent(environment, command=command, env=env)
