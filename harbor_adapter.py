"""Harbor installed-agent adapter for Sigil Code (harbor 0.25, harborframework.com).

    harbor run --agent-import-path harbor_adapter:SigilCodeInstalledAgent -m <model> ...

Inside each task container it installs Sigil with uv, which brings its own Python 3.12 when the image
has an older one and never touches the system Python (Debian, Ubuntu and Alpine refuse a plain
`pip install` there, PEP 668). It then runs one `sigil agent` task in the task's working directory and
reports the token totals and cost Sigil prints back to Harbor.

Model settings come from the environment: OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL (pass them
with --ae, or set them on the host). Without OPENAI_MODEL, Harbor's model name (-m) is used as given.
Agent options (--ak): max_turns (default 100), allow_write / allow_shell (default true), stream,
system_prompt, append_system_prompt, permission_mode, unsafe. A version (--agent-version) is a git
ref of the repo; a bare number such as 1.1.0 means the tag v1.1.0.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, ClassVar

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, EnvVar, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REPO_URL = 'https://github.com/X-AlphaCoder-X/sigil-code.git'
LOG_FILE = 'sigilcode.txt'  # in /logs/agent, the folder Harbor collects
USER_BIN = '$HOME/.local/bin'  # where uv puts itself and `sigil`
DEFAULT_MAX_TURNS = 100  # `sigil agent` stops at 12 on its own, too few for most benchmark tasks

# curl and git for uv and the git install, CA certificates for HTTPS. Missing tools are reported, not ignored.
_SYSTEM_PACKAGES = (
    'if command -v apk >/dev/null 2>&1; then apk add --no-cache curl git ca-certificates bash; '
    'elif command -v apt-get >/dev/null 2>&1; then '
    'apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates bash; '
    'elif command -v dnf >/dev/null 2>&1; then dnf install -y curl git ca-certificates bash; '
    'elif command -v yum >/dev/null 2>&1; then yum install -y curl git ca-certificates bash; '
    'else echo "no apk, apt-get, dnf or yum: using the tools already in the image" >&2; fi; '
    'for tool in curl git; do command -v "$tool" >/dev/null 2>&1 || '
    '{ echo "Sigil Code needs $tool in the task image" >&2; exit 1; }; done'
)

_USAGE_LINE = re.compile(r'^(input_tokens_cumulative|output_tokens|total_cost_usd)=([0-9.]+)\s*$', re.MULTILINE)


def repo_ref(version: str | None) -> str:
    """The pip/uv requirement for a version: a git ref, with a bare number meaning its v-tag."""
    if not version:
        return f'git+{REPO_URL}'
    ref = f'v{version}' if re.fullmatch(r'\d+\.\d+\.\d+', version) else version
    if not re.fullmatch(r'[A-Za-z0-9._/-]+', ref):
        raise ValueError(f'Not a git ref: {version!r}')
    return f'git+{REPO_URL}@{ref}'


def parse_usage(output: str) -> dict[str, float]:
    """The totals from the last `# Usage` block `sigil agent` prints (input includes cached tokens)."""
    found: dict[str, float] = {}
    for name, value in _USAGE_LINE.findall(output):
        found[name] = float(value)  # the last block wins
    return found


class SigilCodeInstalledAgent(BaseInstalledAgent):
    """Installs Sigil Code from GitHub in the task environment and runs it once on the task."""

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag('max_turns', cli='--max-turns', type='int', default=DEFAULT_MAX_TURNS),
        CliFlag('stream', cli='--stream', type='bool'),
        CliFlag('allow_write', cli='--allow-write', type='bool', default=True),
        CliFlag('allow_shell', cli='--allow-shell', type='bool', default=True),
        CliFlag('unsafe', cli='--unsafe', type='bool'),
        CliFlag('permission_mode', cli='--permission-mode', type='enum',
                choices=['default', 'acceptEdits', 'plan', 'bypass']),
    ]

    ENV_VARS: ClassVar[list[EnvVar]] = [
        EnvVar('openai_api_key', env='OPENAI_API_KEY', type='str', env_fallback='OPENAI_API_KEY'),
        EnvVar('openai_base_url', env='OPENAI_BASE_URL', type='str', env_fallback='OPENAI_BASE_URL'),
        EnvVar('openai_model', env='OPENAI_MODEL', type='str', env_fallback='OPENAI_MODEL'),
    ]

    def __init__(self, *args: Any, system_prompt: str | None = None, append_system_prompt: str | None = None,
                 **kwargs: Any) -> None:
        # Text options are quoted here: Harbor writes a CLI_FLAGS value into the command as it is,
        # so a prompt with spaces would be split into separate arguments.
        self._system_prompt = system_prompt
        self._append_system_prompt = append_system_prompt
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return 'sigilcode'

    def get_version_command(self) -> str | None:
        return f'"{USER_BIN}/sigil" --version'

    def parse_version(self, stdout: str) -> str:
        text = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ''
        return text.removeprefix('Sigil Code').strip()  # "Sigil Code 1.1.0" -> "1.1.0"

    async def _installed_version(self, environment: BaseEnvironment) -> str | None:
        """The Sigil version already in the environment (a reused container or an image that ships it)."""
        try:
            result = await environment.exec(command=self.get_version_command())
        except Exception as exc:  # best effort: on any failure Sigil is simply installed
            self.logger.debug(f'Sigil Code version check failed, installing: {exc}')
            return None
        if result.return_code != 0 or not result.stdout:
            return None
        return self.parse_version(result.stdout) or None

    async def install(self, environment: BaseEnvironment) -> None:
        installed = await self._installed_version(environment)
        if installed and (not self._version or installed == self._version.removeprefix('v')):
            self.logger.debug(f'Sigil Code {installed} is already installed')
            return
        await self.exec_as_root(environment, command=_SYSTEM_PACKAGES, env={'DEBIAN_FRONTEND': 'noninteractive'})
        requirement = shlex.quote(repo_ref(self._version))
        await self.exec_as_agent(
            environment,
            command=(
                'set -eu; '
                # uv's own default is the first of $XDG_BIN_HOME, $XDG_DATA_HOME/../bin and ~/.local/bin
                # (docs.astral.sh/uv/reference/storage); pinned here so the run step finds `sigil`
                f'export PATH="{USER_BIN}:$PATH" UV_INSTALL_DIR="{USER_BIN}" UV_TOOL_BIN_DIR="{USER_BIN}" '
                'UV_NO_MODIFY_PATH=1; '
                'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; '
                # uv picks a Python 3.12 from the image or downloads its own, in the user's folders
                f'uv tool install --force --python 3.12 {requirement}; '
                'sigil --version'
            ),
        )

    def _command(self, instruction: str) -> str:
        flags = self.build_cli_flags()
        if self._system_prompt:
            flags += f' --system-prompt {shlex.quote(self._system_prompt)}'
        if self._append_system_prompt:
            flags += f' --append-system-prompt {shlex.quote(self._append_system_prompt)}'
        # `--` ends the options, so a task that starts with "-" is still the prompt
        return (
            f'export PATH="{USER_BIN}:$PATH"; '
            f'sigil agent --cwd "$(pwd)" {flags.strip()} -- {shlex.quote(instruction)} '
            f'2>&1 | tee /logs/agent/{LOG_FILE}'
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        env = self.resolve_env_vars()
        if self.model_name and 'OPENAI_MODEL' not in env:
            env['OPENAI_MODEL'] = self.model_name
        env['PYTHONUNBUFFERED'] = '1'  # the log fills as Sigil works, not only at the end
        await self.exec_as_agent(environment, command=self._command(instruction), env=env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        log = self.logs_dir / LOG_FILE
        try:
            usage = parse_usage(log.read_text(encoding='utf-8', errors='replace'))
        except OSError as exc:
            self.logger.warning(f'Sigil Code usage not recorded: cannot read {log}: {exc}')
            return
        if 'input_tokens_cumulative' not in usage or 'output_tokens' not in usage:
            self.logger.warning(f'Sigil Code usage not recorded: no "# Usage" block in {log}')
            return
        context.n_input_tokens = int(usage['input_tokens_cumulative'])
        context.n_output_tokens = int(usage['output_tokens'])
        cost = usage.get('total_cost_usd', 0.0)
        # 0 means Sigil had no prices for the model, not that the run was free
        context.cost_usd = cost if cost > 0 else None
