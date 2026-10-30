"""Named provider profiles: base URL, model and key, switched with --profile or /profile.

In settings.json (any level) or ~/.sigil/profiles.json:
    "profiles": {
        "local": {"base_url": "http://127.0.0.1:8000/v1", "model": "qwen3-coder"},
        "cloud": {"base_url": "https://openrouter.ai/api/v1", "model": "x/y", "api_key_env": "OPENROUTER_KEY"}
    }
The key is read from the environment variable named by api_key_env (or apiKeyEnv);
a literal api_key works too but is better kept out of files.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


_FROM_PROJECT = '_from_project'  # internal marker: the profile came from a repository settings file


def load_profiles(cwd: Path) -> dict[str, dict[str, Any]]:
    """Profiles by name. The user's own (~/.sigil/settings.json, --settings, ~/.sigil/profiles.json)
    win over a project's `.sigil/settings.json` profile of the same name. A project profile may name
    any ordinary environment variable in api_key_env (e.g. MY_CLOUD_KEY), but not one of the user's
    MAIN model keys (OPENAI_API_KEY, SIGIL_API_KEY, ANTHROPIC_*, or another host's SIGIL_API_KEY_<HOST>):
    those are dropped so a repository cannot send the main key to a host it chose (see profile_config)."""
    from .settings import _home, extra_settings, load_settings, merge_settings, read_settings_file

    user: dict[str, Any] = {}
    if not os.environ.get('SIGIL_DISABLE_USER_SETTINGS'):
        user = read_settings_file(_home() / '.sigil' / 'settings.json')
    user_profiles = merge_settings(user, extra_settings()).get('profiles')
    user_profiles = user_profiles if isinstance(user_profiles, dict) else {}
    profiles: dict[str, dict[str, Any]] = {}
    try:
        project = load_settings(cwd, include_user=False).get('profiles')
    except Exception as exc:  # noqa: BLE001 - a broken file is reported by read_settings_file
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
        project = None
    if isinstance(project, dict):
        profiles.update({k: {**v, _FROM_PROJECT: True} for k, v in project.items()
                         if isinstance(v, dict) and user_profiles.get(k) != v})
    profiles.update({k: v for k, v in user_profiles.items() if isinstance(v, dict)})
    try:
        data = json.loads((_home() / '.sigil' / 'profiles.json').read_text(encoding='utf-8'))
        if isinstance(data, dict):
            profiles.update({k: v for k, v in data.get('profiles', data).items() if isinstance(v, dict)})
    except FileNotFoundError as _ignored:
        _log_ignored(_ignored)
        pass
    except (OSError, ValueError) as exc:
        from .debug_log import get_logger

        get_logger('profiles').warning('~/.sigil/profiles.json is ignored: %s', exc)
    return profiles


def profile_config(config: Any, profile: dict[str, Any]) -> Any:
    base_url = profile.get('base_url') or profile.get('baseUrl') or config.base_url
    model = profile.get('model') or config.model
    key_env = profile.get('api_key_env') or profile.get('apiKeyEnv')
    if key_env and profile.get(_FROM_PROJECT) and _main_key_name(str(key_env), str(base_url)):
        # a repository profile naming the user's main key (OPENAI_API_KEY, SIGIL_API_KEY, another
        # host's SIGIL_API_KEY_<HOST>) would send it to the repository's chosen host
        from .debug_log import get_logger

        get_logger('profiles').warning('project profile for %s: api_key_env %s ignored (a main key; use %s)',
                                       _host(str(base_url)), key_env, host_key_env(str(base_url)))
        key_env = None
    api_key = (os.environ.get(str(key_env)) if key_env else None) or profile.get('api_key')
    if api_key:
        return replace(config, base_url=str(base_url), model=str(model), api_key=str(api_key))
    return config_for_endpoint(config, str(model), str(base_url))


_NOTIFIED_HOSTS: set[str] = set()
_MAIN_KEY_NAMES = {'OPENAI_API_KEY', 'SIGIL_API_KEY', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN'}


def _main_key_name(name: str, base_url: str) -> bool:
    """Is `name` one of the user's main model keys (not meant for the host at `base_url`)?"""
    upper = name.strip().upper()
    if upper in _MAIN_KEY_NAMES:
        return True
    return upper.startswith('SIGIL_API_KEY_') and upper != host_key_env(base_url)


def resolve_api_key(base_url: str, explicit: str | None = None, env: Any = None) -> str:
    """The key for the model server at `base_url`, in one order everywhere (chat, -p, resume,
    background workers, subagents): --api-key, then SIGIL_API_KEY_<HOST> (that server's own key),
    SIGIL_API_KEY, OPENAI_API_KEY, else 'local-token' (enough for local servers)."""
    if explicit:
        return str(explicit)
    env = os.environ if env is None else env
    for name in (host_key_env(str(base_url or '')), 'SIGIL_API_KEY', 'OPENAI_API_KEY'):
        value = str(env.get(name) or '').strip()
        if value:
            return value
    return 'local-token'


def host_key_env(url: str) -> str:
    """SIGIL_API_KEY_<HOST>: the key for another model server, e.g. SIGIL_API_KEY_OPENROUTER_AI."""
    import re
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or '').upper()
    return 'SIGIL_API_KEY_' + re.sub(r'[^A-Z0-9]+', '_', host).strip('_')


def config_for_endpoint(config: Any, model: str, base_url: str) -> Any:
    """A copy of `config` for `model` at `base_url` (fallback models, fastModel, editor model,
    profiles). The main API key goes only to its own server (same scheme, host and port).
    Why: `model@https://other.host/v1` used to send the main key to that other host. Another
    host gets its own key from SIGIL_API_KEY_<HOST>, otherwise the placeholder 'local-token'
    (enough for local servers), and a one-time notice says so."""
    model = model or config.model
    base_url = base_url or config.base_url
    if _host(str(base_url)) == _host(str(config.base_url)):
        api_key = config.api_key
    else:
        env_name = host_key_env(str(base_url))
        api_key = os.environ.get(env_name, '').strip()
        if not api_key:
            api_key = 'local-token'
            if env_name not in _NOTIFIED_HOSTS:
                _NOTIFIED_HOSTS.add(env_name)
                import sys

                print(f'{_host(str(base_url))}: the main API key is not sent to another server; set '
                      f'{env_name} if it needs a key.', file=sys.stderr)
    return replace(config, base_url=str(base_url), model=str(model), api_key=str(api_key), fallback_models=())


def _host(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return f'{parsed.scheme}://{(parsed.hostname or "").lower()}:{parsed.port or ""}'


def apply_profile(agent: Any, name: str, profiles: dict[str, dict[str, Any]] | None = None) -> None:
    from .anthropic_client import make_client

    profiles = profiles if profiles is not None else load_profiles(agent.runtime_config.cwd)
    if name not in profiles:
        raise KeyError(f'Unknown profile {name!r}. Profiles: {", ".join(profiles) or "(none)"}')
    agent.model_config = profile_config(agent.model_config, profiles[name])
    agent.client = make_client(agent.model_config)
    agent.tool_context = replace(agent.tool_context, model_config=agent.model_config)
