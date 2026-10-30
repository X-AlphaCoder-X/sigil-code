"""/plugin: install plugins from a marketplace, and check them (no hosted service).

A marketplace is a git repository (or a local folder, or a direct URL to a marketplace.json) with
.sigil-plugin/marketplace.json (or marketplace.json at its root):
    {"name": "team-tools", "owner": {"name": "..."}, "plugins": [
        {"name": "formatter", "source": "./plugins/formatter", "sha256": "<optional tree hash>"},
        {"name": "linter", "source": {"source": "github", "repo": "org/linter", "ref": "v2", "sha": "<commit>"}},
        {"name": "tools", "source": {"source": "url", "url": "https://git.example/tools.git", "sha": "..."}},
        {"name": "part", "source": {"source": "git-subdir", "url": "https://git.example/mono.git", "path": "plugins/part"}}]}
Sources: "./path" (inside the marketplace),
{"source": "github"|"url", ..., "ref", "sha"} and {"source": "git-subdir", "url", "path"}.
"sha256" (a hash of the plugin's files) is this project's own pin for folder sources.

  /plugin marketplace add <git URL[#ref] | owner/repo | folder | URL to marketplace.json>
  /plugin marketplace list | update <name> | remove <name>
  /plugin install <plugin>[@<marketplace>]     /plugin uninstall <plugin>
  /plugin list                                 /plugin verify [<plugin>]
Everything lives in ~/.sigil/plugins (marketplaces/, installed/, installed.json); installed
plugins load at start-up like --plugin-dir folders.

Checks. At install a pinned plugin must match its pin ("sha" commit or "sha256" file hash) or
it is not installed; the hash of its files is recorded. /plugin verify gives each plugin a
verdict: verified (pinned and unchanged), unpinned (unchanged since install, but nothing to
compare it with), MODIFIED (its files changed after install). Strict mode
(SIGIL_PLUGIN_REQUIRE_VERIFIED=1 or "plugins": {"requireVerified": true}) refuses unpinned
installs and does not load MODIFIED plugins; SIGIL_PLUGIN_REQUIRE_SIGNED=1 (or
"requireSignedMarketplaces") needs `git verify-commit` to pass on a git marketplace. Installing
copies files only, but a plugin's hooks and MCP servers run once it loads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib import request
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


def root() -> Path:
    from .settings import _home

    return _home() / '.sigil' / 'plugins'


def _state_path() -> Path:
    return root() / 'installed.json'


def load_state() -> dict[str, Any]:
    try:
        from .settings import read_text_retry

        data = json.loads(read_text_retry(_state_path()))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        data = {}
    data = data if isinstance(data, dict) else {}
    data.setdefault('marketplaces', {})
    data.setdefault('plugins', {})
    return data


def _save_state(state: dict[str, Any]) -> None:
    from .settings import write_text_atomic

    write_text_atomic(_state_path(), json.dumps(state, indent=2))  # other processes read it


def _policy(name: str, env: str) -> bool:
    if os.environ.get(env, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return True
    try:
        from .settings import load_settings

        plugins = load_settings(Path.cwd()).get('plugins')
    except Exception as _ignored:
        _log_ignored(_ignored)
        plugins = None
    return isinstance(plugins, dict) and plugins.get(name) is True


def require_verified() -> bool:
    return _policy('requireVerified', 'SIGIL_PLUGIN_REQUIRE_VERIFIED')


def require_signed() -> bool:
    return _policy('requireSignedMarketplaces', 'SIGIL_PLUGIN_REQUIRE_SIGNED')


def tree_sha256(folder: Path) -> str:
    """SHA-256 over every file's relative path and bytes (sorted; .git left out)."""
    digest = hashlib.sha256()
    folder = Path(folder)
    for path in sorted(p for p in folder.rglob('*') if p.is_file() and '.git' not in p.relative_to(folder).parts):
        digest.update(path.relative_to(folder).as_posix().encode('utf-8') + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def verdict(entry: dict[str, Any]) -> str:
    path = Path(entry.get('path', ''))
    if not path.is_dir():
        return 'MISSING'
    if tree_sha256(path) != entry.get('tree_sha256'):
        return 'MODIFIED'
    return 'verified' if entry.get('pin') else 'unpinned'


def installed_plugin_dirs() -> list[Path]:
    folders = []
    strict = None
    for key, entry in load_state()['plugins'].items():
        if not isinstance(entry, dict) or not Path(entry.get('path', '')).is_dir():
            continue
        strict = require_verified() if strict is None else strict
        if strict and verdict(entry) == 'MODIFIED':
            try:
                from .debug_log import get_logger

                get_logger('plugins').warning('not loading %s: its files changed after install', key)
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
            continue
        folders.append(Path(entry['path']))
    return folders


def _rmtree(path: Any) -> None:
    """Delete a folder; git's read-only object files on Windows are made writable first."""
    import stat

    def _unlock(func: Any, target: str, _info: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass

    if Path(path).exists():
        shutil.rmtree(path, onerror=_unlock)


def _git(*args: str, cwd: Path | None = None, restrict: bool = False) -> str:
    config: list[str] = []
    env = None
    if restrict:
        # S12: a remote marketplace's source may use only https/ssh (and http/git when sha-pinned);
        # block file/ext/local so a hostile marketplace.json cannot make git read the local machine.
        config = ['-c', 'protocol.allow=never', '-c', 'protocol.https.allow=always',
                  '-c', 'protocol.ssh.allow=always', '-c', 'protocol.http.allow=always',
                  '-c', 'protocol.git.allow=always']
        env = {**os.environ, 'GIT_PROTOCOL_FROM_USER': '0', 'GIT_TERMINAL_PROMPT': '0'}
    result = subprocess.run(['git', *config, *args], cwd=cwd, capture_output=True, text=True, timeout=300, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'git {" ".join(args[:2])} failed: {result.stderr.strip()[:400]}')
    return result.stdout.strip()


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, '') or default)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return default
    return value if value > 0 else default


def _clone(url: str, target: Path, *, ref: str | None = None, sha: str | None = None,
           restrict: bool = False) -> str:
    """Clone (at ref, or exactly at sha) and return the commit checked out.

    url, ref and sha come from a marketplace.json someone else wrote: a value starting with '-'
    would be read by git as an option (`--upload-pack=<program>` runs a program), so such values
    are refused, `--` ends the options before the URL, and a sha must be hex. With restrict=True
    (a remote marketplace, S12) git is run with protocol limits so only https/ssh/http/git work."""
    for label, value in (('url', url), ('ref', ref), ('sha', sha)):
        if value is not None and str(value).lstrip().startswith('-'):
            raise ValueError(f'plugin source {label} may not start with "-": {str(value)[:80]}')
    if sha and not re.fullmatch(r'[0-9a-fA-F]{7,64}', sha):
        raise ValueError(f'plugin source sha must be a hex commit id: {sha[:80]}')
    if sha:
        _git('clone', '--quiet', '--', url, str(target), restrict=restrict)
        _git('checkout', '--quiet', sha, cwd=target, restrict=restrict)
    elif ref:
        _git('clone', '--quiet', '--depth', '1', '--branch', ref, '--', url, str(target), restrict=restrict)
    else:
        _git('clone', '--quiet', '--depth', '1', '--', url, str(target), restrict=restrict)
    head = _git('rev-parse', 'HEAD', cwd=target)
    if sha and not head.startswith(sha.lower()):
        raise ValueError(f'commit {head} does not match the pinned sha {sha}')
    return head


def _git_url(source: str) -> str:
    if re.fullmatch(r'[\w.-]+/[\w.-]+', source) and not Path(source).exists():
        return f'https://github.com/{source}.git'  # owner/repo
    return source


def _market_is_remote(market: dict[str, Any]) -> bool:
    """Was this marketplace itself fetched from a remote source? A remote marketplace.json is not
    trusted to point plugin sources at the local machine (S12). A marketplace the user added from a
    local folder or their own local git keeps the old, permissive behaviour."""
    if market.get('kind') == 'json':
        return True
    source = str(market.get('source') or '')
    if re.match(r'^(https?|git|ssh)://', source) or re.match(r'^[\w.-]+@[\w.-]+:', source):
        return True
    if re.fullmatch(r'[\w.-]+/[\w.-]+', source) and not Path(source).exists():
        return True  # owner/repo shorthand
    return False


def _check_remote_source_url(url: str, sha: str | None) -> None:
    """A plugin `url`/`git-subdir` source inside a remote marketplace must be https or ssh; http and
    git:// only with a sha pin; never a local path, file:// or a UNC share (those make git touch the
    local machine). Override: SIGIL_MARKETPLACE_ALLOW_INSECURE_SOURCES=1 (fail closed)."""
    if os.environ.get('SIGIL_MARKETPLACE_ALLOW_INSECURE_SOURCES', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return
    text = str(url).strip()
    if re.match(r'^[\w.-]+@[\w.-]+:', text):  # scp-like git@host:path (ssh)
        return
    scheme = re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*)://', text)
    name = (scheme.group(1).lower() if scheme else '')
    if name in ('https', 'ssh'):
        return
    if name in ('http', 'git') and sha:
        return
    raise ValueError(f'a remote marketplace may not use source URL {text[:100]!r} '
                     '(use https or ssh; http/git need a pinned sha; set '
                     'SIGIL_MARKETPLACE_ALLOW_INSECURE_SOURCES=1 to override)')


def _read_manifest(folder: Path) -> dict[str, Any]:
    for candidate in (folder / '.sigil-plugin' / 'marketplace.json', folder / 'marketplace.json'):
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding='utf-8'))
            if isinstance(data, dict) and isinstance(data.get('plugins'), list):
                return data
    raise ValueError(f'No marketplace.json in {folder}')


def _safe_name(value: str) -> str:
    return re.sub(r'[^\w.-]+', '-', value).strip('-') or 'marketplace'


def add_marketplace(source: str) -> str:
    source = source.strip()
    if not source:
        raise ValueError('Usage: /plugin marketplace add <git URL[#ref] | owner/repo | folder | URL to marketplace.json>')
    staging = root() / 'marketplaces' / '.incoming'
    _rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    local = Path(source).expanduser()
    kind, head = 'git', None
    if re.match(r'^https?://', source) and source.split('?')[0].endswith('.json'):
        limit = _env_int('SIGIL_MARKETPLACE_MAX_BYTES', 5_000_000)
        with request.urlopen(source, timeout=30) as response:  # a direct link to marketplace.json
            data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f'{source} is larger than SIGIL_MARKETPLACE_MAX_BYTES ({limit} bytes)')
        manifest = json.loads(data.decode('utf-8'))
        if not isinstance(manifest, dict) or not isinstance(manifest.get('plugins'), list):
            raise ValueError(f'{source} is not a marketplace.json')
        (staging / '.sigil-plugin').mkdir(parents=True)
        (staging / '.sigil-plugin' / 'marketplace.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        folder, kind = staging, 'json'
    elif local.is_dir() and not (local / '.git').exists():
        folder, kind = local.resolve(), 'folder'  # a plain folder is used where it is
    else:
        url, _, ref = (str(local.resolve()) if local.is_dir() else source).partition('#')
        head = _clone(_git_url(url), staging, ref=ref or None)
        if require_signed():
            try:
                _git('verify-commit', 'HEAD', cwd=staging)
            except RuntimeError as exc:
                _rmtree(staging)
                raise ValueError(f'the marketplace commit is not signed or the signature is not trusted: {exc}')
        folder = staging
    manifest = _read_manifest(folder)
    name = _safe_name(str(manifest.get('name') or Path(source).name))
    if folder == staging:
        target = root() / 'marketplaces' / name
        _rmtree(target)
        staging.rename(target)
        folder = target
    state = load_state()
    state['marketplaces'][name] = {'source': source, 'path': str(folder), 'kind': kind, 'commit': head}
    _save_state(state)
    return f'Added marketplace {name} ({len(manifest["plugins"])} plugin(s)): {folder}'


def _marketplace(name: str) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    entry = load_state()['marketplaces'].get(name)
    if not isinstance(entry, dict):
        raise ValueError(f'Unknown marketplace: {name}')
    folder = Path(entry['path'])
    return name, folder, _read_manifest(folder), entry


def _place(item: dict[str, Any], market_folder: Path, market: dict[str, Any], target: Path) -> tuple[str | None, str]:
    """Put the plugin's files in target. Returns (commit or None, what was pinned or '')."""
    source = item.get('source')
    if isinstance(source, str) and source.startswith(('./', '../')):
        if market.get('kind') == 'json':
            raise ValueError('a marketplace added by URL to its marketplace.json cannot hold "./" sources')
        path = (market_folder / source).resolve()
        if market_folder.resolve() not in (path, *path.parents):
            raise ValueError(f'source {source} is outside the marketplace')
        if not path.is_dir():
            raise ValueError(f'{path} is not a folder')
        shutil.copytree(path, target, ignore=shutil.ignore_patterns('.git'))
        return None, 'sha256' if item.get('sha256') else ''
    if isinstance(source, str):
        raise ValueError(f'unsupported source {source!r} (use "./path" or a source object)')
    if not isinstance(source, dict):
        raise ValueError('the plugin has no source')
    kind, sha, ref = source.get('source'), source.get('sha'), source.get('ref')
    remote = _market_is_remote(market)
    if kind == 'github':
        commit = _clone(f'https://github.com/{source["repo"]}.git', target, ref=ref, sha=sha)
    elif kind == 'url':
        url = _git_url(str(source['url']))
        extra = {'restrict': True} if remote else {}  # keep the old call shape for local markets
        if remote:
            _check_remote_source_url(url, sha)
        commit = _clone(url, target, ref=ref, sha=sha, **extra)
    elif kind == 'git-subdir':
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'repo'
            url = _git_url(str(source['url']))  # url or owner/repo
            extra = {'restrict': True} if remote else {}
            if remote:
                _check_remote_source_url(url, sha)
            commit = _clone(url, checkout, ref=ref, sha=sha, **extra)
            sub = (checkout / str(source.get('path', ''))).resolve()
            if checkout.resolve() not in (sub, *sub.parents) or not sub.is_dir():
                raise ValueError(f'git-subdir path {source.get("path")!r} is not a folder in the repository')
            shutil.copytree(sub, target, ignore=shutil.ignore_patterns('.git'))
    else:
        raise ValueError(f'source type {kind!r} is not supported here (github, url, git-subdir or "./path")')
    return commit, 'sha' if sha else ''


def install(spec: str) -> str:
    plugin, _, market_name = spec.strip().partition('@')
    if not plugin:
        raise ValueError('Usage: /plugin install <plugin>[@<marketplace>]')
    matches = []
    for name in ([market_name] if market_name else list(load_state()['marketplaces'])):
        name, folder, manifest, market = _marketplace(name)
        matches += [(name, folder, market, item) for item in manifest['plugins']
                    if isinstance(item, dict) and item.get('name') == plugin]
    if not matches:
        raise ValueError(f'No plugin {plugin!r} in ' + (market_name or 'any added marketplace'))
    if len(matches) > 1:
        raise ValueError(f'{plugin} is in several marketplaces; use {plugin}@<marketplace>')
    name, folder, market, item = matches[0]
    target = root() / 'installed' / f'{plugin}@{name}'
    _rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit, pin = _place(item, folder, market, target)
        tree = tree_sha256(target)
        if pin == 'sha256' and tree != str(item['sha256']).lower():
            raise ValueError(f'the files do not match the pinned sha256 ({tree} != {item["sha256"]})')
        if not pin and require_verified():
            raise ValueError('it is not pinned (no "sha" or "sha256") and strict mode is on')
    except (ValueError, RuntimeError, OSError, KeyError):
        _rmtree(target)
        raise
    state = load_state()
    state['plugins'][f'{plugin}@{name}'] = {'path': str(target), 'marketplace': name,
                                            'version': str(item.get('version') or ''), 'commit': commit,
                                            'pin': pin, 'tree_sha256': tree}
    _save_state(state)
    status = (f'verified against its pinned {pin}' if pin else
              'unpinned: the marketplace gives no sha or sha256, so only later changes can be detected')
    return (f'Installed {plugin}@{name} into {target} ({status}). '
            'It loads at the next start (or /reload-plugins).')


def verify(name: str = '') -> str:
    plugins = load_state()['plugins']
    keys = [key for key in plugins if not name or key == name or key.split('@')[0] == name]
    if not keys:
        return f'{name} is not installed' if name else 'No plugins installed.'
    lines = ['Plugin checks:']
    for key in keys:
        result = verdict(plugins[key])
        note = {'verified': f'pinned {plugins[key].get("pin")} matched at install; files unchanged',
                'unpinned': 'files unchanged since install; no pin to compare with',
                'MODIFIED': 'files changed after install' + (' (not loaded: strict mode)' if require_verified() else ''),
                'MISSING': 'folder is gone'}[result]
        lines.append(f'  {key}: {result} ({note})')
    return '\n'.join(lines)


def uninstall(plugin: str) -> str:
    state = load_state()
    keys = [key for key in state['plugins'] if key == plugin or key.split('@')[0] == plugin]
    if not keys:
        raise ValueError(f'{plugin} is not installed')
    for key in keys:
        _rmtree(state['plugins'][key]['path'])
        del state['plugins'][key]
    _save_state(state)
    return f'Uninstalled {", ".join(keys)}.'


def command(args: str) -> str:
    """The /plugin command (and `sigil plugin ...`)."""
    words = args.split()
    try:
        if not words or words[0] == 'list':
            state = load_state()
            return '\n'.join(['Marketplaces: ' + (', '.join(state['marketplaces']) or 'none'),
                              'Installed: ' + (', '.join(state['plugins']) or 'none')])
        if words[0] == 'verify':
            return verify(words[1] if len(words) > 1 else '')
        if words[0] == 'marketplace':
            action = words[1] if len(words) > 1 else 'list'
            if action == 'add':
                return add_marketplace(' '.join(words[2:]))
            if action == 'list':
                rows = []
                for market in load_state()['marketplaces']:
                    _, _folder, manifest, _entry = _marketplace(market)
                    plugins = ', '.join(str(p.get('name')) for p in manifest['plugins'] if isinstance(p, dict))
                    rows.append(f'  {market}: {plugins}')
                return 'Marketplaces:\n' + ('\n'.join(rows) or '  none')
            if action in ('remove', 'update') and len(words) > 2:
                market = words[2]
                state = load_state()
                entry = state['marketplaces'].get(market)
                if not entry:
                    raise ValueError(f'Unknown marketplace: {market}')
                if action == 'update':
                    if (Path(entry['path']) / '.git').exists():
                        _git('pull', '--ff-only', cwd=Path(entry['path']))
                        if require_signed():
                            _git('verify-commit', 'HEAD', cwd=Path(entry['path']))
                    return f'Updated {market}.'
                if Path(entry['path']).is_relative_to(root()):
                    _rmtree(entry['path'])
                removed = [key for key in state['plugins'] if key.rpartition('@')[2] == market]
                for key in removed:  # its plugins are uninstalled too
                    _rmtree(state['plugins'][key]['path'])
                    del state['plugins'][key]
                del state['marketplaces'][market]
                _save_state(state)
                return f'Removed marketplace {market}' + (f' and uninstalled {", ".join(removed)}.' if removed else '.')
        if words[0] == 'install' and len(words) > 1:
            return install(words[1])
        if words[0] in ('uninstall', 'remove') and len(words) > 1:
            return uninstall(words[1])
    except (ValueError, RuntimeError, OSError) as exc:
        return f'/plugin: {exc}'
    return ('Usage: /plugin list | verify [plugin] | marketplace add <git URL[#ref]|owner/repo|folder|URL to '
            'marketplace.json> | marketplace list|update|remove <name> | install <plugin>[@marketplace] | '
            'uninstall <plugin>')
