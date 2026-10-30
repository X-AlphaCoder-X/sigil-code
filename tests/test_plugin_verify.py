"""Plugin marketplaces: the source forms (./path, url + sha/ref, git-subdir, a direct
marketplace.json URL) and the checks (sha / sha256 pins, verdicts, strict mode, signed marketplaces)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src import plugin_marketplace as pm


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=a', '-c', 'commit.gpgsign=false', *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def make_plugin(folder: Path, name: str = 'fmt') -> Path:
    (folder / '.sigil-plugin').mkdir(parents=True)
    (folder / '.sigil-plugin' / 'plugin.json').write_text(json.dumps({'name': name}), encoding='utf-8')
    (folder / 'commands').mkdir()
    (folder / 'commands' / 'hello.md').write_text(f'Hello from {name}', encoding='utf-8')
    return folder


def git_repo(folder: Path) -> str:
    git('init', '-q', cwd=folder)
    git('add', '.', cwd=folder)
    git('commit', '-q', '-m', 'init', cwd=folder)
    return git('rev-parse', 'HEAD', cwd=folder)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        home = self.base / 'home'
        home.mkdir()
        for patcher in (patch('src.settings._home', return_value=home),
                        patch.dict(os.environ, {'SIGIL_PLUGIN_REQUIRE_VERIFIED': '', 'SIGIL_PLUGIN_REQUIRE_SIGNED': ''})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def market(self, plugins: list[dict], *, use_git: bool = True) -> Path:
        folder = self.base / 'market'
        (folder / '.sigil-plugin').mkdir(parents=True, exist_ok=True)
        (folder / '.sigil-plugin' / 'marketplace.json').write_text(json.dumps(
            {'name': 'team', 'owner': {'name': 'me'}, 'plugins': plugins}), encoding='utf-8')
        if use_git:
            git_repo(folder)
        self.assertIn('Added marketplace team', pm.command(f'marketplace add {folder}'))
        return folder


class PinTests(Base):
    def test_sha256_pin_verified_modified_and_strict(self) -> None:
        source = make_plugin(self.base / 'market' / 'plugins' / 'fmt')
        self.market([{'name': 'fmt', 'source': './plugins/fmt', 'sha256': pm.tree_sha256(source)}])
        self.assertIn('verified against its pinned sha256', pm.command('install fmt'))
        self.assertIn('fmt@team: verified', pm.command('verify'))
        installed = pm.root() / 'installed' / 'fmt@team'
        (installed / 'commands' / 'hello.md').write_text('tampered', encoding='utf-8')
        self.assertIn('fmt@team: MODIFIED', pm.command('verify fmt'))
        self.assertIn(installed, pm.installed_plugin_dirs())  # default: still loaded (with a log warning)
        with patch.dict(os.environ, {'SIGIL_PLUGIN_REQUIRE_VERIFIED': '1'}):
            self.assertNotIn(installed, pm.installed_plugin_dirs())
            self.assertIn('not loaded: strict mode', pm.command('verify fmt'))

    def test_sha256_mismatch_is_refused(self) -> None:
        make_plugin(self.base / 'market' / 'plugins' / 'fmt')
        self.market([{'name': 'fmt', 'source': './plugins/fmt', 'sha256': '0' * 64}])
        self.assertIn('do not match the pinned sha256', pm.command('install fmt'))
        self.assertFalse((pm.root() / 'installed' / 'fmt@team').exists())
        self.assertIn('No plugins installed', pm.command('verify'))

    def test_unpinned_and_strict_mode(self) -> None:
        make_plugin(self.base / 'market' / 'plugins' / 'fmt')
        self.market([{'name': 'fmt', 'source': './plugins/fmt'}])
        with patch.dict(os.environ, {'SIGIL_PLUGIN_REQUIRE_VERIFIED': '1'}):
            self.assertIn('not pinned', pm.command('install fmt'))
        self.assertIn('unpinned', pm.command('install fmt'))
        self.assertIn('fmt@team: unpinned', pm.command('verify'))

    def test_git_sources_with_sha_and_subdir(self) -> None:
        repo = make_plugin(self.base / 'linter-repo', 'linter')
        commit = git_repo(repo)
        mono = self.base / 'mono'
        make_plugin(mono / 'plugins' / 'part', 'part')
        mono_commit = git_repo(mono)
        self.market([
            {'name': 'linter', 'source': {'source': 'url', 'url': str(repo), 'sha': commit}},
            {'name': 'wrong', 'source': {'source': 'url', 'url': str(repo), 'sha': 'f' * 40}},
            {'name': 'part', 'source': {'source': 'git-subdir', 'url': str(mono), 'path': 'plugins/part',
                                        'sha': mono_commit}},
            {'name': 'pkg', 'source': {'source': 'npm', 'package': 'x'}}])
        self.assertIn('verified against its pinned sha', pm.command('install linter'))
        self.assertEqual(pm.load_state()['plugins']['linter@team']['commit'], commit)
        self.assertIn('/plugin: ', pm.command('install wrong'))
        self.assertFalse((pm.root() / 'installed' / 'wrong@team').exists())
        self.assertIn('Installed part@team', pm.command('install part'))
        self.assertTrue((pm.root() / 'installed' / 'part@team' / 'commands' / 'hello.md').exists())
        self.assertFalse((pm.root() / 'installed' / 'part@team' / 'plugins').exists())  # only the subdir
        self.assertIn('not supported here', pm.command('install pkg'))


class MarketplaceRemoveAndShorthandTests(Base):
    def test_git_subdir_owner_repo_shorthand(self) -> None:
        urls: list[str] = []

        def fake_clone(url: str, target: Path, *, ref=None, sha=None) -> str:  # noqa: ANN001
            urls.append(url)
            make_plugin(Path(target) / 'plugins' / 'part', 'part')
            return 'c0ffee'

        self.market([{'name': 'part', 'source': {'source': 'git-subdir', 'url': 'acme/mono', 'path': 'plugins/part'}}])
        with patch('src.plugin_marketplace._clone', side_effect=fake_clone):
            self.assertIn('Installed part@team', pm.command('install part'))
        self.assertEqual(urls, ['https://github.com/acme/mono.git'])
        self.assertTrue((pm.root() / 'installed' / 'part@team' / 'commands' / 'hello.md').exists())

    def test_remove_marketplace_uninstalls_its_plugins(self) -> None:
        make_plugin(self.base / 'market' / 'plugins' / 'fmt')
        self.market([{'name': 'fmt', 'source': './plugins/fmt'}])
        self.assertIn('Installed fmt@team', pm.command('install fmt'))
        installed = pm.root() / 'installed' / 'fmt@team'
        self.assertTrue(installed.exists())
        self.assertEqual(pm.command('marketplace remove team'), 'Removed marketplace team and uninstalled fmt@team.')
        self.assertFalse(installed.exists())
        self.assertEqual(pm.load_state()['plugins'], {})
        self.assertNotIn(installed, pm.installed_plugin_dirs())


class MarketplaceSourceTests(Base):
    def test_direct_marketplace_json_url(self) -> None:
        manifest = {'name': 'remote', 'owner': {'name': 'x'}, 'plugins': [{'name': 'local', 'source': './p'}]}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(manifest).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_address[1]}/marketplace.json'
        self.assertIn('Added marketplace remote', pm.command(f'marketplace add {url}'))
        self.assertEqual(pm.load_state()['marketplaces']['remote']['kind'], 'json')
        self.assertIn('cannot hold "./" sources', pm.command('install local'))

    def test_signed_marketplace_required(self) -> None:
        folder = self.base / 'market'
        (folder / '.sigil-plugin').mkdir(parents=True)
        (folder / '.sigil-plugin' / 'marketplace.json').write_text(json.dumps({'name': 'team', 'plugins': []}),
                                                                    encoding='utf-8')
        git_repo(folder)  # an unsigned commit
        with patch.dict(os.environ, {'SIGIL_PLUGIN_REQUIRE_SIGNED': '1'}):
            self.assertIn('not signed', pm.command(f'marketplace add {folder}'))
        self.assertNotIn('team', pm.load_state()['marketplaces'])
        self.assertIn('Added marketplace team', pm.command(f'marketplace add {folder}'))

    def test_plugin_variables(self) -> None:
        from src.sigil_plugins import expand_root, plugin_data_dir

        root = self.base / 'plug'
        text = expand_root('${SIGIL_PLUGIN_ROOT}|${SIGIL_PLUGIN_DATA}|${SIGIL_PROJECT_DIR}', root)
        self.assertEqual(text.split('|'), [str(root), str(plugin_data_dir(root)), str(Path.cwd())])
        self.assertTrue(plugin_data_dir(root).is_dir())


if __name__ == '__main__':
    unittest.main()
