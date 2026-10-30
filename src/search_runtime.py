from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError

from .debug_log import get_logger
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# Module logger: under the sigil logger tree, so warnings reach ~/.sigil/logs (a logger named
# src.search_runtime has no handler and its lines were lost).
_logger = get_logger('search')

# Cross-platform file locking
try:
    import fcntl  # Unix
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows

try:
    import msvcrt  # Windows
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False  # Unix


DEFAULT_SEARCH_STATE_DIR = Path('.port_sessions')
DEFAULT_SEARCH_STATE_FILE = DEFAULT_SEARCH_STATE_DIR / 'search_runtime.json'


def default_usage_file() -> Path:
    """Per-key usage counts: one file per user (the keys are the user's, not a project's), not in
    the folder Sigil happened to start in, and not the file that holds the active provider."""
    return Path.home() / '.sigil' / 'search_usage.json'
SEARCH_MANIFEST_PATHS = (
    Path('.sigil-search.json'),
    Path('.sigil/search.json'),
)
DEFAULT_SEARXNG_BASE_URL = 'http://127.0.0.1:8080'
DEFAULT_BRAVE_BASE_URL = 'https://api.search.brave.com/res/v1/web/search'
DEFAULT_TAVILY_BASE_URL = 'https://api.tavily.com/search'

# Key rotation configuration
DEFAULT_MAX_REQUESTS_PER_KEY = 1000
TAVILY_KEY_ENVS = ('TAVILY_API_KEY', 'TAVILY_API_KEY_2')
USAGE_RESET_SECONDS = 30 * 24 * 60 * 60  # 30 days in seconds
# 401 invalid key, 403 forbidden, 429 rate limited, 432/433 Tavily plan/pay-as-you-go limit
_TAVILY_KEY_FAILOVER_CODES = {401, 403, 429, 432, 433}


@dataclass(frozen=True)
class SearchProviderProfile:
    name: str
    provider: str
    source_manifest: str
    base_url: str
    api_key_env: str | None = None
    description: str | None = None
    default_max_results: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider_name: str
    rank: int


@dataclass
class SearchUsageTracker:
    """Tracks API request counts for key rotation with 30-day monthly reset.
    
    Thread-safe implementation with in-memory caching and atomic file writes.
    """
    state_path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _usage_cache: dict[str, int] = field(default_factory=dict, repr=False)
    _period_start_cache: float = field(default=0.0, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def __post_init__(self) -> None:
        """Load state from disk into in-memory cache on initialization."""
        # Convert string to Path if needed
        if isinstance(self.state_path, str):
            object.__setattr__(self, 'state_path', Path(self.state_path))
        self._load_from_disk()
        object.__setattr__(self, '_initialized', True)
    
    def _acquire_file_lock(self, file_obj, timeout: float = 10.0) -> bool:
        """Acquire exclusive file lock for cross-process safety.
        
        Returns True if lock acquired, False if timeout.
        """
        if _HAS_FCNTL:
            # Unix: use LOCK_EX (blocks until acquired)
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            return True
        elif _HAS_MSVCRT:
            # Windows: LK_NBLCK is non-blocking, retry with timeout
            import time as _time
            start = _time.monotonic()
            while True:
                try:
                    file_obj.seek(0)
                    msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 0x7FFFFFFF)
                    return True
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    if _time.monotonic() - start > timeout:
                        return False
                    _time.sleep(0.01)  # 10ms retry interval
        return False
    
    def _release_file_lock(self, file_obj) -> None:
        """Release file lock."""
        if _HAS_FCNTL:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        elif _HAS_MSVCRT:
            # Windows: unlock the same large region
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 0x7FFFFFFF)
    
    def _ensure_lock_file(self, lock_path: Path) -> None:
        """Create lock file with secure permissions if it doesn't exist."""
        if lock_path.exists():
            # Ensure permissions are correct on existing lock file
            try:
                os.chmod(lock_path, 0o600)
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass  # Best effort on Windows
            return
        
        # Try to create with O_EXCL for atomic creation
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        except FileExistsError as _ignored:
            # Race condition: another thread/process created it first
            _log_ignored(_ignored)
            try:
                os.chmod(lock_path, 0o600)
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass  # Best effort on Windows
    
    def _load_from_disk(self) -> None:
        """Load usage and period_start from disk into cache (called once on init)."""
        if not self.state_path.exists():
            self._period_start_cache = 0.0
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding='utf-8'))
            usage = payload.get('api_usage', {})
            self._usage_cache = {k: int(v) for k, v in usage.items() if isinstance(v, (int, float))}
            period_start = payload.get('period_start', 0)
            # Fix: Handle None explicitly to avoid TypeError
            self._period_start_cache = float(period_start) if period_start else 0.0
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as _ignored:
            _log_ignored(_ignored)
            self._usage_cache = {}
            self._period_start_cache = 0.0
    
    def _with_file_lock(self, operation: str) -> None:
        """Execute a state operation with cross-process file locking.
        
        Args:
            operation: 'load', 'save', or 'load_save' for combined read-modify-write
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Ensure permissions on existing directory (mkdir with mode doesn't change existing dirs)
        try:
            os.chmod(self.state_path.parent, 0o700)
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass  # Best effort on Windows
        lock_path = self.state_path.with_suffix('.lock')
        
        # Ensure lock file exists with secure permissions
        self._ensure_lock_file(lock_path)
        
        # Open lock file and hold lock during entire operation
        # Use 'r+' to avoid truncating before lock is acquired
        lock_file = open(lock_path, 'r+')
        try:
            if not self._acquire_file_lock(lock_file):
                raise OSError("Lock acquisition timed out")
            try:
                if operation == 'load':
                    self._load_from_disk()
                elif operation == 'save':
                    self._save_to_disk_atomic()
                elif operation == 'load_save':
                    # Combined read-modify-write - load, then save
                    self._load_from_disk()
                    self._save_to_disk_atomic()
            finally:
                self._release_file_lock(lock_file)
        except OSError as e:
            # Fall back to unlocked operation if lock fails
            _logger.warning("Failed to acquire file lock for %s operation: %s. Proceeding without lock.", operation, e)
            if operation == 'load':
                self._load_from_disk()
            elif operation == 'save':
                self._save_to_disk_atomic()
            elif operation == 'load_save':
                self._load_from_disk()
                self._save_to_disk_atomic()
        finally:
            lock_file.close()
    
    def _load_from_disk_locked(self) -> None:
        """Load state from disk with file lock for cross-process safety."""
        self._with_file_lock('load')
    
    def _save_to_disk_atomic(self) -> None:
        """Persist cache to disk atomically using unique temp file + os.replace().
        
        Uses retry with small delay to handle transient Windows file locking
        from antivirus or file indexing services.
        Sets restrictive file permissions (0o600) for security.
        """
        # Create directory with restrictive permissions
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            'api_usage': self._usage_cache,
            'period_start': self._period_start_cache,
        }
        # Atomic write: use unique temp file in same directory, then replace
        fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='search_usage_',
            dir=str(self.state_path.parent)
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + '\n')
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass
            raise
        # Retry os.replace up to 10 times to handle transient Windows file locks
        # Windows can hold file locks briefly due to antivirus, indexing, or concurrent threads
        for attempt in range(10):
            try:
                os.replace(temp_path, self.state_path)
                # Set restrictive permissions on the final file
                try:
                    os.chmod(self.state_path, 0o600)
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    pass  # Best effort on Windows
                return
            except PermissionError:
                if attempt < 9:
                    # Exponential backoff: 10ms, 20ms, 40ms, 80ms, 160ms, 320ms, 640ms, 1.28s, 2.56s
                    time.sleep(0.01 * (2 ** attempt))
                else:
                    try:
                        os.unlink(temp_path)
                    except OSError as _ignored:
                        _log_ignored(_ignored)
                        pass
                    raise
    
    def _check_and_reset_if_expired(self) -> None:
        """Check if 30 days have passed and auto-reset if needed. Must hold lock."""
        current_time = time.time()
        
        # If no period start recorded, initialize it
        if self._period_start_cache == 0.0:
            self._period_start_cache = current_time
            self._save_to_disk_atomic()
            return
        
        # Check if 30 days have passed
        if current_time - self._period_start_cache >= USAGE_RESET_SECONDS:
            # Auto-reset: clear usage and set new period start
            self._usage_cache = {}
            self._period_start_cache = current_time
            self._save_to_disk_atomic()
    
    def get_usage(self) -> dict[str, int]:
        """Return usage counts from cache (auto-resets if 30 days passed)."""
        with self._lock:
            self._check_and_reset_if_expired()
            return dict(self._usage_cache)  # Return a copy
    
    def increment_usage(self, key_env: str) -> int:
        """Increment usage for a key and return new count. Thread-safe and cross-process safe.
        
        Args:
            key_env: Environment variable name (must be a valid identifier).
        
        Raises:
            ValueError: If key_env is not a valid identifier.
        """
        # Validate key_env to prevent injection attacks
        if not isinstance(key_env, str) or not key_env.isidentifier():
            raise ValueError(f"Invalid key_env: {key_env!r}. Must be a valid Python identifier.")
        with self._lock:
            # Hold file lock for entire read-modify-write cycle
            self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Ensure permissions on existing directory (mkdir with mode doesn't change existing dirs)
            try:
                os.chmod(self.state_path.parent, 0o700)
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass  # Best effort on Windows
            lock_path = self.state_path.with_suffix('.lock')
            
            # Ensure lock file exists with secure permissions
            self._ensure_lock_file(lock_path)
            
            # Use 'r+' to avoid truncating before lock is acquired
            lock_file = open(lock_path, 'r+')
            try:
                if not self._acquire_file_lock(lock_file):
                    raise OSError("Lock acquisition timed out")
                try:
                    # Load latest state from disk
                    self._load_from_disk()
                    self._check_and_reset_if_expired()
                    new_count = self._usage_cache.get(key_env, 0) + 1
                    self._usage_cache[key_env] = new_count
                    # Save while still holding lock
                    self._save_to_disk_atomic()
                finally:
                    self._release_file_lock(lock_file)
            except OSError as e:
                # Fall back to unlocked operation if lock fails
                _logger.warning("Failed to acquire file lock for increment_usage: %s. Proceeding without lock.", e)
                self._load_from_disk()
                self._check_and_reset_if_expired()
                new_count = self._usage_cache.get(key_env, 0) + 1
                self._usage_cache[key_env] = new_count
                self._save_to_disk_atomic()
            finally:
                lock_file.close()
            return new_count
    
    def get_available_key(self, key_envs: tuple[str, ...], max_requests: int) -> str | None:
        """Find first key that hasn't hit the limit. Thread-safe."""
        with self._lock:
            self._check_and_reset_if_expired()
            for key_env in key_envs:
                key_value = os.environ.get(key_env)
                if isinstance(key_value, str) and key_value.strip():
                    current_count = self._usage_cache.get(key_env, 0)
                    if current_count < max_requests:
                        return key_env
        return None
    
    def mark_exhausted(self, key_env: str, max_requests: int) -> None:
        """Mark a key as used up for the current period (server rejected it)."""
        if not isinstance(key_env, str) or not key_env.isidentifier():
            raise ValueError(f"Invalid key_env: {key_env!r}")
        with self._lock:
            self._check_and_reset_if_expired()
            self._usage_cache[key_env] = max(self._usage_cache.get(key_env, 0), max_requests)
            try:
                self._save_to_disk_atomic()
            except OSError as e:
                _logger.warning("Failed to persist exhausted key %s: %s", key_env, e)

    def is_key_exhausted(self, key_env: str, max_requests: int) -> bool:
        """Check if a key has hit its limit. Thread-safe."""
        with self._lock:
            self._check_and_reset_if_expired()
            return self._usage_cache.get(key_env, 0) >= max_requests
    
    def reset_usage(self, key_env: str | None = None) -> None:
        """Reset usage for a specific key or all keys. Thread-safe.
        
        Args:
            key_env: Environment variable name of the key to reset.
                     If None, resets all usage counts and starts new period.
        """
        with self._lock:
            if key_env is not None:
                self._usage_cache.pop(key_env, None)
            else:
                # Full reset: clear usage and set new period start
                self._usage_cache = {}
                self._period_start_cache = time.time()
            self._save_to_disk_atomic()
    
    def get_period_info(self) -> dict[str, Any]:
        """Get information about the current usage period. Thread-safe.
        
        Returns:
            Dict with 'period_start', 'period_end', 'seconds_remaining', 
            'days_remaining', and 'is_expired' keys.
        """
        with self._lock:
            self._check_and_reset_if_expired()
            current_time = time.time()
            
            if self._period_start_cache == 0.0:
                return {
                    'period_start': None,
                    'period_end': None,
                    'seconds_remaining': None,
                    'days_remaining': None,
                    'is_expired': False,
                }
            
            period_end = self._period_start_cache + USAGE_RESET_SECONDS
            seconds_remaining = max(0, period_end - current_time)
            days_remaining = seconds_remaining / (24 * 60 * 60)
            
            return {
                'period_start': self._period_start_cache,
                'period_end': period_end,
                'seconds_remaining': seconds_remaining,
                'days_remaining': days_remaining,
                'is_expired': seconds_remaining == 0,
            }


@dataclass(frozen=True)
class SearchStatusReport:
    configured: bool
    detail: str
    provider_name: str | None = None
    provider_kind: str | None = None
    base_url: str | None = None
    manifest_count: int = 0
    provider_count: int = 0
    api_key_env: str | None = None

    def as_text(self) -> str:
        lines = [
            f'configured={self.configured}',
            f'detail={self.detail}',
            f'manifest_count={self.manifest_count}',
            f'provider_count={self.provider_count}',
        ]
        if self.provider_name:
            lines.append(f'provider={self.provider_name}')
        if self.provider_kind:
            lines.append(f'provider_kind={self.provider_kind}')
        if self.base_url:
            lines.append(f'base_url={self.base_url}')
        if self.api_key_env:
            lines.append(f'api_key_env={self.api_key_env}')
        return '\n'.join(lines)


@dataclass
class SearchRuntime:
    cwd: Path
    providers: tuple[SearchProviderProfile, ...] = field(default_factory=tuple)
    manifests: tuple[str, ...] = field(default_factory=tuple)
    state_path: Path = field(default_factory=lambda: DEFAULT_SEARCH_STATE_FILE.resolve())
    active_provider_name: str | None = None

    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'SearchRuntime':
        manifest_paths = _discover_manifest_paths(cwd, additional_working_directories)
        providers: list[SearchProviderProfile] = []
        trusted = _folder_trusted(cwd)
        for manifest_path in manifest_paths:
            providers.extend(_key_stays_home(provider, trusted) for provider in _load_profiles_from_manifest(manifest_path))
        providers.extend(_load_profiles_from_env())
        providers = _dedupe_profiles(providers)
        state_path = (cwd.resolve() / DEFAULT_SEARCH_STATE_FILE).resolve()
        payload = _load_state_payload(state_path)
        active_provider_name = payload.get('active_provider_name')
        if not isinstance(active_provider_name, str):
            active_provider_name = None
        return cls(
            cwd=cwd.resolve(),
            providers=tuple(providers),
            manifests=tuple(str(path) for path in manifest_paths),
            state_path=state_path,
            active_provider_name=active_provider_name,
        )

    def has_search_runtime(self) -> bool:
        return bool(self.providers)

    def list_providers(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
    ) -> tuple[SearchProviderProfile, ...]:
        providers = self.providers
        if query:
            needle = query.lower()
            providers = tuple(
                provider
                for provider in providers
                if needle in provider.name.lower()
                or needle in provider.provider.lower()
                or needle in provider.base_url.lower()
                or needle in (provider.description or '').lower()
            )
        if limit is not None and limit >= 0:
            providers = providers[:limit]
        return providers

    def get_provider(self, name: str) -> SearchProviderProfile | None:
        needle = name.strip().lower()
        if not needle:
            return None
        for provider in self.providers:
            if provider.name.lower() == needle:
                return provider
        return None

    def current_provider(self) -> SearchProviderProfile | None:
        if self.active_provider_name:
            active = self.get_provider(self.active_provider_name)
            if active is not None:
                return active
        env_default = os.environ.get('SIGIL_SEARCH_PROVIDER')
        if isinstance(env_default, str) and env_default.strip():
            active = self.get_provider(env_default.strip())
            if active is not None:
                return active
        return self.providers[0] if self.providers else None

    def activate_provider(self, name: str) -> SearchStatusReport:
        provider = self.get_provider(name)
        if provider is None:
            raise KeyError(name)
        self.active_provider_name = provider.name
        self._persist_state()
        return SearchStatusReport(
            configured=True,
            detail=f'Activated search provider {provider.name}',
            provider_name=provider.name,
            provider_kind=provider.provider,
            base_url=provider.base_url,
            manifest_count=len(self.manifests),
            provider_count=len(self.providers),
            api_key_env=provider.api_key_env,
        )

    def render_summary(self) -> str:
        lines = [
            f'Local search manifests: {len(self.manifests)}',
            f'Configured search providers: {len(self.providers)}',
        ]
        current = self.current_provider()
        if current is None:
            lines.append('- Active search provider: none')
            return '\n'.join(lines)
        lines.append(f'- Active search provider: {current.name} ({current.provider})')
        for provider in self.providers[:5]:
            details = [provider.name, provider.provider, provider.base_url]
            if provider.api_key_env:
                details.append(f'api_key_env={provider.api_key_env}')
            lines.append('- Provider: ' + ' ; '.join(details))
        return '\n'.join(lines)

    def render_providers_index(self, *, query: str | None = None) -> str:
        providers = self.list_providers(query=query, limit=100)
        lines = ['# Search Providers', '']
        if not providers:
            lines.append('No local search providers discovered.')
            return '\n'.join(lines)
        for provider in providers:
            details = [provider.name, provider.provider, provider.base_url]
            if provider.api_key_env:
                details.append(f'api_key_env={provider.api_key_env}')
            lines.append('- ' + ' ; '.join(details))
        return '\n'.join(lines)

    def render_provider(self, name: str) -> str:
        provider = self.get_provider(name)
        if provider is None:
            return f'# Search Provider\n\nUnknown search provider: {name}'
        lines = [
            '# Search Provider',
            '',
            f'- Name: {provider.name}',
            f'- Provider: {provider.provider}',
            f'- Base URL: {provider.base_url}',
            f'- Source manifest: {provider.source_manifest}',
        ]
        if provider.api_key_env:
            lines.append(f'- API key env: {provider.api_key_env}')
        if provider.description:
            lines.extend(['', provider.description])
        return '\n'.join(lines)

    def render_search_results(
        self,
        query: str,
        *,
        provider_name: str | None = None,
        max_results: int = 5,
        domains: tuple[str, ...] = (),
        timeout_seconds: float = 20.0,
    ) -> str:
        provider, results = self.search(
            query,
            provider_name=provider_name,
            max_results=max_results,
            domains=domains,
            timeout_seconds=timeout_seconds,
        )
        lines = ['# Web Search', '']
        lines.append(f'- Provider: {provider.name} ({provider.provider})')
        lines.append(f'- Query: {query}')
        lines.append(f'- Results: {len(results)}')
        lines.append('')
        if not results:
            lines.append('No search results.')
            return '\n'.join(lines)
        for result in results:
            lines.append(f'{result.rank}. {result.title}')
            lines.append(f'   {result.url}')
            if result.snippet:
                lines.append(f'   {result.snippet}')
        return '\n'.join(lines)

    def search(
        self,
        query: str,
        *,
        provider_name: str | None = None,
        max_results: int = 5,
        domains: tuple[str, ...] = (),
        timeout_seconds: float = 20.0,
        blocked_domains: tuple[str, ...] = (),
    ) -> tuple[SearchProviderProfile, tuple[SearchResult, ...]]:
        provider = self._resolve_provider(provider_name)
        backend = provider.provider.lower()
        # Over-fetch when filtering client-side so we can still return max_results.
        fetch_count = min(max_results * 2, 20) if (domains or blocked_domains) else max_results
        if backend == 'searxng':
            results = _search_searxng(provider, query, max_results=fetch_count, timeout_seconds=timeout_seconds)
        elif backend == 'brave':
            results = _search_brave(provider, query, max_results=fetch_count, timeout_seconds=timeout_seconds)
        elif backend == 'tavily':
            results = _search_tavily(
                provider, query, max_results=fetch_count, domains=domains,
                timeout_seconds=timeout_seconds, blocked_domains=blocked_domains,
            )
        else:
            raise ValueError(f'Unsupported search provider: {provider.provider}')
        if domains:
            results = tuple(result for result in results if _matches_domains(result.url, domains))
        if blocked_domains:
            results = tuple(
                result for result in results if not _matches_domains(result.url, blocked_domains)
            )
        return provider, tuple(results[:max_results])

    def _resolve_provider(self, provider_name: str | None) -> SearchProviderProfile:
        if provider_name:
            provider = self.get_provider(provider_name)
            if provider is None:
                raise KeyError(provider_name)
            return provider
        provider = self.current_provider()
        if provider is None:
            raise LookupError('No local search provider is configured.')
        return provider

    def _persist_state(self) -> None:
        payload = {'active_provider_name': self.active_provider_name}
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Ensure permissions on existing directory (mkdir with mode doesn't change existing dirs)
        try:
            os.chmod(self.state_path.parent, 0o700)
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass  # Best effort on Windows
        from .settings import write_text_atomic

        write_text_atomic(
            self.state_path,
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )


def _discover_manifest_paths(cwd: Path, additional_working_directories: tuple[str, ...]) -> tuple[Path, ...]:
    candidate_roots = [cwd.resolve()]
    for raw_path in additional_working_directories:
        path = Path(raw_path).resolve()
        if path not in candidate_roots:
            candidate_roots.append(path)
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in candidate_roots:
        for relative_path in SEARCH_MANIFEST_PATHS:
            path = (root / relative_path).resolve()
            if path in seen or not path.exists() or not path.is_file():
                continue
            seen.add(path)
            discovered.append(path)
    return tuple(discovered)


def _load_profiles_from_manifest(path: Path) -> list[SearchProviderProfile]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as _ignored:
        _log_ignored(_ignored)
        return []
    if isinstance(payload, dict):
        providers_payload = payload.get('providers')
        if isinstance(providers_payload, list):
            return [
                provider
                for item in providers_payload
                for provider in [_provider_from_payload(item, path)]
                if provider is not None
            ]
        single = _provider_from_payload(payload, path)
        return [single] if single is not None else []
    return []


def _provider_from_payload(payload: Any, path: Path) -> SearchProviderProfile | None:
    if not isinstance(payload, dict):
        return None
    name = payload.get('name')
    provider = payload.get('provider')
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(provider, str) or not provider.strip():
        return None
    normalized_provider = provider.strip().lower()
    base_url = _optional_str(payload.get('baseUrl') or payload.get('base_url')) or _default_base_url(normalized_provider)
    if base_url is None:
        return None
    api_key_env = _optional_str(payload.get('apiKeyEnv') or payload.get('api_key_env')) or _default_api_env(normalized_provider)
    description = _optional_str(payload.get('description'))
    default_max_results = payload.get('defaultMaxResults') or payload.get('default_max_results') or 5
    if isinstance(default_max_results, bool) or not isinstance(default_max_results, int):
        default_max_results = 5
    metadata = payload.get('metadata')
    return SearchProviderProfile(
        name=name.strip(),
        provider=normalized_provider,
        source_manifest=str(path),
        base_url=base_url,
        api_key_env=api_key_env,
        description=description,
        default_max_results=max(default_max_results, 1),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _load_profiles_from_env() -> list[SearchProviderProfile]:
    providers: list[SearchProviderProfile] = []
    searxng_base = os.environ.get('SEARXNG_BASE_URL')
    if isinstance(searxng_base, str) and searxng_base.strip():
        providers.append(
            SearchProviderProfile(
                name='searxng',
                provider='searxng',
                source_manifest='env:SEARXNG_BASE_URL',
                base_url=searxng_base.strip(),
            )
        )
    brave_key = os.environ.get('BRAVE_SEARCH_API_KEY')
    if isinstance(brave_key, str) and brave_key.strip():
        providers.append(
            SearchProviderProfile(
                name='brave',
                provider='brave',
                source_manifest='env:BRAVE_SEARCH_API_KEY',
                base_url=DEFAULT_BRAVE_BASE_URL,
                api_key_env='BRAVE_SEARCH_API_KEY',
            )
        )
    tavily_key = os.environ.get('TAVILY_API_KEY')
    if isinstance(tavily_key, str) and tavily_key.strip():
        providers.append(
            SearchProviderProfile(
                name='tavily',
                provider='tavily',
                source_manifest='env:TAVILY_API_KEY',
                base_url=DEFAULT_TAVILY_BASE_URL,
                api_key_env='TAVILY_API_KEY',
            )
        )
    return providers


def _folder_trusted(cwd: Path) -> bool:
    try:
        from .workspace_trust import is_trusted

        return is_trusted(cwd)
    except Exception as exc:  # noqa: BLE001 - fail closed: keys stay home
        _logger.warning('folder trust check failed (%s); search keys go only to their own hosts', exc)
        return False


def _key_stays_home(provider: SearchProviderProfile, trusted: bool) -> SearchProviderProfile:
    """A repository manifest (.sigil-search.json, .sigil/search.json) may point a provider at any
    URL, but in a folder the user has not trusted the user's key (api_key_env) is sent only to that
    provider's own host (Tavily / Brave); elsewhere the provider runs without the key."""
    if trusted or not provider.api_key_env:
        return provider
    default = _default_base_url(provider.provider)

    def _origin(url: str) -> tuple[str, str, int | str]:  # scheme+host+port, not host alone (N42)
        parts = parse.urlsplit(url)
        return (parts.scheme.lower(), (parts.hostname or '').lower(), parts.port or '')

    if default and _origin(default) == _origin(provider.base_url):
        return provider
    _logger.warning('search provider %s from %s: key %s not sent to %s (folder not trusted)', provider.name,
                    provider.source_manifest, provider.api_key_env, parse.urlsplit(provider.base_url).hostname)
    from dataclasses import replace

    return replace(provider, api_key_env=None)


def _dedupe_profiles(providers: list[SearchProviderProfile]) -> list[SearchProviderProfile]:
    seen: set[str] = set()
    deduped: list[SearchProviderProfile] = []
    for provider in providers:
        key = provider.name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(provider)
    return deduped


def _load_state_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as _ignored:
        _log_ignored(_ignored)
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_base_url(provider: str) -> str | None:
    if provider == 'searxng':
        return DEFAULT_SEARXNG_BASE_URL
    if provider == 'brave':
        return DEFAULT_BRAVE_BASE_URL
    if provider == 'tavily':
        return DEFAULT_TAVILY_BASE_URL
    return None


def _default_api_env(provider: str) -> str | None:
    if provider == 'brave':
        return 'BRAVE_SEARCH_API_KEY'
    if provider == 'tavily':
        return 'TAVILY_API_KEY'
    return None


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _search_searxng(
    provider: SearchProviderProfile,
    query: str,
    *,
    max_results: int,
    timeout_seconds: float,
) -> tuple[SearchResult, ...]:
    endpoint = provider.base_url.rstrip('/')
    if not endpoint.endswith('/search'):
        endpoint += '/search'
    url = endpoint + '?' + parse.urlencode(
        {
            'q': query,
            'format': 'json',
        }
    )
    req = request.Request(url, headers={'User-Agent': 'sigilcode/1.0'})
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode('utf-8', errors='replace'))
    except (HTTPError, URLError, TimeoutError) as e:
        raise LookupError(f'SearXNG search failed: {e}') from e
    except ValueError as e:
        raise LookupError(f'SearXNG search failed: the reply is not JSON ({e})') from e
    if not isinstance(payload, dict):
        raise LookupError('SearXNG search failed: the reply is not a JSON object')
    results = payload.get('results')
    if not isinstance(results, list):
        return ()
    rendered: list[SearchResult] = []
    for index, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = item.get('url')
        title = item.get('title')
        snippet = item.get('content') or item.get('snippet') or ''
        if not isinstance(url_value, str) or not url_value.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            title = url_value
        rendered.append(
            SearchResult(
                title=title.strip(),
                url=url_value.strip(),
                snippet=str(snippet).strip(),
                provider_name=provider.name,
                rank=index,
            )
        )
    return tuple(rendered)


def _search_brave(
    provider: SearchProviderProfile,
    query: str,
    *,
    max_results: int,
    timeout_seconds: float,
    usage_tracker: SearchUsageTracker | None = None,
) -> tuple[SearchResult, ...]:
    """Search using Brave API with usage tracking."""
    api_key = _require_api_key(provider)
    
    # Initialize usage tracker if not provided
    if usage_tracker is None:
        usage_tracker = SearchUsageTracker(state_path=default_usage_file())
    
    # Check Brave limit
    max_requests = _env_int('BRAVE_MAX_REQUESTS', DEFAULT_MAX_REQUESTS_PER_KEY)
    if usage_tracker.is_key_exhausted('BRAVE_SEARCH_API_KEY', max_requests):
        raise LookupError(
            f'Brave API key has reached limit of {max_requests} requests. '
            'Reset usage or add more API keys.'
        )
    
    url = provider.base_url + '?' + parse.urlencode({'q': query, 'count': max_results})
    req = request.Request(
        url,
        headers={
            'User-Agent': 'sigilcode/1.0',
            'X-Subscription-Token': api_key,
            'Accept': 'application/json',
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode('utf-8', errors='replace'))
    except (HTTPError, URLError, TimeoutError) as e:
        raise LookupError(f'Brave search failed: {e}') from e
    except ValueError as e:
        raise LookupError(f'Brave search failed: the reply is not JSON ({e})') from e
    if not isinstance(payload, dict) or not isinstance(payload.get('web', {}), dict):
        raise LookupError('Brave search failed: unexpected reply shape')
    
    # Track usage after successful request
    usage_tracker.increment_usage('BRAVE_SEARCH_API_KEY')
    
    results = payload.get('web', {}).get('results')
    if not isinstance(results, list):
        return ()
    rendered: list[SearchResult] = []
    for index, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = item.get('url')
        title = item.get('title')
        snippet = item.get('description') or ''
        if not isinstance(url_value, str) or not url_value.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            title = url_value
        rendered.append(
            SearchResult(
                title=title.strip(),
                url=url_value.strip(),
                snippet=str(snippet).strip(),
                provider_name=provider.name,
                rank=index,
            )
        )
    return tuple(rendered)


def _search_tavily(
    provider: SearchProviderProfile,
    query: str,
    *,
    max_results: int,
    domains: tuple[str, ...],
    timeout_seconds: float,
    usage_tracker: SearchUsageTracker | None = None,
    blocked_domains: tuple[str, ...] = (),
) -> tuple[SearchResult, ...]:
    """Search using Tavily API with automatic key rotation."""
    # Get max requests limit from env or use default
    max_requests = _env_int('TAVILY_MAX_REQUESTS', DEFAULT_MAX_REQUESTS_PER_KEY)
    
    # Initialize usage tracker if not provided
    if usage_tracker is None:
        usage_tracker = SearchUsageTracker(state_path=default_usage_file())
    
    # Try each Tavily key in order. A key the server rejects for auth/quota
    # reasons (its own quota may run out before our local count does) is
    # marked exhausted and the next key is tried, then Brave.
    body: dict[str, Any] | None = None
    last_error: str | None = None
    tried: set[str] = set()
    while True:
        # a provider whose key was withheld (_key_stays_home) gets none of the rotation keys either
        rotation = TAVILY_KEY_ENVS if provider.api_key_env else ()
        available_key_env = usage_tracker.get_available_key(
            tuple(k for k in rotation if k not in tried), max_requests
        )
        if available_key_env is None:
            break
        tried.add(available_key_env)
        api_key = os.environ.get(available_key_env, '').strip()
        payload: dict[str, Any] = {
            'api_key': api_key,
            'query': query,
            'max_results': max_results,
        }
        if domains:
            payload['include_domains'] = list(domains)
        if blocked_domains:
            payload['exclude_domains'] = list(blocked_domains)
        req = request.Request(
            provider.base_url,
            data=json.dumps(payload, ensure_ascii=True).encode('utf-8'),
            headers={
                'User-Agent': 'sigilcode/1.0',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode('utf-8', errors='replace'))
        except HTTPError as e:
            last_error = f'Tavily search failed ({available_key_env}): {e}'
            if e.code in _TAVILY_KEY_FAILOVER_CODES:
                _logger.warning('%s; trying next key', last_error)
                usage_tracker.mark_exhausted(available_key_env, max_requests)
                continue
            raise LookupError(last_error) from e
        except (URLError, TimeoutError) as e:
            raise LookupError(f'Tavily search failed: {e}') from e
        except ValueError as e:
            raise LookupError(f'Tavily search failed: the reply is not JSON ({e})') from e
        if not isinstance(body, dict):
            raise LookupError('Tavily search failed: the reply is not a JSON object')
        usage_tracker.increment_usage(available_key_env)
        break

    if body is None:
        # All Tavily keys exhausted or rejected: fall back to Brave.
        brave_key = os.environ.get('BRAVE_SEARCH_API_KEY')
        if isinstance(brave_key, str) and brave_key.strip():
            brave_max = _env_int('BRAVE_MAX_REQUESTS', DEFAULT_MAX_REQUESTS_PER_KEY)
            if not usage_tracker.is_key_exhausted('BRAVE_SEARCH_API_KEY', brave_max):
                brave_provider = SearchProviderProfile(
                    name='brave-fallback',
                    provider='brave',
                    source_manifest='fallback',
                    base_url=DEFAULT_BRAVE_BASE_URL,
                    api_key_env='BRAVE_SEARCH_API_KEY',
                )
                return _search_brave(
                    brave_provider,
                    query,
                    max_results=max_results,
                    timeout_seconds=timeout_seconds,
                    usage_tracker=usage_tracker,
                )
        raise LookupError(
            (last_error + '. ' if last_error else '')
            + 'All Tavily API keys exhausted and no Brave fallback available. '
            'Set TAVILY_API_KEY, TAVILY_API_KEY_2, or BRAVE_SEARCH_API_KEY.'
        )

    # Parse results
    results = body.get('results')
    if not isinstance(results, list):
        return ()
    rendered: list[SearchResult] = []
    for index, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url_value = item.get('url')
        title = item.get('title')
        snippet = item.get('content') or ''
        if not isinstance(url_value, str) or not url_value.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            title = url_value
        rendered.append(
            SearchResult(
                title=title.strip(),
                url=url_value.strip(),
                snippet=str(snippet).strip(),
                provider_name=provider.name,
                rank=index,
            )
        )
    return tuple(rendered)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError:
        _logger.warning('%s is not a number; using %s', name, default)
        return default


def _require_api_key(provider: SearchProviderProfile) -> str:
    if provider.api_key_env is None:
        raise LookupError(f'Search provider {provider.name} does not define an API key env var.')
    value = os.environ.get(provider.api_key_env)
    if not isinstance(value, str) or not value.strip():
        raise LookupError(
            f'Search provider {provider.name} requires env var {provider.api_key_env}.'
        )
    return value.strip()


def _matches_domains(url: str, domains: tuple[str, ...]) -> bool:
    hostname = parse.urlparse(url).hostname or ''
    hostname = hostname.lower()
    for domain in domains:
        normalized = domain.strip().lower()
        if not normalized:
            continue
        if hostname == normalized or hostname.endswith('.' + normalized):
            return True
    return False
