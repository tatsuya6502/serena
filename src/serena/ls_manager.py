import logging
import threading
import time
from collections.abc import Callable, Iterator

from sensai.util.logging import LogTime

from serena.constants import SERENA_MANAGED_DIR_IN_HOME, SERENA_MANAGED_DIR_NAME
from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)


class LanguageServerFactory:
    def __init__(
        self,
        project_root: str,
        encoding: str,
        ignored_patterns: list[str],
        ls_timeout: float | None = None,
        ls_specific_settings: dict | None = None,
        log_level: int = logging.INFO,
        trace_lsp_communication: bool = False,
    ):
        self.project_root = project_root
        self.encoding = encoding
        self.ignored_patterns = ignored_patterns
        self.ls_timeout = ls_timeout
        self.ls_specific_settings = ls_specific_settings
        self.log_level = log_level
        self.trace_lsp_communication = trace_lsp_communication

    def create_language_server(self, language: Language) -> SolidLanguageServer:
        ls_config = LanguageServerConfig(
            code_language=language,
            ignored_paths=self.ignored_patterns,
            trace_lsp_communication=self.trace_lsp_communication,
            encoding=self.encoding,
        )
        ls_logger = LanguageServerLogger(log_level=self.log_level)

        log.info(f"Creating language server instance for {self.project_root}, language={language}.")
        return SolidLanguageServer.create(
            ls_config,
            ls_logger,
            self.project_root,
            timeout=self.ls_timeout,
            solidlsp_settings=SolidLSPSettings(
                solidlsp_dir=SERENA_MANAGED_DIR_IN_HOME,
                project_data_relative_path=SERENA_MANAGED_DIR_NAME,
                ls_specific_settings=self.ls_specific_settings or {},
            ),
        )


class LanguageServerManager:
    """
    Manages one or more language servers for a project.
    Supports lazy initialization and automatic shutdown of idle servers.
    """

    def __init__(
        self,
        language_servers: dict[Language, SolidLanguageServer],
        language_server_factory: LanguageServerFactory | None = None,
        lazy_languages: set[Language] | None = None,
        idle_timeout: int = 0,
    ) -> None:
        """
        :param language_servers: a mapping from language to language server; the servers are assumed to be already started.
            The first server in the iteration order is used as the default server.
            All servers are assumed to serve the same project root.
        :param language_server_factory: factory for language server creation; if None, dynamic (re)creation of language servers
            is not supported
        :param lazy_languages: set of languages that should be initialized lazily (not started until first use)
        :param idle_timeout: seconds of inactivity before stopping lazy language servers (0 = disabled)
        """
        self._language_servers = language_servers
        self._language_server_factory = language_server_factory
        # Default language server might not exist if all are lazy
        self._default_language_server = next(iter(language_servers.values())) if language_servers else None
        self._root_path = self._default_language_server.repository_root_path if self._default_language_server else None
        self._lazy_languages = lazy_languages or set()
        self._lazy_language_configs: dict[Language, None] = {}  # Placeholder for lazy languages not yet started
        self._ls_lock = threading.Lock()  # Protects lazy language server creation

        # Initialize idle monitor
        self._idle_monitor = IdleMonitor(idle_timeout=idle_timeout)
        if idle_timeout > 0 and self._lazy_languages:
            self._idle_monitor.start(shutdown_callback=self._stop_idle_language_server)

    @staticmethod
    def from_languages(
        languages: list[Language],
        factory: LanguageServerFactory,
        lazy_languages: set[Language] | None = None,
        idle_timeout: int = 0,
    ) -> "LanguageServerManager":
        """
        Creates a manager with language servers for the given languages using the given factory.
        Non-lazy language servers are started in parallel threads. Lazy language servers are registered
        but not started until first use.

        :param languages: the languages for which to spawn language servers
        :param factory: the factory for language server creation
        :param lazy_languages: set of languages that should be initialized lazily
        :param idle_timeout: seconds of inactivity before stopping lazy language servers (0 = disabled)
        :return: the instance
        """
        lazy_languages = lazy_languages or set()
        eager_languages = [lang for lang in languages if lang not in lazy_languages]

        language_servers: dict[Language, SolidLanguageServer] = {}
        threads = []
        exceptions = {}
        lock = threading.Lock()

        def start_language_server(language: Language) -> None:
            try:
                with LogTime(f"Language server startup (language={language.value})"):
                    language_server = factory.create_language_server(language)
                    language_server.start()
                    if not language_server.is_running():
                        raise RuntimeError(f"Failed to start the language server for language {language.value}")
                    with lock:
                        language_servers[language] = language_server
            except Exception as e:
                log.error(f"Error starting language server for language {language.value}: {e}", exc_info=e)
                with lock:
                    exceptions[language] = e

        # Start only eager (non-lazy) language servers in parallel threads
        for language in eager_languages:
            thread = threading.Thread(target=start_language_server, args=(language,), name="StartLS:" + language.value)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        # If any server failed to start up, raise an exception and stop all started language servers
        if exceptions:
            for ls in language_servers.values():
                ls.stop()
            failure_messages = "\n".join([f"{lang.value}: {e}" for lang, e in exceptions.items()])
            raise Exception(f"Failed to start language servers:\n{failure_messages}")

        # Log lazy languages
        if lazy_languages:
            lazy_lang_names = ", ".join([lang.value for lang in lazy_languages])
            log.info(f"Lazy language servers configured (will start on first use): {lazy_lang_names}")

        manager = LanguageServerManager(language_servers, factory, lazy_languages=lazy_languages, idle_timeout=idle_timeout)

        # Register lazy languages with idle monitor
        for language in lazy_languages:
            manager._lazy_language_configs[language] = None  # Mark as lazy but not yet started

        return manager

    def get_root_path(self) -> str:
        if self._root_path is None:
            # Try to get from factory if available
            if self._language_server_factory:
                return self._language_server_factory.project_root
            raise RuntimeError("No language servers started yet, cannot determine root path")
        return self._root_path

    def _ensure_functional_ls(self, ls: SolidLanguageServer) -> SolidLanguageServer:
        if not ls.is_running():
            log.warning(f"Language server for language {ls.language} is not running; restarting ...")
            ls = self.restart_language_server(ls.language)
        return ls

    def _start_lazy_language_server(self, language: Language) -> SolidLanguageServer:
        """Start a lazy language server that hasn't been started yet."""
        with self._ls_lock:
            # Check again inside lock in case another thread started it
            if language in self._language_servers:
                return self._language_servers[language]

            if language not in self._lazy_language_configs:
                raise ValueError(f"Language {language.value} is not configured as a lazy language")

            log.info(f"Starting lazy language server for {language.value}...")
            language_server = self._create_and_start_language_server(language)

            # Set as default if no default exists yet
            if self._default_language_server is None:
                self._default_language_server = language_server
                self._root_path = language_server.repository_root_path

            # Register with idle monitor
            if self._idle_monitor.idle_timeout > 0:
                self._idle_monitor.register_lazy_language(language)

            # Remove from lazy configs since it's now started
            del self._lazy_language_configs[language]

            return language_server

    def get_language_server(self, relative_path: str) -> SolidLanguageServer:
        # First, try to find an already-running language server
        ls: SolidLanguageServer | None = None
        if len(self._language_servers) > 1:
            for candidate in self._language_servers.values():
                if not candidate.is_ignored_path(relative_path, ignore_unsupported_files=True):
                    ls = candidate
                    break

        # If no running server found, check if there's a lazy language server that should be started
        if ls is None:
            # Check if any lazy language could handle this file
            for language in self._lazy_language_configs.keys():
                # Create a temporary config to check if this language would handle the file
                if self._language_server_factory:
                    temp_ls = self._language_server_factory.create_language_server(language)
                    try:
                        if not temp_ls.is_ignored_path(relative_path, ignore_unsupported_files=True):
                            # This lazy language should handle this file - start it
                            return self._start_lazy_language_server(language)
                    finally:
                        # Don't start the temp server, just clean up
                        pass

        if ls is None:
            if self._default_language_server is None:
                # All languages are lazy and none are started yet
                # Start the first lazy language as a fallback
                if self._lazy_language_configs:
                    first_lazy_lang = next(iter(self._lazy_language_configs.keys()))
                    ls = self._start_lazy_language_server(first_lazy_lang)
                else:
                    raise RuntimeError("No language servers available (all are lazy and none have been started)")
            else:
                ls = self._default_language_server

        # Update idle monitor timestamp for lazy languages
        if ls.language in self._lazy_languages:
            self._idle_monitor.update_last_used(ls.language)

        return self._ensure_functional_ls(ls)

    def _create_and_start_language_server(self, language: Language) -> SolidLanguageServer:
        if self._language_server_factory is None:
            raise ValueError(f"No language server factory available to create language server for {language}")
        language_server = self._language_server_factory.create_language_server(language)
        language_server.start()
        self._language_servers[language] = language_server
        return language_server

    def restart_language_server(self, language: Language) -> SolidLanguageServer:
        """
        Forces recreation and restart of the language server for the given language.
        It is assumed that the language server for the given language is no longer running.

        :param language: the language
        :return: the newly created language server
        """
        if language not in self._language_servers:
            raise ValueError(f"No language server for language {language.value} present; cannot restart")
        return self._create_and_start_language_server(language)

    def add_language_server(self, language: Language) -> SolidLanguageServer:
        """
        Dynamically adds a new language server for the given language.

        :param language: the language
        :param factory: the factory to create the language server
        :return: the newly created language server
        """
        if language in self._language_servers:
            raise ValueError(f"Language server for language {language.value} already present")
        return self._create_and_start_language_server(language)

    def remove_language_server(self, language: Language, save_cache: bool = False) -> None:
        """
        Removes the language server for the given language, stopping it if it is running.

        :param language: the language
        """
        if language not in self._language_servers:
            raise ValueError(f"No language server for language {language.value} present; cannot remove")
        ls = self._language_servers.pop(language)
        self._stop_language_server(ls, save_cache=save_cache)
        # Remove from idle monitor if it's a lazy language
        if language in self._lazy_languages:
            self._idle_monitor.unregister_language(language)

    def _stop_idle_language_server(self, language: Language) -> None:
        """
        Stop a language server that has been idle. Called by IdleMonitor.
        Saves cache before stopping.

        :param language: the language
        """
        if language in self._language_servers:
            ls = self._language_servers.pop(language)
            self._stop_language_server(ls, save_cache=True)
            # Re-add to lazy configs so it can be restarted later
            self._lazy_language_configs[language] = None
            log.info(f"Idle language server for {language.value} stopped and marked for lazy restart")

    @staticmethod
    def _stop_language_server(ls: SolidLanguageServer, save_cache: bool = False) -> None:
        if ls.is_running():
            if save_cache:
                ls.save_cache()
            log.info(f"Stopping language server for language {ls.language} ...")
            ls.stop()

    def iter_language_servers(self) -> Iterator[SolidLanguageServer]:
        for ls in self._language_servers.values():
            yield self._ensure_functional_ls(ls)

    def stop_all(self, save_cache: bool = False) -> None:
        """
        Stops all managed language servers and the idle monitor.

        :param save_cache: whether to save the cache before stopping
        """
        # Stop idle monitor first
        self._idle_monitor.stop()

        # Stop all language servers
        for ls in self.iter_language_servers():
            self._stop_language_server(ls, save_cache=save_cache)

    def save_all_caches(self) -> None:
        """
        Saves the caches of all managed language servers.
        """
        for ls in self.iter_language_servers():
            if ls.is_running():
                ls.save_cache()


class IdleMonitor:
    """
    Monitors language servers for idle time and stops them after a configured timeout.
    Runs in a background thread and only stops language servers that are marked as lazy.
    """

    def __init__(self, idle_timeout: int, check_interval: int = 60):
        """
        :param idle_timeout: Seconds of inactivity before stopping a language server. Set to 0 to disable.
        :param check_interval: How often to check for idle servers (in seconds)
        """
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self._last_used_times: dict[Language, float] = {}
        self._lazy_languages: set[Language] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._shutdown_callback: Callable[[Language], None] | None = None

    def start(self, shutdown_callback: Callable[[Language], None]) -> None:
        """
        Start the idle monitor thread.

        :param shutdown_callback: Function to call to stop a language server. Should take a Language parameter.
        """
        if self.idle_timeout <= 0:
            log.info("Idle timeout disabled (idle_timeout <= 0), not starting idle monitor")
            return

        if self._thread is not None:
            raise RuntimeError("IdleMonitor already started")

        self._shutdown_callback = shutdown_callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, name="LSIdleMonitor", daemon=True)
        self._thread.start()
        log.info(f"Started language server idle monitor (timeout={self.idle_timeout}s, check_interval={self.check_interval}s)")

    def stop(self) -> None:
        """Stop the idle monitor thread gracefully."""
        if self._thread is None:
            return

        log.info("Stopping language server idle monitor...")
        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            log.warning("Idle monitor thread did not stop within timeout")
        self._thread = None

    def register_lazy_language(self, language: Language) -> None:
        """Mark a language as lazy (subject to idle timeout)."""
        with self._lock:
            self._lazy_languages.add(language)
            self._last_used_times[language] = time.time()

    def unregister_language(self, language: Language) -> None:
        """Remove a language from monitoring (e.g., when explicitly stopped)."""
        with self._lock:
            self._lazy_languages.discard(language)
            self._last_used_times.pop(language, None)

    def update_last_used(self, language: Language) -> None:
        """Update the last used timestamp for a language server."""
        with self._lock:
            self._last_used_times[language] = time.time()

    def _monitor_loop(self) -> None:
        """Background thread that periodically checks for idle servers."""
        while not self._stop_event.is_set():
            try:
                self._check_and_stop_idle_servers()
            except Exception as e:
                log.exception(f"Error in idle monitor loop: {e}")

            # Wait for check_interval or until stop event is set
            self._stop_event.wait(timeout=self.check_interval)

    def _check_and_stop_idle_servers(self) -> None:
        """Check all lazy language servers and stop those that are idle."""
        current_time = time.time()
        languages_to_stop = []

        with self._lock:
            for language in self._lazy_languages:
                last_used = self._last_used_times.get(language)
                if last_used is None:
                    continue

                idle_time = current_time - last_used
                if idle_time >= self.idle_timeout:
                    languages_to_stop.append(language)

        # Stop servers outside the lock to avoid deadlock
        for language in languages_to_stop:
            try:
                log.info(f"Stopping idle language server for {language.value} (idle for {idle_time:.1f}s)")
                if self._shutdown_callback:
                    self._shutdown_callback(language)
                self.unregister_language(language)
            except Exception as e:
                log.exception(f"Error stopping idle language server for {language.value}: {e}")
