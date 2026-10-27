"""Save home and results screenshots of the Streamlit UI.

Starts a mock API that serves ``tests/fixtures/search_response.json``, runs
Streamlit against that API, and writes PNGs under ``artifacts/ui/``.

The fixture is a search response plus an ``answer`` object. ``POST /api/search``
returns the search fields. ``POST /api/answer`` returns the answer object.
Screenshots are taken at 1280px and 390px wide.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "search_response.json"
OUTPUT_DIR = ROOT / "artifacts" / "ui"
APP_PATH = ROOT / "streamlit_app.py"
WIDTHS = (1280, 390)
VIEWPORT_HEIGHT = 900
_SEARCH_KEYS = ("context_chunks", "facets", "query")
_ANSWER_KEYS = ("answer", "summary", "points", "citations")
# Streamlit scrolls inside the main pane. This lets the PNG include that pane.
_REVEAL_CSS = """
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}
"""


def load_fixture(path: Path) -> dict:
    """Read the search fixture. It must include an answer for the results page."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    missing = [key for key in (*_SEARCH_KEYS, "answer") if key not in data]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return data


def search_body(fixture: dict, request: dict) -> dict:
    """Search fields from the fixture. A posted query replaces the stored one."""
    body = {key: fixture[key] for key in _SEARCH_KEYS}
    posted = str(request.get("query") or "").strip()
    if posted:
        body["query"] = posted
    return body


def answer_body(fixture: dict) -> dict:
    """Answer fields the results page renders."""
    answer = fixture.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("search fixture is missing an answer object")
    return {key: answer.get(key, [] if key in ("points", "citations") else "") for key in _ANSWER_KEYS}


class _FixtureHandler(BaseHTTPRequestHandler):
    """Serves the fixture on the search and answer routes. Other GETs are 404."""

    fixture: dict = {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/", "/api/health"}:
            self._send(200, {"status": "healthy"})
            return
        self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            request = {}
        if not isinstance(request, dict):
            request = {}
        path = urlparse(self.path).path.rstrip("/")
        if path == "/api/search":
            self._send(200, search_body(self.fixture, request))
            return
        if path == "/api/answer":
            self._send(200, answer_body(self.fixture))
            return
        self._send(404, {"detail": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_api(fixture: dict) -> ThreadingHTTPServer:
    _FixtureHandler.fixture = fixture
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(url: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _start_streamlit(ui_port: int, api_port: int, log_file) -> subprocess.Popen:
    env = os.environ.copy()
    env["API_BASE_URL"] = f"http://127.0.0.1:{api_port}"
    env["ARXIV_HTML_BASE_URL"] = f"http://127.0.0.1:{api_port}/html"
    env["UI_FIGURE_TIMEOUT_S"] = "1"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(ui_port),
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def _reveal_full_page(page, width: int) -> None:
    page.add_style_tag(content=_REVEAL_CSS)
    height = page.evaluate(
        """() => Math.ceil(Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight,
            (document.querySelector('[data-testid="stAppViewContainer"]') || document.body).scrollHeight
        ))"""
    )
    page.set_viewport_size({"width": width, "height": max(int(height), VIEWPORT_HEIGHT)})
    page.wait_for_timeout(200)


def _wait_until_settled(page) -> None:
    page.wait_for_function(
        """() => {
            const text = document.body.innerText || "";
            const busy = text.includes("Retrieving papers")
                || text.includes("Fetching figures")
                || text.includes("Writing the summary");
            if (busy) return false;
            const widget = document.querySelector('[data-testid="stStatusWidget"]');
            if (widget && /running/i.test(widget.innerText || "")) return false;
            return true;
        }""",
        timeout=30_000,
    )
    page.wait_for_timeout(400)


def _shoot(page, width: int, path: Path) -> None:
    _reveal_full_page(page, width)
    page.screenshot(path=str(path), full_page=True)
    print(f"Wrote {path}")


def _capture(base_url: str, query: str, output_dir: Path) -> None:
    from playwright.sync_api import sync_playwright

    results_url = f"{base_url}/?query={quote(query)}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for width in WIDTHS:
                context = browser.new_context(
                    viewport={"width": width, "height": VIEWPORT_HEIGHT},
                    device_scale_factor=1,
                    color_scheme="dark",
                )
                page = context.new_page()
                page.set_default_timeout(30_000)
                page.goto(base_url, wait_until="domcontentloaded")
                page.get_by_text("Not sure where to start? Try one of these:").wait_for()
                _wait_until_settled(page)
                _shoot(page, width, output_dir / f"home-{width}.png")
                page.set_viewport_size({"width": width, "height": VIEWPORT_HEIGHT})
                page.goto(results_url, wait_until="domcontentloaded")
                page.locator("article.paper-card").first.wait_for()
                page.get_by_text("AI summary of these papers").wait_for()
                _wait_until_settled(page)
                _shoot(page, width, output_dir / f"results-{width}.png")
                context.close()
        finally:
            browser.close()


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_fixture(FIXTURE_PATH)
    server = _start_api(fixture)
    api_port = int(server.server_address[1])
    ui_port = _free_port()
    log_path = output_dir / "streamlit.log"
    log_file = log_path.open("w", encoding="utf-8")
    ui_proc = None
    try:
        ui_proc = _start_streamlit(ui_port, api_port, log_file)
        _wait_for(f"http://127.0.0.1:{ui_port}/_stcore/health")
        _capture(f"http://127.0.0.1:{ui_port}", str(fixture["query"]), output_dir)
    except Exception:
        log_file.flush()
        if log_path.is_file():
            print(log_path.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        raise
    finally:
        _stop_process(ui_proc)
        log_file.close()
        log_path.unlink(missing_ok=True)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
