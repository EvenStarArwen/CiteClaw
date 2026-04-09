"""Persistent Chrome profile manager via Playwright.

The whole point of pdfclaw is to drive a real Chrome instance that has
your institutional SSO cookies in it — so this module is the bit that
boots that browser. Two operations:

  * ``ensure_profile_dir(path)`` — make sure the profile directory exists.
    Doesn't launch anything; safe to call from anywhere.

  * ``launch_for_login(profile_path)`` — open a non-headless Chromium
    pinned to the persistent profile, navigate to a list of "please log
    in" URLs, and block until the user closes the window. Used by the
    ``login`` subcommand.

  * ``open_browser_context(profile_path, *, headless, downloads_dir)`` —
    context manager that yields a Playwright ``Page`` plugged into the
    persistent profile, ready for recipes to use. Used by the ``fetch``
    subcommand.

Why a dedicated profile and not your daily Chrome:
  * Chrome 136+ blocks remote debugging on the default profile for
    security reasons (you literally can't ``--remote-debugging-port`` it).
  * Even if you could, mixing Playwright control into your daily browsing
    is asking for trouble (extensions, ad-blockers, autofill).
  * A dedicated profile is **persistent**: log in once, the cookies stay.

The Chromium binary used here is the one Playwright downloads via
``playwright install chromium``, NOT your system Chrome. They're the
same engine, but the Playwright build doesn't depend on your system
Chrome version.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import BrowserContext, Page

log = logging.getLogger("pdfclaw.browser")

# URLs we open during the ``login`` subcommand. The user is expected to
# click "Access through your institution" / "Sign in via Shibboleth" /
# "OpenAthens" on each, complete the SSO dance, and then close the
# window. The cookies persist in the profile directory.
LOGIN_URLS = [
    "https://www.nature.com/",
    "https://www.sciencedirect.com/",
    "https://www.science.org/",
    "https://www.cell.com/",
]


def ensure_profile_dir(path: Path) -> Path:
    """Create the profile directory if it doesn't exist."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def launch_for_login(profile_path: Path) -> None:
    """Open a non-headless Chromium pinned to the profile, wait for close.

    The user logs into their institution interactively. We don't need
    to know what they did — Playwright's persistent context will save
    every cookie and storage entry to ``profile_path`` automatically.
    Closing the window cleans up the Playwright context and exits.
    """
    profile_path = ensure_profile_dir(profile_path)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright not installed. Run:\n"
            "  pip install 'playwright>=1.40,<2'\n"
            "  python -m playwright install chromium"
        ) from exc

    print(
        f"\n[pdfclaw] Launching Chromium with persistent profile:\n"
        f"  {profile_path}\n"
        f"\n[pdfclaw] I'll open these tabs:\n"
        + "\n".join(f"  - {u}" for u in LOGIN_URLS)
        + "\n\n[pdfclaw] Please:\n"
        f"  1. Click 'Access through your institution' on each site\n"
        f"  2. Pick your university (e.g. University of Exeter)\n"
        f"  3. Complete the SSO flow\n"
        f"  4. Verify you can see a 'Download PDF' button on a paywalled article\n"
        f"  5. Close the Chromium window when done\n"
        f"\n[pdfclaw] Cookies will persist in the profile directory automatically.\n"
    )

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_path),
            headless=False,
            viewport={"width": 1280, "height": 900},
            accept_downloads=True,
        )
        # Open one tab per login URL
        for url in LOGIN_URLS:
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to open %s: %s", url, exc)

        print(
            "[pdfclaw] When you've finished logging in, come back to this terminal\n"
            "[pdfclaw] and press Enter. (Closing the browser window first is fine.)\n"
        )
        try:
            input("[pdfclaw] >>> Press Enter to close the browser context: ")
        except (EOFError, KeyboardInterrupt):
            pass
        with contextlib.suppress(Exception):
            context.close()

    print("\n[pdfclaw] Browser closed. Profile saved. You can now run:")
    print("  python -m pdfclaw fetch <checkpoint_dir>\n")


@contextlib.contextmanager
def open_browser_context(
    profile_path: Path,
    *,
    headless: bool = False,
    downloads_dir: Path | None = None,
) -> "Iterator[tuple[BrowserContext, Page]]":
    """Yield ``(context, page)`` for a persistent Chromium session.

    Caller is responsible for using the page (recipes do `page.goto`,
    `page.click`, etc.) and the context will be cleaned up on exit.
    """
    profile_path = ensure_profile_dir(profile_path)
    if downloads_dir is not None:
        Path(downloads_dir).mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright not installed. Run:\n"
            "  pip install 'playwright>=1.40,<2'\n"
            "  python -m playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_path),
            headless=headless,
            viewport={"width": 1280, "height": 900},
            accept_downloads=True,
        )
        page = context.new_page()
        try:
            yield context, page
        finally:
            with contextlib.suppress(Exception):
                context.close()
