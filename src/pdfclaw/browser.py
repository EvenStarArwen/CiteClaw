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

# URLs we open during the ``login`` subcommand. We use ARTICLE pages
# (not homepages) so the "Access through your institution" link is
# visible. The user clicks that link, picks their university, completes
# SSO, and the cookies persist in the profile directory.
#
# Homepage login (e.g. nature.com "Log in" button) takes you to a
# PERSONAL account page, not the institutional Shibboleth flow.
LOGIN_URLS = [
    # Nature — a PAYWALLED article (NOT OA); look for "Access through your institution"
    "https://www.nature.com/articles/nbt.3988",
    # ScienceDirect — institutional login portal (already confirmed working)
    "https://www.sciencedirect.com/user/institution/login",
    # Science.org — a paywalled article
    "https://www.science.org/doi/10.1126/science.ade2574",
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
        # Use the SYSTEM Chrome (channel="chrome") instead of
        # Playwright's bundled Chromium. Real Chrome doesn't have the
        # --enable-automation flag and passes Cloudflare Turnstile
        # challenges that the bundled Chromium can't.
        # Key trick: suppress --enable-automation and AutomationControlled
        # so Cloudflare Turnstile doesn't loop. Without this, the
        # "Chrome is being controlled by automated test software" banner
        # appears and CF rejects every challenge attempt.
        stealth_kwargs: dict = {
            "user_data_dir": str(profile_path),
            "headless": False,
            "viewport": {"width": 1280, "height": 900},
            "accept_downloads": True,
            "ignore_default_args": ["--enable-automation"],
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            context = p.chromium.launch_persistent_context(
                channel="chrome", **stealth_kwargs,
            )
        except Exception:  # noqa: BLE001
            log.warning(
                "System Chrome not found; falling back to Playwright Chromium. "
                "Cloudflare-protected sites (Cell, Science) may show infinite "
                "captcha loops. Install Google Chrome to fix this."
            )
            context = p.chromium.launch_persistent_context(**stealth_kwargs)
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
        # Prefer system Chrome for the same reason as launch_for_login:
        # Cloudflare/Akamai detect bundled Chromium's automation flags.
        launch_kwargs: dict = {
            "user_data_dir": str(profile_path),
            "headless": headless,
            "viewport": {"width": 1280, "height": 900},
            "accept_downloads": True,
            "ignore_default_args": ["--enable-automation"],
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if downloads_dir is not None:
            launch_kwargs["downloads_path"] = str(downloads_dir)
        try:
            context = p.chromium.launch_persistent_context(
                channel="chrome", **launch_kwargs,
            )
        except Exception:  # noqa: BLE001
            context = p.chromium.launch_persistent_context(**launch_kwargs)
        page = context.new_page()

        # Force PDF downloads instead of opening in Chrome's PDF viewer.
        # Without this, clicking "Download PDF" on Nature/Science/etc.
        # opens the PDF in a new tab/inline viewer and Playwright's
        # expect_download never fires — the root cause of "browser opens
        # the page with Exeter access but still reports FAIL".
        def _force_pdf_download(route):
            resp = route.fetch()
            headers = dict(resp.headers)
            headers["content-disposition"] = "attachment"
            route.fulfill(response=resp, headers=headers)

        page.route("**/*.pdf", _force_pdf_download)
        page.route("**/*.pdf?*", _force_pdf_download)
        page.route("**/pdf/**", _force_pdf_download)
        page.route("**/*pdf*download*", _force_pdf_download)

        try:
            yield context, page
        finally:
            with contextlib.suppress(Exception):
                context.close()
