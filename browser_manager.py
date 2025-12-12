from playwright.sync_api import sync_playwright, Browser, Page, Playwright

class BrowserManager:
    """Manages browser lifecycle properly with context manager support."""

    def __init__(self):
        self._playwright: Playwright = None
        self._browser: Browser = None
        self._pages: dict[str, Page] = {}  # key = agent/session name

    def start(self):
        """Initialize browser if not already running."""
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        return self._browser
    
    def get_page(self, session_id: str) -> Page:
        """Get or create a page for a given session/agent."""
        self.start()
        if session_id not in self._pages or self._pages[session_id].is_closed():
            self._pages[session_id] = self._browser.new_page()
            self._pages[session_id].set_default_timeout(15000)
        return self._pages[session_id]
    
    def close_page(self, session_id: str):
        """Close a single page."""
        page = self._pages.get(session_id)
        if page and not page.is_closed():
            page.close()
            del self._pages[session_id]

    def close_browser(self):
        """Close all pages and the browser."""
        for page in self._pages.values():
            if not page.is_closed():
                page.close()
        self._pages.clear()
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
# Singleton  class
__browser_manager = BrowserManager()

def get_page(session_id: str) -> Page:
    return __browser_manager.get_page(session_id)



def close_page(session_id: str) -> Page:
    return __browser_manager.close_page(session_id)