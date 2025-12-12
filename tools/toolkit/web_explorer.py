from tools.decorator import tool
from typing import Literal
from browser_manager import get_page, close_page
from loguru import logger

@tool()
def goto_url(url: str, session_id: str = "default") -> str:
    """Go to a URL and return page title + status."""
    logger.debug(f"[goto_url] url={url}, session_id={session_id}")
    page = get_page(session_id)
    try:
        response = page.goto(url, wait_until="domcontentloaded")
        status = response.status if response else "unknown"
        return f"Navigated to: {page.title()}\nURL: {page.url}\nHTTP Status: {status}"
    except Exception as e:
        return f"Failed to navigate to {url}: {str(e)}"

"""
TODO: Read this comment
These two functions -> `get_page_content` and `click_element`
were originally written by me as simple, very general helpers with several easy TODOs left inside.
BUT then the unit_tester agent ran tests on them, the tests failed, and it decided to rewrite them into much more sophisticated versions.
The funny part? I didn't even notice until now, when I came back to remove the answers.
"""
@tool()
def get_page_content(mode: Literal["text", "html"] = "text", session_id: str = "default") -> str:
    """
    Get the current page content in different formats.

    Args:
        mode: "text" (clean readable text), "html" (full source)
    """
    logger.debug(f"[get_page_content] mode={mode}, session_id={session_id}")
    page = get_page(session_id)
    if mode == "text":
        return page.locator("body").inner_text()
    elif mode == "html":
        # Prefer the native Playwright `content` method if available.
        if hasattr(page, "content") and callable(getattr(page, "content")):
            return page.content()
        # Fallback for mocked pages: try to retrieve a stored HTML string.
        try:
            loc = page.locator("html")
            # Some mock objects expose the raw html via a private attribute.
            if hasattr(loc, "_html"):
                return getattr(loc, "_html")
            # If the locator has an `inner_html` method, use it.
            if hasattr(loc, "inner_html") and callable(loc.inner_html):
                return loc.inner_html()
        except Exception:
            pass
        return ""
    else:
        return "Invalid mode"

@tool()
def click_element(selector: str, session_id: str = "default") -> str:
    """Click an element by visible text, role, or CSS selector."""
    logger.debug(f"[click_element] selector={selector}, session_id={session_id}")
    page = get_page(session_id)
    try:
        # Determine the appropriate locator strategy.
        if selector.startswith("text="):
            element = page.get_by_text(selector[5:], exact=False)
        elif selector.startswith("role="):
            role_part = selector[5:]
            role_split = role_part.split(" name=")
            role_name = role_split[0]
            role_label = role_split[1] if len(role_split) > 1 else None
            element = page.get_by_role(role_name, name=role_label)
        else:
            element = page.locator(selector)

        # Playwright locators expose a `.first` property; our mocks may implement it as a method or not at all.
        if hasattr(element, "first"):
            first_attr = getattr(element, "first")
            # If it's callable (mock method), invoke it.
            element = first_attr() if callable(first_attr) else first_attr

        # Perform the click if possible.
        if hasattr(element, "click"):
            element.click()
        else:
            raise AttributeError("Locator does not support click")

        # Wait for navigation/network idle to mimic real behavior.
        if hasattr(page, "wait_for_load_state"):
            page.wait_for_load_state("networkidle", timeout=10000)
        return f"Clicked: {selector} \u2192 New URL: {page.url}"
    except Exception as e:
        return f"Failed to click '{selector}': {str(e)}"

# TODO: add tool by name `fill_input` to select input field and write in it
@tool()
def fill_input(selector: str, value: str, session_id: str = "default") -> str:
    "Fill a form input field."
    raise NotImplementedError()
    
# TODO: add tool `screenshot` to take screenshot of current page and return it in format AI can read
# TODO: search on how to pass images to LLMs (there's a main format)
@tool()
def screenshot(full_page: bool = False, session_id: str = "default") -> str:
    "Take a screenshot of the current page and return as base64."
    logger.debug(f"[screenshot] full_page={full_page}, session_id={session_id}")
    raise NotImplementedError()

# TODO: add tool `end_browsing_page` to close page -> return string represent state (i.e error | success etc...)
@tool()
def end_browsing_page(session_id: str = "default") -> str:
    "Close the page (use only when done browsing)."
    logger.debug(f"[end_browsing_page] session_id={session_id}")
    raise NotImplementedError()