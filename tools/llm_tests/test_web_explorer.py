import pytest
from unittest.mock import MagicMock, patch
import builtins

# Patch get_page and close_page at the import level
goto_url_path = "tools.toolkit.web_explorer.get_page"
close_page_path = "tools.toolkit.web_explorer.close_page"

@pytest.fixture
def mock_page():
    mock = MagicMock()
    mock.title.return_value = "Test Page Title"
    mock.url = "http://example.com"
    mock.goto.return_value = MagicMock(status=200)
    mock.locator.return_value = MagicMock(
        inner_text=MagicMock(return_value="Hello Text Body"),
        inner_html=MagicMock(return_value="<p>hi</p>"),
        _html="<html></html>"
    )
    mock.content.return_value = "<html><body>Test</body></html>"
    mock.screenshot.return_value = b"imagedata123"
    mock.get_by_text.return_value = MagicMock(click=MagicMock(), first=MagicMock(return_value=MagicMock(click=MagicMock())),)
    mock.get_by_role.return_value = MagicMock(click=MagicMock(), first=MagicMock(return_value=MagicMock(click=MagicMock())),)
    mock.wait_for_load_state.return_value = None
    return mock

@patch(goto_url_path)
@patch(close_page_path)
def test_goto_url(mock_close_page, mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    mock_get_page.return_value = mock_page
    out = web_explorer.goto_url("http://example.com")
    assert "Test Page Title" in out
    assert "Navigated" in out

@patch(goto_url_path)
def test_get_page_content_text(mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    mock_get_page.return_value = mock_page
    result = web_explorer.get_page_content(mode="text")
    assert "Hello Text Body" == result

@patch(goto_url_path)
def test_get_page_content_html_content_method(mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    mock_get_page.return_value = mock_page
    out = web_explorer.get_page_content(mode="html")
    assert "Test" in out

@patch(goto_url_path)
def test_click_element_text(mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    # Selector with text=
    mock_get_page.return_value = mock_page
    out = web_explorer.click_element("text=MyText")
    assert "Clicked" in out
    # role selector
    out = web_explorer.click_element("role=button name=Submit")
    assert "Clicked" in out
    # css selector
    out = web_explorer.click_element(".myclass")
    assert "Clicked" in out

@patch(goto_url_path)
def test_fill_input(mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    mock_get_page.return_value = mock_page
    out = web_explorer.fill_input("#input", "testvalue")
    assert "Filled" in out

@patch(goto_url_path)
def test_screenshot(mock_get_page, mock_page):
    from tools.toolkit import web_explorer
    mock_get_page.return_value = mock_page
    out = web_explorer.screenshot(full_page=True)
    assert out.startswith("data:image/png;base64,")

@patch(close_page_path)
def test_end_browsing_page(mock_close_page):
    from tools.toolkit import web_explorer
    mock_close_page.return_value = None
    out = web_explorer.end_browsing_page()
    assert "Page closed" in out

@patch(close_page_path)
def test_end_browsing_page_error(mock_close_page):
    from tools.toolkit import web_explorer
    mock_close_page.side_effect = RuntimeError("can't close")
    out = web_explorer.end_browsing_page()
    assert "Couldn't close page" in out
