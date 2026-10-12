"""Exercise a running RepoLens through a browser; save desktop and mobile screenshots.

Install playwright, run `python -m playwright install chromium`, and start RepoLens first.
Use --live to also index the public project repository (requires network access).
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.get_by_text("Server connected", exact=True)).to_be_visible()
        page.get_by_role("button", name="Try the sample shop").click()
        expect(page.locator("#active-index strong")).to_have_text("Sample shop · demonstration")
        expect(page.get_by_role("button", name="Try the sample shop")).to_be_enabled()
        search = page.get_by_role("button", name="Find relevant code")
        expect(search).to_be_enabled(timeout=30000)
        search.click()
        expect(page.locator(".result-card").first).to_be_visible()
        expect(page.locator(".result-name").filter(has_text="inventory.py")).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role("button", name="Export JSON").click()
        download.value.save_as(output / "demo-search.json")
        result = json.loads((output / "demo-search.json").read_text())
        assert result["files"] and result["selected_characters"] <= result["context_budget_chars"]
        page.screenshot(path=str(output / "repolens-desktop.png"), full_page=True)
        page.get_by_role("button", name="Evaluation lab").click()
        expect(page.get_by_text("0.425", exact=True)).to_be_visible()
        assert page.locator("tbody tr").count() == 20
        page.screenshot(path=str(output / "repolens-evaluation.png"), full_page=True)
        page.get_by_role("button", name="Investigate", exact=True).click()
        if args.live:
            page.get_by_label("Public GitHub repository").fill("RyanBalech/llm-agent-evaluation")
            page.get_by_role("button", name="Index repo").click()
            expect(page.locator("#active-index strong")).to_have_text(
                "RyanBalech/llm-agent-evaluation", timeout=10000)
            expect(search).to_be_enabled(timeout=180000)
            page.get_by_label("What is going wrong?").fill(
                "select_under_character_budget skips chunks that exceed the context budget")
            search.click()
            expect(page.locator(".result-card").first).to_be_visible()
            assert page.locator(".snippet-heading a").first.get_attribute("href").startswith(
                "https://github.com/RyanBalech/llm-agent-evaluation/blob/")
            page.screenshot(path=str(output / "repolens-live.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "repolens-mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print("Browser smoke passed: demo, search, export, evaluation, mobile" +
          (", live GitHub indexing" if args.live else ""))


if __name__ == "__main__":
    main()
