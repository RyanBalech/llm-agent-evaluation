"""Exercise a running RepoLens through a browser; save desktop and mobile screenshots.

Install playwright, run `python -m playwright install chromium`, and start RepoLens first.
Use --live to also index the public project repository (requires network access).
"""
import argparse
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--static", action="store_true", help="Verify the static browser deployment")
    parser.add_argument("--explain", action="store_true", help="Download and exercise the real local model")
    parser.add_argument("--channel", help="Optional installed browser, e.g. chrome")
    args = parser.parse_args()
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        gpu_args = ["--enable-unsafe-webgpu"] if args.explain else []
        if args.explain and sys.platform == "win32":
            gpu_args.append("--use-angle=d3d11")
        browser = p.chromium.launch(channel=args.channel, args=gpu_args)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.get_by_text("Browser ready" if args.static else "Server connected", exact=True)).to_be_visible()
        page.get_by_role("button", name="Try the sample shop").click()
        expect(page.locator("#active-index strong")).to_have_text("Sample shop · demonstration")
        expect(page.get_by_role("button", name="Try the sample shop")).to_be_enabled()
        search = page.get_by_role("button", name="Find relevant code")
        expect(search).to_be_enabled(timeout=30000)
        search.click()
        expect(page.locator(".result-card").first).to_be_visible(timeout=120000)
        expect(page.locator(".result-card").first.locator(".code-line").first).to_be_visible()
        expect(page.locator(".result-card").first.locator(".line-number").first).to_have_text(
            re.compile(r"[1-9][0-9]*"))
        expect(page.locator(".result-name").filter(has_text="inventory.py")).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role("button", name="Export JSON").click()
        download.value.save_as(output / "demo-search.json")
        result = json.loads((output / "demo-search.json").read_text())
        assert result["files"] and result["selected_characters"] <= result["context_budget_chars"]
        assert result["files"][0]["snippets"][0]["text"]
        page.screenshot(path=str(output / "repolens-desktop.png"), full_page=True)
        if args.explain:
            page.locator("#explain-button").click()
            page.wait_for_function("['ready', 'failed'].includes(document.getElementById('explanation').dataset.status)", timeout=900000)
            assert page.locator("#explanation").get_attribute("data-status") == "ready", page.locator("#explanation").inner_text()
            assert page.locator("#explanation blockquote").count() > 0
            with page.expect_download() as generated_download:
                page.get_by_role("button", name="Export JSON").click()
            generated_download.value.save_as(output / "browser-explanation.json")
            page.screenshot(path=str(output / "repolens-explanation.png"), full_page=True)
        if args.static:
            page.locator(".index-select").filter(has_text="RyanBalech/llm-agent-evaluation").click()
            page.get_by_label("What is going wrong?").fill("select_under_character_budget skips chunks that exceed the context budget")
            search.click()
            expect(page.locator(".result-card").first).to_be_visible(timeout=30000)
            source_url = page.locator(".snippet-heading a").first.get_attribute("href")
            assert re.match(
                r"https://github\.com/RyanBalech/llm-agent-evaluation/blob/[0-9a-f]{40}/",
                source_url)
        page.get_by_role("button", name="Evaluation lab").click()
        expect(page.get_by_text("0.425", exact=True).first).to_be_visible()
        expect(page.get_by_role("heading", name="Frozen 20-task comparison")).to_be_visible()
        assert page.locator("#baseline-tasks tbody tr").count() == 20
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
          (", live GitHub indexing" if args.live else "") +
          (", real local model generation" if args.explain else ""))


if __name__ == "__main__":
    main()
