import unittest
from unittest.mock import patch

from src.ui.paper_visuals import (
    category_color,
    clean_arxiv_id,
    extract_figure_url,
    fetch_figure_url,
    paper_media_html,
    passage_body,
    passage_preview,
    primary_category,
)

PAGE_URL = "https://arxiv.org/html/2407.01449"
SAMPLE_HTML = """
<html><body>
<img src="https://static.arxiv.org/static/design-system/latest/logos/arxiv-logo.svg" alt="archive">
<figure class="ltx_figure">
<img src="2407.01449v6/images/final_architecture.png" class="ltx_graphics ltx_centering" alt="Refer to caption">
</figure>
<figure class="ltx_figure">
<img src="2407.01449v6/images/other.png" class="ltx_graphics" alt="Refer to caption">
</figure>
</body></html>
"""


class FigureExtractionTests(unittest.TestCase):
    def test_skips_logo_and_uses_first_graphic(self):
        found = extract_figure_url(SAMPLE_HTML, PAGE_URL)
        self.assertEqual(
            found,
            "https://arxiv.org/html/2407.01449v6/images/final_architecture.png",
        )

    def test_accepts_figure_on_configured_html_host(self):
        page = '<figure><img src="https://mirror.example/fig.png" class="ltx_graphics"></figure>'
        with patch("src.ui.paper_visuals.ARXIV_HTML_BASE_URL", "https://mirror.example/html"):
            found = extract_figure_url(page, "https://mirror.example/html/2407.01449")
        self.assertEqual(found, "https://mirror.example/fig.png")

    def test_rejects_offsite_figure(self):
        page = '<figure><img src="https://evil.example/fig.png" class="ltx_graphics"></figure>'
        self.assertIsNone(extract_figure_url(page, PAGE_URL))

    def test_rejects_javascript_url(self):
        page = '<figure><img src="javascript:alert(1)" class="ltx_graphics"></figure>'
        self.assertIsNone(extract_figure_url(page, PAGE_URL))


class PaperIdTests(unittest.TestCase):
    def test_accepts_modern_and_legacy_ids(self):
        self.assertEqual(clean_arxiv_id("2407.01449v6"), "2407.01449v6")
        self.assertEqual(clean_arxiv_id("arXiv:hep-th/9901001"), "hep-th/9901001")
        self.assertEqual(clean_arxiv_id("math.GT/0301001"), "math.GT/0301001")

    def test_rejects_path_tricks(self):
        self.assertEqual(clean_arxiv_id("../etc/passwd"), "")
        self.assertEqual(clean_arxiv_id("2407.01449/extra"), "")


class CoverTests(unittest.TestCase):
    def test_subcategories_use_distinct_colors(self):
        colors = [category_color(code) for code in ("cs.CL", "cs.AI", "cs.LG")]
        self.assertEqual(len(set(colors)), 3)
        self.assertEqual(category_color("cs.CL"), category_color("cs.CL"))
        self.assertEqual(primary_category(["cs.CL", "cs.AI"]), "cs.CL")

    def test_passage_preview_drops_repeated_title(self):
        title = "CHAMP: A Dataset"
        text = title + "\n\n" + ("Models solve problems with hints. " * 40)
        preview, truncated = passage_preview(text, title, limit=80)
        self.assertNotIn("CHAMP", preview)
        self.assertTrue(truncated)
        self.assertTrue(preview.endswith("…"))
        self.assertNotIn("CHAMP", passage_body(text, title))

    def test_cover_escapes_text(self):
        markup = paper_media_html(
            paper_id='<img src=x onerror=alert(1)>',
            category="cs.CL",
            year=2024,
            figure_url=None,
            abs_url="https://arxiv.org/abs/2407.01449",
        )
        self.assertNotIn("<img src=x", markup)
        self.assertIn("cover-cat", markup)
        self.assertIn("2024", markup)

    def test_figure_markup_keeps_arxiv_url_only(self):
        markup = paper_media_html(
            paper_id="2407.01449",
            category="cs.CL",
            year=2024,
            figure_url="https://arxiv.org/html/2407.01449v1/x.png",
            abs_url="https://arxiv.org/abs/2407.01449",
        )
        self.assertIn("https://arxiv.org/html/2407.01449v1/x.png", markup)
        self.assertIn("has-figure", markup)
        self.assertIn("Figure from the paper on arXiv", markup)

        blocked = paper_media_html(
            paper_id="2407.01449",
            category="cs.CL",
            year=2024,
            figure_url="https://evil.example/x.png",
            abs_url=None,
        )
        self.assertNotIn("evil.example", blocked)
        self.assertIn("paper-cover", blocked)


class FetchTests(unittest.TestCase):
    def test_fetch_returns_none_for_bad_id_without_network(self):
        self.assertIsNone(fetch_figure_url("not an id"))

    def test_fetch_checks_image_content_type(self):
        class Response:
            def __init__(self, status_code, headers, text="", url=""):
                self.status_code = status_code
                self.headers = headers
                self.text = text
                self.url = url

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_get(url, **kwargs):
            if url.endswith(".png"):
                return Response(200, {"content-type": "image/png"})
            return Response(
                200,
                {"content-type": "text/html"},
                text=SAMPLE_HTML,
                url=PAGE_URL,
            )

        with patch("src.ui.paper_visuals.requests.get", side_effect=fake_get):
            found = fetch_figure_url("2407.01449")
        self.assertEqual(
            found,
            "https://arxiv.org/html/2407.01449v6/images/final_architecture.png",
        )


if __name__ == "__main__":
    unittest.main()
