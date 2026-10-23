"""Host-allowlist tests for the scrapers. No network: requests.get is mocked.

Run from the repo root:  python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bs4 import BeautifulSoup

from src.scraping import utils

ALLOWED = ["gbls.org"]


def page(*hrefs: str) -> BeautifulSoup:
    return BeautifulSoup("".join(f'<a href="{h}">x</a>' for h in hrefs), "html.parser")


def fake_response(final_url: str, body: bytes = b"%PDF-1.4"):
    resp = mock.Mock()
    resp.url = final_url
    resp.raise_for_status = mock.Mock()
    resp.iter_content = mock.Mock(return_value=[body])
    resp.text = "<html></html>"
    return resp


class HostAllowedTest(unittest.TestCase):
    def test_exact_host_and_subdomains_pass(self):
        for url in ["https://gbls.org/a", "https://www.gbls.org/a", "http://GBLS.org:443/a"]:
            self.assertTrue(utils.host_allowed(url, ALLOWED), url)

    def test_lookalike_hosts_userinfo_and_other_schemes_fail(self):
        for url in [
            "https://xgbls.org/a",
            "https://gbls.org.attacker.example/a",
            "https://gbls.org@attacker.example/a",
            "https://attacker.example/gbls.org",
            "ftp://gbls.org/a",
            "javascript:alert(1)",
        ]:
            self.assertFalse(utils.host_allowed(url, ALLOWED), url)


class LinkExtractionTest(unittest.TestCase):
    def test_extract_links_keeps_only_allowlisted_hosts(self):
        soup = page("/housing", "https://xgbls.org/housing", "https://gbls.org@attacker.example/x", "https://www.gbls.org/b")
        self.assertEqual(
            utils.extract_links(soup, "https://www.gbls.org/", allowed_domains=ALLOWED),
            ["https://www.gbls.org/b", "https://www.gbls.org/housing"],
        )

    def test_extract_pdf_links_filters_hosts_when_given_an_allowlist(self):
        soup = page("/guide.pdf", "https://attacker.example/tenant-guide.pdf", "https://cdn.gbls.org/download/1")
        self.assertEqual(
            utils.extract_pdf_links(soup, "https://www.gbls.org/", allowed_domains=ALLOWED),
            ["https://cdn.gbls.org/download/1", "https://www.gbls.org/guide.pdf"],
        )


@mock.patch.object(utils.time, "sleep", lambda _s: None)
class RedirectTest(unittest.TestCase):
    def test_fetch_page_rejects_a_redirect_off_the_allowlist(self):
        with mock.patch.object(utils.requests, "get", return_value=fake_response("https://attacker.example/landing")):
            self.assertIsNone(utils.fetch_page("https://www.gbls.org/x", allowed_domains=ALLOWED))

    def test_fetch_page_accepts_a_redirect_within_the_allowlist(self):
        resp = fake_response("https://www.gbls.org/x/")
        with mock.patch.object(utils.requests, "get", return_value=resp):
            self.assertIs(utils.fetch_page("https://gbls.org/x", allowed_domains=ALLOWED), resp)

    def test_fetch_pdf_rejects_an_off_allowlist_url_or_redirect_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "doc.pdf"
            with mock.patch.object(utils.requests, "get", return_value=fake_response("https://attacker.example/x.pdf")):
                self.assertIsNone(utils.fetch_pdf("https://www.gbls.org/x.pdf", target, allowed_domains=ALLOWED))
                self.assertIsNone(utils.fetch_pdf("https://attacker.example/x.pdf", target, allowed_domains=ALLOWED))
            self.assertFalse(target.exists())

    def test_fetch_pdf_saves_an_allowlisted_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "doc.pdf"
            with mock.patch.object(utils.requests, "get", return_value=fake_response("https://www.gbls.org/x.pdf")):
                self.assertEqual(utils.fetch_pdf("https://www.gbls.org/x.pdf", target, allowed_domains=ALLOWED), target)
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
