import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

HOME_URL = "https://www.berkshirehathaway.com/letters/letters.html"
REQUEST_HEADERS = {
    # Berkshire blocks generic clients; mimic a normal browser UA
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
BASE_DIR = Path("data/raw")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def load_homepage(html_path: str | None = None) -> str:
    """Return HTML of the shareholder letters homepage.
    If html_path is provided, read from file; otherwise download."""
    if html_path:
        try:
            return Path(html_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Some archived files are saved with Windows-1252 characters
            return Path(html_path).read_text(encoding="windows-1252")
    resp = requests.get(HOME_URL, timeout=15, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return resp.text


def extract_letter_links(home_html: str) -> dict[int, str]:
    """
    Parse homepage HTML and return {year: absolute_url} for each letter.
    Handles both .html and .pdf links.
    """
    soup = BeautifulSoup(home_html, "html.parser")
    links = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # We only care about /letters/... links
        if "/letters/" not in href:
            continue

        # Try to find a 4-digit year in the href (1977-2099)
        m = re.search(r"(19[7-9]\d|20[0-4]\d)", href)
        if not m:
            continue

        year = int(m.group(1))

        # Build absolute URL in case href is relative
        url = urljoin(HOME_URL, href)

        # In case of duplicates, last one wins (usually same anyway)
        links[year] = url

    return dict(sorted(links.items()))


def download_letter(year: int, url: str, delay: float = 0.5) -> Path:
    """
    Download one letter and save to data/raw/<year>/letter_<year>.<ext>.
    Returns the saved path.
    """
    year_dir = BASE_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # Decide extension based on URL (html/pdf/other)
    if url.lower().endswith(".pdf"):
        ext = ".pdf"
    elif url.lower().endswith(".html") or url.lower().endswith(".htm"):
        ext = ".html"
    else:
        # fallback if no extension, treat as .html
        ext = ".html"

    out_path = year_dir / f"letter_{year}{ext}"

    print(f"Downloading {year} from {url} -> {out_path}")
    resp = requests.get(url, timeout=30, headers=REQUEST_HEADERS)
    resp.raise_for_status()

    # Use binary write so it works for both HTML and PDF
    out_path.write_bytes(resp.content)

    # Be polite: small delay between requests
    time.sleep(delay)
    return out_path


def main(html_path: str | None = None):
    # 1. Load homepage HTML
    home_html = load_homepage(html_path)

    # 2. Extract {year: url}
    links = extract_letter_links(home_html)
    print(f"Found {len(links)} letter links:")
    for y, u in links.items():
        print(f"  {y}: {u}")

    # 3. Download each letter
    for year, url in links.items():
        try:
            saved = download_letter(year, url)
            print(f"✔ Saved {year} -> {saved}")
        except Exception as e:
            print(f"✘ Error downloading {year} from {url}: {e}")


if __name__ == "__main__":
    # Usage:
    #   python download_letters.py           # fetch homepage from web
    #   python download_letters.py homepage.html  # use local saved homepage
    html_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(html_arg)
