from bs4 import BeautifulSoup


def safe_extract_from_soup(soup: BeautifulSoup, tag: str):
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()
