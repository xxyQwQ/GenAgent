from bs4 import BeautifulSoup


def fetch_reference_by_name(references: list, name: str):
    for reference in references:
        if reference.metadata['name'] == name:
            return reference
    return None


def safe_extract_from_soup(soup: BeautifulSoup, tag: str):
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()
