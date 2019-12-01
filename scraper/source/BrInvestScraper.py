from bs4 import BeautifulSoup
import re
from os import path, remove
from urllib.request import urlopen, Request
import logging as log


BROWSER_USER_AGENT = "Chrome/77.0.3865.75"
RAW_TEXT_PATH = f"{path.dirname(path.abspath(__file__))}\\..\\data\\"
HTTPS_URL_PATTERN = "^https:\/\/.*"
RE_HTTPS = re.compile(HTTPS_URL_PATTERN)
ROOT_URL = "https://br.investing.com"
SEARCH_PAGE_URL = "/equities/petrobras-on-news{page_index}"
log.basicConfig(level=log.DEBUG)


def run(page_depth):
    for page_number in range(1, page_depth):
        new_links = get_new_links(page_number)
        for link in new_links:
            link_file_path, text_number, link_content = get_link_content(link)
            save_link_content(link_file_path, text_number, link_content)
    log.info("All texts have been saved!")


def get_new_links(page_index, app=False):
    links_list = []
    soup = get_soup_element(page_index)
    article_tags = get_articles(soup)
    if article_tags is None:
        log.error("There isn't a div tag with 10 articles in this page!")
        return None
    for article in article_tags:
        link_complement = article.a.get("href")
        if RE_HTTPS.match(link_complement):
            # The complement is a full link, discard this type of article
            log.warning(f"Discarding link: {link_complement}")
            continue
        new_link = ROOT_URL + link_complement
        if app:
            links_list.append(new_link)
            log.info(f"New link: {new_link}")
        elif not link_exists(new_link):
            links_list.append(new_link)
            log.info(f"New link: {new_link}")
    return links_list


def get_soup_element(page):
    log.info(f"Looking for soup element for page {page} ... ")
    url = make_page_url(page)
    req = Request(url, headers={"User-Agent": BROWSER_USER_AGENT})
    resp = urlopen(req).read()
    soup = BeautifulSoup(resp, "html.parser")
    return soup


def make_page_url(page_index: int):
    url_complement = SEARCH_PAGE_URL.format(page_index=f"/{page_index}")
    page_url = ROOT_URL + url_complement
    return page_url


def get_articles(soup):
    div_tags = soup.find_all("div", {"class": "mediumTitle1"})
    for div_tag in div_tags:
        article_tags = div_tag("article", {"class": "articleItem"})
        if len(article_tags) != 10:
            continue
        return article_tags
    return None


def save_link_content(link_file_path, link_number, link_content):
    link_file = open(link_file_path, "w+")
    try:
        link_file.write(link_content)
        link_file.close()
        log.info(f"Content saved, number: {link_number}")
    except Exception as error:
        log.error(error)
        link_file.close()
        remove(link_file_path)
    return


def link_exists(link):
    text_title = link.split("/")[-1]
    text_file_name = f"{text_title}.csv"
    text_file_path = f"{RAW_TEXT_PATH}{text_file_name}"
    return path.exists(text_file_path)


def get_link_content(link):
    text_title = link.split("/")[-1]
    text_number = link.split("-")[-1]
    text_file_name = f"noticia_{text_number}.csv"
    text_file_path = f"{RAW_TEXT_PATH}{text_file_name}"
    req = Request(link, headers={"User-Agent": BROWSER_USER_AGENT})
    resp = urlopen(req).read()
    soup = BeautifulSoup(resp, "html.parser")
    text_content = make_text_content(soup, text_number)
    return text_file_path, text_number, text_content


def make_text_content(soup, text_number):
    # Treat the Text Header
    text_header = soup.find("h1", {"class": "articleHeader"}).text
    text_header = treat_text(text_header)
    # Treat the Text Date
    text_date = soup.find("meta", {"itemprop": "dateModified"}).attrs["content"].split(" ")[0]
    # Treat the Text Content
    text_content = ""
    content_div = soup.find("div", {"class": "articlePage"})
    for p_tag in content_div:
        text_content = text_content + str(p_tag)
    text_content = treat_text(text_content)
    # Build a csv like file, using the "¨" character as a separator
    text_content = text_header + "¨" + text_date + "¨" + text_number + "¨" + text_content
    return text_content


def treat_text(text):
    # Remove html tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove two or more consecutive spaces from the text to avoid csv problems
    text = re.sub(r"\s+", " ", text)
    return text


if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log.INFO)
    run(page_depth=101)


