import requests
from bs4 import BeautifulSoup

ARTICLE = "https://pubmed.ncbi.nlm.nih.gov/33667416/"

class PudmedArticleScrapper:
    """Scraper for extracting article contents from PubMed."""

    HEADER = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    MD_HEADERS = {
        "h1": "#",
        "h2": "##",
        "h3": "###",
        "h4": "####",
        "h5": "#####",
        "h6": "######",
    }


    def __init__(self, url):
        self.url = url


    def get_contents(self):
        """Fetch and parse the article content."""
        response = requests.get(url=self.url)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pmc_tag = soup.find('a', class_="link-item pmc", title="Free full text at PubMed Central")

        full_text_url = pmc_tag.get('href')

        response = requests.get(url=full_text_url, headers=self.HEADER)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article title
        title_tag = soup.find('h1')
        title = title_tag.text.strip() if title_tag else "Untitled"

        # Extract article body
        article = soup.find('section', class_="body main-article-body")

        sections = article.find_all("section", class_=False)
        contents = f"#{title}\n\n"
        for section in sections:
            for elem in section.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
                if elem.name in self.MD_HEADERS:
                    contents += self.MD_HEADERS[elem.name] + elem.text.strip() + "\n\n"
                else:
                    contents += elem.text.strip() + "\n\n"
        
        return contents.strip()

    
    def get_content_chunks(self, chunk_size=2500, overlap = 500):
        """Split article content into smaller chunks."""
        content = self.get_contents()
        return [content[i:i+chunk_size] for i in range(0, len(content), overlap)]