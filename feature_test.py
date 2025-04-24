import requests
from bs4 import BeautifulSoup
import tldextract
from textblob import TextBlob
from urllib.parse import urlparse
import re


class PhishingFeatureExtractor:
    FREE_HOSTS = [
        "vercel.app", "netlify.app", "glitch.me", "webflow.io", "000webhostapp.com",
        "pages.dev", "github.io", "firebaseapp.com", "wordpress.com", "weebly.com",
        "sites.google.com", "blogspot.com", "wixsite.com", "strikingly.com",
        "yolasite.com", "jimdosite.com", "biz.nf", "awardspace.com", "tripod.com",
        "angelfire.com", "geocities.ws", "infinityfree.net", "zyro.com"
    ]

    def __init__(self, url):
        self.url = url
        self.html = None
        self.soup = None
        self.domain = None
        self._fetch_page()

    def _fetch_page(self):
    
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(self.url, headers=headers, timeout=10)
        response.raise_for_status()
        self.html = response.text
        self.soup = BeautifulSoup(self.html, 'html.parser')

        # Extract base domain
        ext = tldextract.extract(self.url)
        self.domain = f"{ext.domain}.{ext.suffix}"
    

    def is_free_hosting_platform(self):
        if not self.domain:
            return 0

        for free_host in self.FREE_HOSTS:
            if self.domain.endswith(free_host):
                return 1
        return 0

    def is_link_behavior_suspicious(self, null_threshold=0.7, external_threshold=0.7):
        if not self.soup or not self.domain:
            return 0

        a_tags = self.soup.find_all('a')
        total_links = len(a_tags)
        if total_links == 0:
            return 0  # Not enough data to analyze

        null_count = 0
        external_count = 0

        parsed_base = urlparse(self.url)
        base_netloc = parsed_base.netloc.replace("www.", "")

        for tag in a_tags:
            href = tag.get('href', '').strip().lower()
            if not href or href in ('#', 'javascript:void(0)', 'javascript:;'):
                null_count += 1
                continue

            parsed_href = urlparse(href)
            if parsed_href.netloc and base_netloc not in parsed_href.netloc:
                external_count += 1
        
        null_ratio = null_count / total_links
        external_ratio = external_count / total_links

        if null_ratio >= null_threshold or external_ratio >= external_threshold:
            return 1
        return 0

    def is_form_dominated_page(self, form_threshold=0.6):
        if not self.soup:
            return 0

        num_forms = len(self.soup.find_all('form'))

        # Tags considered as content-bearing blocks
        content_tags = ['div', 'p', 'span', 'section', 'article', 'ul', 'ol', 'li', 'img', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        non_form_blocks = sum(len(self.soup.find_all(tag)) for tag in content_tags)

        total_blocks = num_forms + non_form_blocks

        if total_blocks == 0:
            return 0  # Not enough data

        form_ratio = num_forms / total_blocks

        return int(form_ratio >= form_threshold)
    

    def is_language_suspicious(self):
        if not self.soup:
            return 0

        try:
            # Extract visible text
            for script_or_style in self.soup(['script', 'style']):
                script_or_style.decompose()
            text = self.soup.body.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())  # Normalize whitespace

            if len(text) < 30:
                return 0  # Not enough text to detect language

            # Detect language using TextBlob
            blob = TextBlob(text)
            detected_lang = blob.detect_language()

            if detected_lang != 'en':
                return 1  # Suspicious (non-English language detected)
            return 0  # English language

        except Exception:
            return 0  # In case of any error, assume it's not suspicious
    # In case of parsing or detection failure

    def is_right_click_disabled(self):
        if not self.soup:
            return 0

        # Check for inline 'oncontextmenu' attributes
        elements_with_oncontext = self.soup.find_all(attrs={"oncontextmenu": True})
        for el in elements_with_oncontext:
            val = el.get("oncontextmenu", "").lower()
            if "return false" in val or "preventdefault" in val:
                return 1

        # Check JavaScript that disables context menu
        scripts = self.soup.find_all("script")
        for script in scripts:
            if not script.string:
                continue
            js = script.string.lower()
            if "oncontextmenu" in js and ("return false" in js or "preventdefault" in js):
                return 1
            if "addEventListener" in js and "contextmenu" in js and ("preventdefault" in js or "return false" in js):
                return 1

        return 0
    def has_suspicious_tld(self):
        suspicious_tlds = [
            "tk", "ga", "ml", "cf", "gq",     # Freenom ccTLDs
            "buzz", "xyz", "top", "icu", "wang", "live", "online", "host"
        ]

        parsed_url = urlparse(self.url)
        domain = parsed_url.netloc.lower()

        # Extract the TLD
        if '.' in domain:
            tld = domain.split('.')[-1]
            if tld in suspicious_tlds:
                return 1
        return 0
    def has_bad_letter_char_ratio(self, threshold=0.6):
        parsed_url = urlparse(self.url)
        domain = parsed_url.netloc.lower()

        # Remove 'www.' if present
        if domain.startswith("www."):
            domain = domain[4:]

        # Remove port if any
        domain = domain.split(":")[0]

        total_chars = len(domain)
        if total_chars == 0:
            return 0  # avoid division by zero

        letter_count = sum(c.isalpha() for c in domain)
        ratio = letter_count / total_chars

        return 1 if ratio < threshold else 0

    def has_fullpage_image(self):
        soup = self.soup
        if not soup:
            return 0
        for img in soup.find_all("img"):
            # 1. Check style attribute (same as before)
            style = img.get("style", "")
            width_match = re.search(r'width\s*:\s*100%', style)
            height_match_px = re.search(r'height\s*:\s*(\d+)px', style)
            height_match_vh = re.search(r'height\s*:\s*(\d+)(vh|%)', style)

            if width_match:
                if height_match_px and int(height_match_px.group(1)) >= 600:
                    return 1
                if height_match_vh and int(height_match_vh.group(1)) >= 80:
                    return 1

            width_attr = img.get("width")
            height_attr = img.get("height")
            if width_attr and height_attr:
                try:
                    w = int(width_attr)
                    h = int(height_attr)
                    if w >= 1000 and h >= 600:
                        return 1
                except ValueError:
                    pass

           
            sizes = img.get("sizes", "")
            if "100vw" in sizes:
                return 1

            class_attr = img.get("class", [])
            if isinstance(class_attr, str):
                class_attr = class_attr.split()
            if any(cls.lower() in ["full-width", "hero", "banner", "image"] for cls in class_attr):
                return ArithmeticError

        return 0


    def mentions_popular_site_but_not_in_domain(self):
        popular_sites = [
             "facebook", "instagram", "apple", "microsoft",
            "paypal", "amazon", "netflix", "linkedin", "yahoo", "snapchat"
        ]

        parsed = urlparse(self.url)
        domain = parsed.netloc.lower()

        # Extract title and header-related divs
        soup = self.soup
        if not soup:
            return 0
        
        title_text = soup.title.string.lower() if soup.title and soup.title.string else ""

        # Gather header-like divs
        header_texts = []
        for tag in soup.find_all(True, {"id": True, "class": True}):
            id_class = (tag.get("id", "") + " " + " ".join(tag.get("class", []))).lower()
            if any(word in id_class for word in ["header", "head", "top"]):
                header_texts.append(tag.get_text(strip=True).lower())

        combined_text = title_text + " " + " ".join(header_texts)

        # Look for popular site names
        for site in popular_sites:
            if site in combined_text and site not in domain:
                return 1
        return 0
    def is_domain_too_long(self, threshold=30):
        parsed = urlparse(self.url)
        domain = parsed.netloc.lower().replace("www.", "")
        return 1 if len(domain) > threshold else 0
    def calculate_phishing_score(self):
        if not self.soup:
            return 0, [], "Error loading site"

        f1 = self.is_free_hosting_platform()
        f2 = self.is_link_behavior_suspicious()
        f3 = self.is_form_dominated_page()
        f4 = self.is_right_click_disabled()
        f5 = self.is_language_suspicious()
        f6 = self.has_suspicious_tld()
        f7 = self.has_bad_letter_char_ratio()
        f8 = self.has_fullpage_image()
        f9 = self.mentions_popular_site_but_not_in_domain()
        f10 = self.is_domain_too_long()

        return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

warnings = [
    "🚩 Free hosting platform.",
    "⚠️ Suspicious link behavior.",
    "📝 Form-heavy page.",
    "🚫 Right-click disabled.",
    "🌐 Mismatched language.",
    "🔻 Suspicious TLD.",
    "🧪 Unusual letter-char ratio.",
    "🖼️ Full-page image detected.",
    "🎭 Popular site mentioned but not in domain.",
    "🔡 Domain is too long."
]

def is_Safe(url):
    extractor = PhishingFeatureExtractor(url)
    scores= extractor.calculate_phishing_score()
    print("scores",scores)
    total_score=sum(scores)
    print(total_score)
    if total_score>=2:
        triggered_warnings = [msg for flag, msg in zip(scores, warnings) if flag]
        print(triggered_warnings)
        return triggered_warnings
    return False
