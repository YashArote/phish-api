import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import tldextract
import nltk
from nltk.corpus import words

english_words = set(words.words())

def extract_domain(url):
    ext = tldextract.extract(url)
    
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return url  # fallback

def remove_tld(domain_or_url: str) -> str:
    extracted = tldextract.extract(domain_or_url)
    parts = [part for part in [extracted.subdomain, extracted.domain] if part]
    result = ".".join(parts)
    return result[4:] if result.startswith("www.") else result

def jaccard_similarity(str1, str2, n=2):
    shingles1 = {str1[i:i + n] for i in range(len(str1) - n + 1)}
    shingles2 = {str2[i:i + n] for i in range(len(str2) - n + 1)}
    
    intersection = len(shingles1.intersection(shingles2))
    union = len(shingles1.union(shingles2))
    
    return (intersection / union)**0.5 if union != 0 else 0

def levenshtein_similarity(str1, str2):
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    distance = Levenshtein.distance(str1, str2)
    return 1 - (distance / max_len)

def cosine_tfidf_similarity(str1, str2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([str1, str2])
    
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0, 1]

#extract_domain = lambda url: tldextract.extract(url).domain
def domain_match_similarity(main1, main2):
   # print("Checking sim for",main1,main2)
   # main1, main2 = map(extract_domain, [main1, main2])
    print("Checking sim for",main1,main2)
    if not main1 or not main2:
        return 0

    if main1 == main2:
        return 1

    # Check if one is substring of another
    if main1 in main2 or main2 in main1:
        print("yes those are similar")
        shorter, longer = (main1, main2) if len(main1) < len(main2) else (main2, main1)
        proportion = len(shorter) / len(longer)
       
        if main1 in english_words or main2 in english_words:
            print("yes in english words")
            return 0

        # Proportion check: too little match
        if proportion < 0.3:
            print("yes less proportion")
            return 0

        print("matched this:", main1)
        print("matched this:", main2)
        return 2

    return 0.0



def start_end_similarity(main1, main2, length=5):
   
    def partial_score(a, b):
        matches = sum(c1 == c2 for c1, c2 in zip(a, b))
        return matches / length

    start1, end1 = main1[:length], main1[-length:]
    start2, end2 = main2[:length], main2[-length:]

    start_score = partial_score(start1, start2)
    end_score = partial_score(end1, end2)

    return (start_score + end_score) / 2


def combined_similarity(url1, url2, w1=0.2, w2=0.6, w3=0.2):
    print("url is: ",url1)
    print("url is: ",url2)
    
    url1 = remove_tld(url1)
    url2 = remove_tld(url2)
 
    jac_sim = jaccard_similarity(url1, url2)
    lev_sim = levenshtein_similarity(url1, url2)
    #cos_sim = cosine_tfidf_similarity(url1, url2)
    dom_sim = domain_match_similarity(url1, url2)
    se_sim = start_end_similarity(url1, url2)

    combined_score = (w1 * jac_sim) + (w2 * lev_sim) + (w3 * se_sim)

    print(f"Similarity Report for URLs:")
    print(f"URL 1: {url1}\nURL 2: {url2}")
    print(f"Jaccard Similarity: {jac_sim:.4f}")
    print(f" Levenshtein Similarity: {lev_sim:.4f}")
   # print(f" Cosine Similarity (Not Included in Score): {cos_sim:.4f}")
    print(f" Domain Match Similarity (Not Included in Score): {dom_sim:.4f}")
    print(f" Start-End Similarity: {se_sim:.4f}")
    print(f" Combined Similarity Score: {combined_score:.4f}")
    print("url is: ",url1)
    print("url is: ",url2)
    if dom_sim==1:
        return "no"
    if combined_score>=0.8 or dom_sim==2:
        return "true"
    return "false"

url1 = "google.com"
url2 = "googlie.com"

print(combined_similarity(url1, url2))
