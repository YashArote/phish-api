
import pandas as pd
import numpy as np
import faiss
import os
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
import sys
from similarity_measure import combined_similarity


def shingler(url, k=3):
    """Generates k-shingles for a URL (removing 'www.') with float handling"""
    if isinstance(url, float):
        return set()

    url = str(url)
    if url.startswith("www."):
        url = url[4:]

    shingles = {url[i:i + k] for i in range(len(url) - k + 1)}
    return shingles


def get_minhash_signature(url, num_perm=128):
    """Generates MinHash signature for the URL"""
    shingles = shingler(url)
    if not shingles:
        return np.zeros(num_perm, dtype=np.float32)

    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode('utf8'))

    return np.array(m.digest(), dtype=np.float32)


def lev_dis(target, candidate):
    """Levenshtein similarity with length penalty."""
    base_similarity = Levenshtein.ratio(target, candidate)
    m, n = len(target), len(candidate)
    length_penalty = abs(m - n) / max(m, n)
    score = max(base_similarity - length_penalty, 0)

    print(f"🔹 Levenshtein similarity ({target} vs {candidate}): {score:.4f}")
    return round(score, 4)


def sequential_similarity(target, candidate):
    """Strict sequential similarity with length penalty."""
    m, n = len(target), len(candidate)
    i, j = 0, 0
    match_len = 0

    while i < m and j < n:
        if target[i] == candidate[j]:
            match_len += 1
            i += 1
            j += 1
        else:
            j += 1  

    min_length = min(m, n)
    base_similarity = match_len / min_length if min_length > 0 else 0
    length_penalty = abs(m - n) / max(m, n)
    score = max(base_similarity - length_penalty, 0)

    print(f"🔹 Sequential similarity ({target} vs {candidate}): {score:.4f}")
    return round(score, 4)


LEXICAL_INDEX_PATH = "faiss_lexical.index"
SEMANTIC_INDEX_PATH = "faiss_semantic.index"
EMBEDDINGS_PATH = "semantic_embeddings.npy"
DATASET_PATH = "majestic_thousands.csv"

num_perm = 128
semantic_dim = 384
top_k = 5

def load_dataset():
    """Load the first 10,000 URLs from CSV"""
    print("\n📊 Loading dataset...")
    df = pd.read_csv(DATASET_PATH, usecols=["Domain"], nrows=10000)
    df['Domain'] = df['Domain'].astype(str).fillna("")
    
    urls = df['Domain'].tolist()
    print("Done loading")
    return urls


def build_lexical_index(urls):
    """Build and store lexical FAISS index with MinHash signatures"""
    print("\n⚙️ Building Lexical FAISS index with MinHash...")

    lexical_signatures = np.array([get_minhash_signature(url, num_perm) for url in urls])

    lexical_index = faiss.IndexFlatL2(num_perm)
    lexical_index.add(lexical_signatures)

    faiss.write_index(lexical_index, LEXICAL_INDEX_PATH)
    print(f"✅ Lexical FAISS index saved at {LEXICAL_INDEX_PATH}")

    return lexical_index


def load_lexical_index(urls):
    """Load or build lexical FAISS index"""
    if os.path.exists(LEXICAL_INDEX_PATH):
        print("\n🔄 Loading existing Lexical FAISS index...")
        return faiss.read_index(LEXICAL_INDEX_PATH)
    else:
        return build_lexical_index(urls)




model = SentenceTransformer('all-MiniLM-L6-v2')

def build_semantic_index(urls):
    """Build and store semantic FAISS index"""
    print("\n⚙️ Building Semantic FAISS index...")

    semantic_embeddings = model.encode(urls, convert_to_tensor=False)

    np.save(EMBEDDINGS_PATH, semantic_embeddings)

    semantic_index = faiss.IndexFlatL2(semantic_dim)
    semantic_index.add(np.array(semantic_embeddings))

    faiss.write_index(semantic_index, SEMANTIC_INDEX_PATH)
    print(f" Semantic FAISS index saved at {SEMANTIC_INDEX_PATH}")

    return semantic_index


def load_semantic_index(urls):
    """Load or build semantic FAISS index"""
    if os.path.exists(SEMANTIC_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        print("\n🔄 Loading existing Semantic FAISS index...")
        return faiss.read_index(SEMANTIC_INDEX_PATH)
    else:
        return build_semantic_index(urls)


if os.path.exists(LEXICAL_INDEX_PATH) and os.path.exists(SEMANTIC_INDEX_PATH):
    print("\n FAISS indexes already exist. Skipping dataset loading.")
    
    lexical_index = faiss.read_index(LEXICAL_INDEX_PATH)
    semantic_index = faiss.read_index(SEMANTIC_INDEX_PATH)

    embeddings = np.load(EMBEDDINGS_PATH)
    num_urls = embeddings.shape[0]
    urls = load_dataset()

else:
    urls = load_dataset()
    
    lexical_index = load_lexical_index(urls)
    semantic_index = load_semantic_index(urls)

print(f"\n FAISS Indexing completed with {len(urls)} URLs")



def search_url(target_url, top_k=1, lexical_weight=0.4, semantic_weight=0.4, seq_weight=0.2, lev_weight=0.4):
    """Search for similar URLs using combined lexical, semantic, and sequential similarity"""

    print(f"\n🔎 Searching for: {target_url}")

    # Lexical Matching (MinHash)
    lexical_signature = get_minhash_signature(target_url, num_perm).reshape(1, -1)
    lexical_distances, lexical_indices = lexical_index.search(lexical_signature, top_k)

    # Semantic Matching
    semantic_embedding = model.encode([target_url], convert_to_tensor=False)
    semantic_distances, semantic_indices = semantic_index.search(semantic_embedding, top_k)

    results = []
    for i in range(top_k):
        lex_idx = lexical_indices[0][i]
        sem_idx = semantic_indices[0][i]

        lex_url = urls[lex_idx]
        sem_url = urls[sem_idx]

        lex_score = 1 / (1 + lexical_distances[0][i])
        sem_score = 1 / (1 + semantic_distances[0][i])

        #seq_lex = sequential_similarity(target_url, lex_url)
        #seq_sem = sequential_similarity(target_url, sem_url)
        #lev_lex = lev_dis(target_url, lex_url)
        #lev_sem = lev_dis(target_url, sem_url)

        final_lex = (
             lex_score
            #(seq_weight * seq_lex) +
            #(lev_weight * lev_lex)
        )

        final_sem = (
            sem_score
        )

        results.append((lex_url, sem_url, final_lex, final_sem))

    results.sort(key=lambda x: max(x[2], x[3]), reverse=True)

    print("\n Final Combined Results:")
    #for i, (l_url, s_url, score_lex, score_sem) in enumerate(results):
    #    print(f"{i+1}. {l_url}, {s_url}, Lex: {score_lex:.4f}, Sem: {score_sem:.4f}")
    return results

def is_Similar(target_url):
    results=search_url(target_url,20)
    for i, (l_url, s_url, score_lex, score_sem) in enumerate(results):
        #print(f"{i+1}. {l_url}, {s_url}, Lex: {score_lex:.4f}, Sem: {score_sem:.4f}")
        similarity_lex=combined_similarity(target_url,l_url)
        similarity__sem=combined_similarity(target_url,s_url)
        print("similarity lex",similarity_lex)
        if similarity_lex=="no" or similarity__sem=="no":
            return False
        if similarity_lex=="true" or similarity__sem=="true":
            print(l_url)
            print(s_url)
            return  s_url if similarity__sem=="true" else l_url
    return False
        #print(f"{i+1}. {l_url}, {s_url}, Lex: {score_lex:.4f}, Sem: {score_sem:.4f}")
