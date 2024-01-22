import streamlit as st
import os
import pickle
import uuid

import sys
sys.path.append("..")

import pandas as pd

from kgsearch import preprocess, prompts, graph

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

with open('../data/text.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys:")
print(*data.keys(), sep = "\n")


import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Function to map NLTK position tags to WordNet position tags
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# Instantiate the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False
)

df = pd.DataFrame()
for key in data.keys():
    # Split the string using the splitter
    _ = pd.DataFrame(splitter.split_text(str(data[key])), columns=['chunk'])
    _['source'] = key
    _['chunk_id'] = [uuid.uuid4().hex for i in range(len(_))]

    df = pd.concat([df, _], ignore_index=True)



# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

df['lamma'] = [' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(text)]) for text in df.chunk]


# Sample documents
documents = df['lamma'] 
# Vectorize documents
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df.lamma)

# Perform K-means clustering
num_clusters = 3  # Adjust based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
from sklearn.metrics.pairwise import cosine_similarity

def search_and_rank(query, vectorizer, kmeans, documents):
    # Transform the query and find the nearest cluster
    query_vec = vectorizer.transform([query])
    query_cluster = kmeans.predict(query_vec)[0]

    # Find documents in the same cluster
    cluster_docs_idx = np.where(kmeans.labels_ == query_cluster)[0]
    cluster_docs = [documents[idx] for idx in cluster_docs_idx]

    # Calculate distances from each doc in the cluster to the query's vector
    distances = cosine_similarity(query_vec, X[cluster_docs_idx]).flatten()

    # Rank documents based on their distance to the query's vector
    ranked_docs_idx = np.argsort(distances)[::-1]
    ranked_docs = [cluster_docs[idx] for idx in ranked_docs_idx]

    return ranked_docs, distances[ranked_docs_idx]

# Streamlit interface code remains the same
# Replace the search function call with search_and_rank

# Streamlit interface
st.title("Document Search with TF-IDF and K-Means")

query = st.text_input("Enter your search query:")

if query:
    ranked_results, distances = search_and_rank(query, vectorizer, kmeans, documents)
    st.write("Ranked matching documents:")
    for result, dist in zip(ranked_results, distances):
        st.write(f"{result} (Similarity: {dist:.2f})")
