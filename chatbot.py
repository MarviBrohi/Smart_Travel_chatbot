import json
import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load trained vectorizer and data
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
df = pd.read_csv("model/data.csv")

def get_response_ml(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, vectorizer.transform(df["text"])).flatten()
    top_index = scores.argmax()
    row = df.iloc[top_index]
    return f"🧭 I recommend '{row['name']}' ({row['category']}) in {row['city'].title()}."
