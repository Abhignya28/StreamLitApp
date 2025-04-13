# Import libraries
import streamlit as st
import pinecone 
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec


# Set up the app title
st.title("Movie Recommendation Engine")
st.write("Describe the type of movie you want to watch, and we'll recommend the best matches!")


@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("pcsk_3xYAm8_3qHVaSMC7D93DqS2vTHGQNg2fDiMz9hCzcS2S68sMCXsGpdWp8DeoVgxzT8gJCK") or "pcsk_3xYAm8_3qHVaSMC7D93DqS2vTHGQNg2fDiMz9hCzcS2S68sMCXsGpdWp8DeoVgxzT8gJCK")
    index = pc.Index("imdb-movies")
    return index


# Load the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone and the model
index = init_pinecone()
model = load_model()

# User input
user_query = st.text_input("What kind of movie are you looking for? (e.g., 'funny superhero movies' or 'scary ghost stories')")

num_results = st.slider("Number of recommendations:", 1, 10, 5)

if user_query:
    query_embedding = model.encode([user_query])[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=num_results,  
        include_metadata=True  
    )

    
    st.subheader("🎥 Recommended Movies")
    for i, match in enumerate(results["matches"], 1):
        metadata = match["metadata"]
        st.write(f"**{i}. {metadata['Series_Title']}** ⭐ ({metadata['IMDB_Rating']}/10)")
        st.write(f"**Genre:** {metadata['Genre']}")
        st.write(f"**Director:** {metadata['Director']}")
        st.write(f"**Stars:** {metadata['Star1']}, {metadata['Star2']}")
        st.write(f"**Overview:** {metadata['Overview']}")
        st.write(f"**Similarity Score:** {match['score']:.2f}")
        st.write("---")

# Add a footer
st.write("Powered by Pinecone, Streamlit, and Hugging Face 🤗")
