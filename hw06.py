import streamlit as st
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAIError
import pickle  # For saving embeddings
import os

def load_api_key():
    """Load OpenAI API key from secrets."""
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found in secrets.")
        st.stop()
    return openai_api_key

def load_data(uploaded_file=None):
    """Load and preprocess the news data."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Load the default CSV file
        default_file_path = "/workspaces/688-HW/Data/Example_news_info_for_testing.csv"
        if os.path.exists(default_file_path):
            df = pd.read_csv(default_file_path)
            st.write("Using default CSV file.")
        else:
            st.error("Default file not found.")
            st.stop()

    st.write("### Columns in the CSV:", df.columns.tolist())
    st.write("### Sample Data:", df.head())
    # Basic preprocessing (add more as needed)
    df['Document'] = df['Document'].fillna('').astype(str).str.lower()
    return df

def load_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('all-mpnet-base-v2')

def generate_embeddings(df, model, document_col, limit=10):
    """Generate and store embeddings for the first 'limit' number of rows."""
    embedding_file = "news_embeddings.pkl"
    embeddings = []
    progress = st.progress(0)
    
    # Limit the number of rows for testing
    df_limited = df.head(limit)
    total_documents = len(df_limited)
    st.write(f"Generating embeddings for {total_documents} documents...")

    for idx, doc in enumerate(df_limited[document_col].fillna('').tolist()):
        embedding = model.encode(doc, convert_to_tensor=True)
        embeddings.append(embedding)
        progress.progress((idx + 1) / total_documents)

    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)
    st.write(f"Embeddings generation complete for {total_documents} documents and saved to cache.")
    return embeddings

def load_embeddings():
    """Load embeddings from the vector store (pickle file)."""
    embedding_file = "news_embeddings.pkl"
    try:
        with open(embedding_file, "rb") as f:
            embeddings = pickle.load(f)
            st.write("Embeddings loaded from cache.")
            return embeddings
    except FileNotFoundError:
        st.warning("No embeddings found in cache. Please generate embeddings first.")
        return None

def get_most_interesting_news(df):
    """Get the most interesting news stories using OpenAI API."""
    news_list = df['Document'].fillna('').values.tolist()
    prompt = f"""Given the following news stories about recent global events and legal developments, rank them in order of relevance and importance to a large global law firm. Consider factors such as potential legal impact, client relevance, and emerging trends.

    {news_list}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can experiment with other models
        messages=[
            {"role": "system", "content": "You are a helpful and informative news analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_news_by_topic(df, model, topic, similarity_threshold=0.6):
    """Get news stories related to a specific topic."""
    topic_embedding = model.encode(topic, convert_to_tensor=True)
    news_embeddings = load_embeddings()  # Load embeddings from the vector store
    if news_embeddings is None:
        return "Embeddings not available. Please generate embeddings first."

    similarities = cosine_similarity(topic_embedding.reshape(1, -1), news_embeddings)

    # Filter based on similarity threshold
    top_indices = [i for i, sim in enumerate(similarities[0]) if sim > similarity_threshold]
    if top_indices:
        top_news = df.iloc[top_indices]
        # Construct prompt for summarization
        top_news_list = top_news['Document'].fillna('').values.tolist()
        prompt = f"""From the provided news dataset, identify and summarize the top 3 most relevant news stories about {topic}. If no relevant stories are found, say 'No news found about {topic} in the dataset.'

        {top_news_list}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # You can experiment with other models
            messages=[
                {"role": "system", "content": "You are a helpful and informative news analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        return f"No news found about {topic} in the dataset."

def hw06():
    """Main function to run the Streamlit app."""
    openai_api_key = load_api_key()
    openai.api_key = openai_api_key

    # Streamlit UI enhancements
    st.sidebar.title("News Analyzer")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    df = load_data(uploaded_file)  # Load data with the new function
    model = load_embedding_model()

    st.title("News Reporting Bot for Law Firms")

    # Generate embeddings if new data is loaded
    if uploaded_file or os.path.exists("/workspaces/688-HW/Data/Example_news_info_for_testing.csv"):
        if st.button("Generate Embeddings"):
            generate_embeddings(df, model, 'Document')

    # Chatbot-style interaction
    user_input = st.text_input("Ask something (e.g., 'What is the most interesting news?' or 'Tell me about climate change'):")
    
    if user_input:
        if 'interesting' in user_input.lower():
            try:
                result = get_most_interesting_news(df)
                st.write(result)
            except OpenAIError as e:
                st.error(f"An error occurred: {e}")
        else:
            try:
                result = get_news_by_topic(df, model, user_input)
                st.write(result)
            except OpenAIError as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    hw06()