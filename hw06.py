import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, OpenAIError

def hw06():
    # --- Configuration ---
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found in secrets.")
        st.stop()

    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Path to your CSV file
    CSV_FILE = "/workspaces/688-HW/Data/Example_news_info_for_testing.csv"

    # --- Data Loading and Preprocessing ---
    @st.cache_data
    def load_data(csv_file, limit=10):
        df = pd.read_csv(csv_file).head(limit)  # Limit to the first 10 rows for testing
        st.write("### Columns in the CSV:", df.columns.tolist()) 
        st.write("### Sample Data:", df.head())
        return df

    df = load_data(CSV_FILE)  # Load only the first 10 rows for testing

    document_column = 'Document'

    # --- Embedding Generation ---
    @st.cache_resource
    def load_embedding_model():
        model = SentenceTransformer('all-mpnet-base-v2')
        return model

    model = load_embedding_model()

    # Added a progress bar to track embedding generation
    @st.cache_data
    def generate_embeddings(df, _model, document_col):
        embeddings = []
        progress = st.progress(0)  # Initialize the progress bar
        total_documents = len(df)  # Total number of documents to embed
        st.write(f"Generating embeddings for {total_documents} documents...")
        
        for idx, doc in enumerate(df[document_col].fillna('').tolist()):
            # Print the current document index and part of the document content
            st.write(f"Embedding document {idx + 1}/{total_documents}: {doc[:100]}...")  # Show first 100 characters for testing
            
            embedding = _model.encode(doc, convert_to_tensor=True)
            embeddings.append(embedding)
            
            # Update the progress bar
            progress.progress((idx + 1) / total_documents)
        
        st.write("Embeddings generation complete.")
        return embeddings

    # Generating an embeddings for the news data
    news_embeddings = generate_embeddings(df, model, document_column)

    # --- Streamlit App ---
    st.title("News Reporting Bot for Law Firms")

    query_type = st.radio("Select query type:", 
                        ("Most interesting news", "News about a specific topic"))

    if query_type == "Most interesting news":
        if st.button("Get most interesting news"):
            # --- Prompt Engineering for "Most Interesting News" ---
            news_list = df[document_column].fillna('').values.tolist()
            prompt = f"""Given the following news stories about recent global events and legal developments, rank them in order of relevance and importance to a large global law firm. Consider factors such as potential legal impact, client relevance, and emerging trends. Provide a brief explanation for each ranking.

    {news_list}
    """
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Or use gpt-4
                    messages=[
                        {"role": "system", "content": "You are a helpful and informative news analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.write(response.choices[0].message.content)
            except OpenAIError as e:
                st.error(f"An error occurred: {e}")

    elif query_type == "News about a specific topic":
        topic = st.text_input("Enter a topic:")
        if st.button("Search"):
            if not topic:
                st.warning("Please enter a topic to search.")
            else:
                # --- Embedding and Similarity Search ---
                topic_embedding = model.encode(topic, convert_to_tensor=True)
                similarities = cosine_similarity(topic_embedding.reshape(1, -1), news_embeddings)
        
                # --- Getting Top Relevant News ---
                top_indices = similarities.argsort()[0][::-1]  # Sort by similarity in descending order
                top_news = df.iloc[top_indices[:3]]  # Get top 3
        
                # --- Prompt Engineering for "News about a specific topic" ---
                top_news_list = top_news[document_column].fillna('').values.tolist()
                prompt = f"""From the provided news dataset, identify and summarize the top 3 most relevant news stories about {topic}. If no relevant stories are found, say 'No news found about {topic} in the dataset.'

    {top_news_list}
    """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful and informative news analyst."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.write(response.choices[0].message.content)
                except OpenAIError as e:
                    st.error(f"An error occurred: {e}")