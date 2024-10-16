import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, OpenAIError

def load_api_key():
    """Load OpenAI API key from secrets."""
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found in secrets.")
        st.stop()
    return openai_api_key

def load_data(uploaded_file):
    """Load the news data from the uploaded CSV file with a limit of 100 rows."""
    df = pd.read_csv(uploaded_file, nrows=100)  # Load only the first 100 rows
    st.write("### Columns in the CSV:", df.columns.tolist()) 
    st.write("### Sample Data:", df.head())
    return df


def load_embedding_model():
    """Load the sentence transformer model for embedding generation."""
    return SentenceTransformer('all-mpnet-base-v2')

def generate_embeddings(df, model, document_col):
    """Generate embeddings for the news documents."""
    embeddings = []
    progress = st.progress(0)
    total_documents = len(df)
    st.write(f"Generating embeddings for {total_documents} documents...")

    for idx, doc in enumerate(df[document_col].fillna('').tolist()):
        embedding = model.encode(doc, convert_to_tensor=True)
        embeddings.append(embedding)

        # Update the progress bar
        progress.progress((idx + 1) / total_documents)
    
    st.write("Embeddings generation complete.")
    return embeddings

def get_most_interesting_news(df, client):
    """Get the most interesting news stories using OpenAI API."""
    news_list = df['Document'].fillna('').values.tolist()
    prompt = f"""Given the following news stories about recent global events and legal developments, rank them in order of relevance and importance to a large global law firm. Consider factors such as potential legal impact, client relevance, and emerging trends. Provide a brief explanation for each ranking.

    {news_list}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and informative news analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_news_by_topic(df, model, topic, client):
    """Get news stories related to a specific topic."""
    topic_embedding = model.encode(topic, convert_to_tensor=True)
    news_embeddings = generate_embeddings(df, model, 'Document')
    similarities = cosine_similarity(topic_embedding.reshape(1, -1), news_embeddings)
    
    # Getting top relevant news
    top_indices = similarities.argsort()[0][::-1]
    top_news = df.iloc[top_indices[:3]]
    
    # Construct prompt for summarization
    top_news_list = top_news['Document'].fillna('').values.tolist()
    prompt = f"""From the provided news dataset, identify and summarize the top 3 most relevant news stories about {topic}. If no relevant stories are found, say 'No news found about {topic} in the dataset.'

    {top_news_list}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and informative news analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def hw06():
    """Main function to run the Streamlit app."""
    openai_api_key = load_api_key()
    client = OpenAI(api_key=openai_api_key)

    # Streamlit file uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        model = load_embedding_model()

        st.title("News Reporting Bot for Law Firms")

        query_type = st.radio("Select query type:", 
                               ("Most interesting news", "News about a specific topic"))

        if query_type == "Most interesting news":
            if st.button("Get most interesting news"):
                try:
                    result = get_most_interesting_news(df, client)
                    st.write(result)
                except OpenAIError as e:
                    st.error(f"An error occurred: {e}")

        elif query_type == "News about a specific topic":
            topic = st.text_input("Enter a topic:")
            if st.button("Search"):
                if not topic:
                    st.warning("Please enter a topic to search.")
                else:
                    try:
                        result = get_news_by_topic(df, model, topic, client)
                        st.write(result)
                    except OpenAIError as e:
                        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    hw06()
