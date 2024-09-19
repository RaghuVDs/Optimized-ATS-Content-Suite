import streamlit as st
from openai import OpenAI
from openai import AuthenticationError
import tiktoken
import uuid
import google.generativeai as genai
import anthropic

def lab3():

    # Initialize chat history if not present in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Title and description
    st.title("🤖 Chat with GPT")
    st.markdown("Ask me anything!")

    try:
        # API key handling for OpenAI
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        if not openai_api_key:
            st.error("OpenAI API key not found in secrets.")
            st.stop()

        # API key handling for Google Gemini 
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key:
            st.error("Google API key not found in secrets.")
            st.stop()

        # API key handling for Anthropic
        anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
        if not anthropic_api_key:
            st.error("Anthropic API key not found in secrets.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)

        # Sidebar with model choice AND URL inputs
        with st.sidebar:
            st.subheader("Model Options")

            # Add a selectbox for model selection
            selected_model = st.selectbox("Select Model Provider", ["OpenAI", "Google Gemini", "Anthropic"])

            if selected_model == "OpenAI":
                # OpenAI model selection
                openai_models = ["gpt-4o", "gpt-4o-mini"]  # Add more OpenAI models as needed
                model_name = st.selectbox("Select OpenAI Model", openai_models)
                encoding = tiktoken.encoding_for_model(model_name)  # Explicitly get tokenizer for OpenAI

            elif selected_model == "Google Gemini":
                gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]  # Placeholder model names
                model_name = st.selectbox("Select Gemini Model", gemini_models)
                encoding = tiktoken.get_encoding("cl100k_base")  # Explicitly use cl100k_base for Gemini

            elif selected_model == "Anthropic":
                anthropic_models = ["Claude 3 Haiku", "Claude 3.5 Sonnet"]  # Placeholder model names
                model_name = st.selectbox("Select Anthropic Model", anthropic_models)
                encoding = tiktoken.get_encoding("cl100k_base")  # Explicitly use cl100k_base for Anthropic

            # URL input fields
            st.subheader("URLs (Optional)")
            url1 = st.text_input("Enter URL 1:")
            url2 = st.text_input("Enter URL 2:")

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input for questions
        if prompt := st.chat_input("You:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Calculate token count for the new prompt
            if model_name:  # Only calculate if a model is selected
                new_prompt_tokens = len(encoding.encode(prompt))

            # Maintain a conversation buffer, limiting it to 'max_tokens'
            max_tokens = 3000  # Adjust this as needed
            conversation_buffer = []
            total_tokens = 0
            for message in reversed(st.session_state.messages):
                if model_name:  # Only calculate if a model is selected
                    message_tokens = len(encoding.encode(message["content"]))
                else:
                    message_tokens = 0  # Placeholder for other models
                if total_tokens + message_tokens > max_tokens:
                    break
                conversation_buffer.insert(0, message)
                total_tokens += message_tokens

            # Add system message tokens
            total_tokens += len(encoding.encode("You are a helpful AI assistant."))

            # Placeholder to potentially incorporate URLs into the system message
            system_message = "You are a helpful AI assistant. "
            if url1:
                system_message += f"You have access to information from this webpage: {url1}. "
            if url2:
                system_message += f"You also have access to information from this webpage: {url2}. "
            total_tokens += len(encoding.encode(system_message))

            # Display token count information
            st.write(f"Total tokens used for this request: {total_tokens}")
            if total_tokens > max_tokens:
                st.warning(f"Conversation buffer truncated to fit within {max_tokens} tokens.")

            # Construct the full message history for context
            messages_for_request = [
                {"role": "system", "content": system_message}
            ] + conversation_buffer

            # Handle different model providers
            if selected_model == "OpenAI":
                # OpenAI API call
                response_container = st.empty()
                full_response = ""
                for chunk in client.chat.completions.create(
                    model=model_name,
                    messages=messages_for_request,
                    stream=True
                ):
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)

            elif selected_model == "Google Gemini":
                # Google Gemini API call
                try:
                    genai.configure(api_key=google_api_key)
                    model = genai.GenerativeModel(model_name=model_name)
                    response = model.generate_content(prompt)
                    full_response = response.text
                    st.markdown(full_response)
                except Exception as e:  # Catch any potential errors during the API call
                    st.error(f"Error calling Google Gemini API: {e}")
                    full_response = "An error occurred while processing your request."

            elif selected_model == "Anthropic":
                # Anthropic API call 
                try:
                    anthropic_client = anthropic.Client(api_key=anthropic_api_key)
                    message = anthropic_client.completions.create(
                        model=model_name,
                        max_tokens_to_sample=1024, 
                        messages=[{"role": "user", "content": prompt}]
                    )
                    full_response = message.completion
                    st.markdown(full_response)
                except Exception as e:  # Catch any potential errors during the API call
                    st.error(f"Error calling Anthropic API: {e}")
                    full_response = "An error occurred while processing your request."

            # Append the response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    except AuthenticationError:
        st.error(
            "Invalid OpenAI API key. "
            "Please double-check your key and ensure it's correct."
        )

# If you're running this script directly, you might need this line
if __name__ == "__main__":
    lab3()