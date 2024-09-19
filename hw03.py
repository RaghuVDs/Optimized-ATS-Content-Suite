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

            #Placeholder for Google Gemini and Anthropic model selection
            #Comment out these blocks if you only want to use OpenAI for now
            elif selected_model == "Google Gemini":
                st.warning("Google Gemini integration is not yet implemented.")
                gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]  # Placeholder model names
                model_name = st.selectbox("Select Gemini Model", gemini_models)
                encoding = tiktoken.get_encoding("cl100k_base")  # Explicitly use cl100k_base for Gemini

            elif selected_model == "Anthropic":
                st.warning("Anthropic integration is not yet implemented.")
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
            if model_name:  # Only calculate if a model is selected
                total_tokens += len(encoding.encode("You are a helpful AI assistant."))

            # Placeholder to potentially incorporate URLs into the system message
            if url1 or url2:
                system_message = "You are a helpful AI assistant. "
                if url1:
                    system_message += f"You have access to information from this webpage: {url1}. "
                if url2:
                    system_message += f"You also have access to information from this webpage: {url2}. "
                if model_name:  # Only calculate if a model is selected
                    total_tokens += len(encoding.encode(system_message))

            # Display token count information
            if model_name:  # Only display if a model is selected
                st.write(f"Total tokens used for this request: {total_tokens}")
                if total_tokens > max_tokens:
                    st.warning(f"Conversation buffer truncated to fit within {max_tokens} tokens.")

            # Construct the full message history for context
            messages_for_request = [
                {"role": "system", "content": system_message}  # Use the potentially updated system_message
            ] + conversation_buffer

            # Handle different model providers
            if selected_model == "OpenAI":
                # Stream the response for OpenAI
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

            # Comment out the following blocks until official integrations are available
            elif selected_model == "Google Gemini":
                    # Configure Google Generative AI
                    genai.configure(api_key=google_api_key)

                    # Create a GenerativeModel instance
                    model = genai.GenerativeModel(model_name=model_name)

                    # Generate content
                    response = model.generate_content(prompt)

                    # Extract the generated text from the response
                    full_response = response.text

            elif selected_model == "Anthropic":
                    message = anthropic_client.completions.create(
                        model=model_name,  # Use the selected Anthropic model
                        max_tokens_to_sample=1024, 
                        messages=[
                            {"role": "user", "content": prompt}  # The user's prompt or query
                        ]
                    )
                    full_response = message.completion  # Extract the generated text


                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # Flag to control whether to ask for more information
                    ask_for_more_information = True

            # Callback function to handle "Yes" button click
                    def on_yes_click():

                        nonlocal ask_for_more_information
                        st.session_state.messages.append({"role": "user", "content": "Can you give more information on that?"})
                        ask_for_more_information = True

                    # Callback function to handle "No" button click
                    def on_no_click():
                        nonlocal ask_for_more_information
                        st.session_state.messages.append({"role": "assistant", "content": "What question can I help you with?"})
                        ask_for_more_information = False  # Exit the loop

                    while ask_for_more_information:

                        # Trigger another API call to get the elaboration
                        conversation_buffer = st.session_state.messages[-4:]
                        messages_for_request = [
                            {"role": "system", "content": system_message}  # Use the potentially updated system_message
                        ] + conversation_buffer

                        # Stream the elaboration
                        response_container = st.empty()
                        full_response = ""
                        if selected_model == "OpenAI" and model_name:
                            for chunk in client.chat.completions.create(
                                model=model_name,
                                messages=messages_for_request,
                                stream=True
                            ):
                                content = chunk.choices[0].delta.content or ""
                                full_response += content
                                response_container.markdown(full_response + "▌")

                        #Comment out the following blocks until official integrations are available
                        elif selected_model == "Google Gemini":
                            # Google Gemini API call
                            try:
                                ##Configure Google Generative AI (already done earlier)
                                genai.configure(api_key=google_api_key)

                                #Create a GenerativeModel instance (if needed, otherwise reuse the existing one)
                                model = genai.GenerativeModel(model_name=model_name)

                                # Generate content
                                response = model.generate_content(
                                    "Can you give more information on that?",  # Use the appropriate prompt for elaboration
                                    # Add other parameters as needed (e.g., temperature, max_tokens)
                                )

                                # Extract the generated text from the response
                                full_response = response.text

                            except Exception as e:  # Catch any potential errors during the API call
                                st.error(f"Error calling Google Gemini API: {e}")
                                full_response = "An error occurred while processing your request."

                        elif selected_model == "Anthropic":
                            # Anthropic API call 
                            try:
                                # Construct the messages for the elaboration request
                                elaboration_messages = [
                                    {"role": "system", "content": system_message},
                                ] + conversation_buffer  # Include previous messages for context

                                message = anthropic_client.completions.create(
                                    model=model_name,  # Use the selected Anthropic model
                                    max_tokens_to_sample=1024, 
                                    messages=elaboration_messages
                                )
                                full_response = message.completion  # Extract the generated text

                            except Exception as e:  # Catch any potential errors during the API call
                                st.error(f"Error calling Anthropic API: {e}")
                                full_response = "An error occurred while processing your request."

                        response_container.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                        # Ask if the user wants more information
                        st.markdown("**Do you want more information?**")
                        col1, col2 = st.columns(2)

                        yes_button_key = str(uuid.uuid4())
                        no_button_key = str(uuid.uuid4())

                        with col1:
                            st.button("Yes", on_click=on_yes_click, key=yes_button_key)

                        with col2:
                            if st.button("No", on_click=on_no_click, key=no_button_key):
                                pass  # No need for additional logic here

                        ask_for_more_information = False  # Reset the flag at the end of the loop

    except AuthenticationError:
        st.error(
            "Invalid OpenAI API key. "
            "Please double-check your key and ensure it's correct."
        )

        # If you're running this script directly, you might need this line
        if __name__ == "__main__":
            lab3()