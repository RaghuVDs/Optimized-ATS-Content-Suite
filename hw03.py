import streamlit as st
from openai import OpenAI
from openai import AuthenticationError
from tiktoken import encoding_for_model
import uuid

def lab3():

    # Initialize chat history if not present in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Title and description
    st.title("🤖 Chat with GPT")
    st.markdown("Ask me anything!")

    # API key handling with a more helpful error message
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not openai_api_key:
        st.error(
            "OpenAI API key not found in secrets. "
            "Make sure you've added it to your Streamlit secrets file."
        )
        st.stop()

    try:
        client = OpenAI(api_key=openai_api_key)

        # Sidebar with model choice
        with st.sidebar:
            st.subheader("Model Options")
            use_advanced_model = st.checkbox("Use Advanced Model (gpt-4)")
            model_name = "gpt-4" if use_advanced_model else "gpt-3.5-turbo"

            # Initialize encoding after model_name is assigned
            encoding = encoding_for_model(model_name)

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
            new_prompt_tokens = len(encoding.encode(prompt))

            # Maintain a conversation buffer, limiting it to 'max_tokens'
            max_tokens = 3000  # Adjust this as needed
            conversation_buffer = []
            total_tokens = 0
            for message in reversed(st.session_state.messages):
                message_tokens = len(encoding.encode(message["content"]))
                if total_tokens + message_tokens > max_tokens:
                    break
                conversation_buffer.insert(0, message)
                total_tokens += message_tokens

            # Add system message tokens
            total_tokens += len(encoding.encode("You are a helpful AI assistant."))

            # Display token count information
            st.write(f"Total tokens used for this request: {total_tokens}")
            if total_tokens > max_tokens:
                st.warning(f"Conversation buffer truncated to fit within {max_tokens} tokens.")

            # Construct the full message history for context
            messages_for_request = [
                {"role": "system", "content": "You are a helpful AI assistant."}
            ] + conversation_buffer

            # Stream the response
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
                {"role": "system", "content": "You are a helpful AI assistant."}
            ] + conversation_buffer

            # Stream the elaboration
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

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Ask if the user wants more information
            st.markdown("**Do you want more information?**")
            col1, col2 = st.columns(2)

            # Generate unique keys for the buttons
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