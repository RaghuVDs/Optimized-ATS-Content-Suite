import streamlit as st
from openai import OpenAI
import os
import glob
import time
from pathlib import Path

class VectorStoreManager:
    def __init__(self, client):
        self.client = client
        self.vector_store_name = "Course_Documentation"
        self.vector_store = None
        self.assistant = None
        
    def upload_file(self, file_path):
        with open(file_path, "rb") as f:
            return self.client.files.create(
                file=f,
                purpose="assistants"
            )

    def create_or_get_vector_store(self):
        vector_stores = self.client.beta.vector_stores.list().data
        existing_store = next(
            (vs for vs in vector_stores if vs.name == self.vector_store_name), 
            None
        )
        
        if existing_store:
            self.vector_store = existing_store
        else:
            self.vector_store = self.client.beta.vector_stores.create(
                name=self.vector_store_name
            )
        
        return self.vector_store

    def add_files_to_vector_store(self, file_ids):
        if not self.vector_store:
            self.create_or_get_vector_store()
            
        for file_id in file_ids:
            self.client.beta.vector_stores.files.create_and_poll(
                vector_store_id=self.vector_store.id,
                file_id=file_id
            )

    def create_or_get_assistant(self):
        if not self.vector_store:
            self.create_or_get_vector_store()
            
        assistants = self.client.beta.assistants.list().data
        existing_assistant = next(
            (a for a in assistants if "Course Assistant" in str(a.name)), 
            None
        )
        
        if existing_assistant:
            self.assistant = existing_assistant
        else:
            self.assistant = self.client.beta.assistants.create(
                name="Course Assistant",
                instructions="You are a helpful course assistant. Answer questions based on the course materials provided.",
                model="gpt-4-turbo-preview",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [self.vector_store.id]
                    }
                }
            )
        
        return self.assistant

def initialize_session():
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(st.session_state.client)
    if "thread" not in st.session_state:
        st.session_state.thread = st.session_state.client.beta.threads.create()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initial_data_loaded" not in st.session_state:
        st.session_state.initial_data_loaded = False

def load_initial_pdfs():
    data_dir = Path("D:\\3rd Sem\\02.AI APP\\pdffolder")
    if not data_dir.exists():
        st.warning("Data directory not found. Creating it...")
        data_dir.mkdir(exist_ok=True)
        return
        
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        with st.spinner("Loading initial PDF files..."):
            file_ids = []
            for pdf_path in pdf_files:
                file = st.session_state.vector_store_manager.upload_file(str(pdf_path))
                file_ids.append(file.id)
            
            st.session_state.vector_store_manager.add_files_to_vector_store(file_ids)
            st.success(f"Loaded {len(pdf_files)} PDF files from Data directory")


def handle_file_upload(uploaded_files):
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            file_ids = []
            for uploaded_file in uploaded_files:
                # Save the uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Upload to OpenAI
                file = st.session_state.vector_store_manager.upload_file(temp_path)
                file_ids.append(file.id)
                
                # Clean up temp file
                os.remove(temp_path)
            
            st.session_state.vector_store_manager.add_files_to_vector_store(file_ids)
            st.success(f"Successfully processed {len(uploaded_files)} new files")

def get_assistant_response(prompt):
    client = st.session_state.client
    thread = st.session_state.thread
    assistant = st.session_state.vector_store_manager.create_or_get_assistant()
    
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == 'completed':
            break
        time.sleep(1)
    
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value


def hw05():
        st.title("HW5 - Course Information Chatbot")

        initialize_session()

        with st.sidebar:
            st.header("Upload Additional Files")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True,
                type=["pdf"]
            )
            if uploaded_files:
                handle_file_upload(uploaded_files)

        if not st.session_state.initial_data_loaded:
            load_initial_pdfs()
            st.session_state.initial_data_loaded = True

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("What would you like to know?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = get_assistant_response(prompt)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})