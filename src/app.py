# src/app.py
import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer
import time

# Import local modules
from pitchdeck_splitter import pdf_to_images
from slide_description_gen import describe_slides_in_folder

# --- Configuration ---
PERSIST_DIR = "./data/VectorStoreIndex/RAG"
PITCHDECKS_DIR = "./data/pitchdecks"
SLIDES_OUTPUT_DIR = "./data/slides_output"

# Ensure directories exist
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PITCHDECKS_DIR, exist_ok=True)
os.makedirs(SLIDES_OUTPUT_DIR, exist_ok=True)

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="VC Pitch Deck Analyzer & RAG Chat")

# --- Helper Functions ---
@st.cache_resource(show_spinner=False)
def get_openai_api_key():
    """Retrieves the OpenAI API key from Streamlit secrets."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API Key not found. Please add it to your `.streamlit/secrets.toml` file.")
        st.stop()
    return api_key

@st.cache_resource(show_spinner=False)
def initialize_rag_components(api_key: str):
    """
    Initializes the RAG vector index and chat engine.
    Loads existing index if available, otherwise creates an empty one.
    """
    os.environ["OPENAI_API_KEY"] = api_key

    system_prompt_chat = (
        "You are an AI assistant specialized in venture capital and startup evaluation. "
        "You will answer questions based on the provided context from pitch deck slides "
        "and your general knowledge. If the context does not contain the answer, "
        "state that you don't have enough information from the provided documents. "
        "Be concise and professional. Do not hallucinate."
    )

    llm = LI_OpenAI(temperature=0, model="gpt-4o", api_key=api_key)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

    vector_index = None
    chat_engine = None

    try:
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            with st.spinner("Loading existing RAG knowledge base..."):
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                vector_index = load_index_from_storage(storage_context, llm=llm)
            st.success("RAG knowledge base loaded successfully!")
        else:
            with st.spinner("Initializing new RAG knowledge base..."):
                vector_index = VectorStoreIndex.from_documents([], llm=llm)
                vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.info("New RAG knowledge base initialized.")

        chat_engine = vector_index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            llm=llm,
            system_prompt=system_prompt_chat,
            verbose=True,
            similarity_top_k=20,
        )
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        st.warning("The RAG chat might not function correctly.")

    return vector_index, chat_engine

def evaluate_startup_criteria_raw(slide_descriptions_text: str, openai_api_key: str) -> str:
    """
    Evaluates a startup based on aggregated slide descriptions against specific criteria.
    Returns the raw JSON string.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key
    evaluation_llm = LI_OpenAI(temperature=0.2, model="gpt-4o", api_key=openai_api_key)

    prompt = f"""
    You are an expert Venture Capital analyst. Based on the following aggregated descriptions of pitch deck slides,
    evaluate the startup against the following criteria. **Your response MUST be a valid JSON object ONLY**,
    with no preamble, no markdown formatting (e.g., ```json), and no extra text outside the JSON.

    Aggregated Slide Descriptions:
    {slide_descriptions_text}

    Evaluation Criteria:
    1.  **Funding Round**: What funding round is the startup seeking or currently in (e.g., Seed, Series A, Series B, etc.)? If not explicitly stated, infer based on the stage (e.g., early traction suggests Seed/Pre-Seed, significant revenue suggests Series A/B).
    2.  **Region**: What is the primary geographical region or target market of the startup (e.g., San Francisco, Europe, Global, specific countries)?
    3.  **Category**: What is the primary industry or category of the startup (e.g., SaaS, FinTech, AI, Healthcare, E-commerce, Deep Tech)?
    4.  **Excluded Fields**: List any areas or aspects that are explicitly stated as *not* being part of the startup's focus, product, or strategy. If none are explicitly excluded, state "None explicitly mentioned."

    Provide the response in a JSON format with keys: "funding_round", "region", "category", "excluded_fields".
    Example JSON structure:
    {{
        "funding_round": "Seed",
        "region": "Global, focusing on North America",
        "category": "AI-powered B2B SaaS for Supply Chain",
        "excluded_fields": ["direct-to-consumer sales", "hardware manufacturing"]
    }}
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = evaluation_llm.complete(prompt)
            if response.text.strip():
                return response.text.strip()
            else:
                print(f"Attempt {attempt+1}: AI response was empty.")
        except Exception as e:
            print(f"Attempt {attempt+1}: Error during AI evaluation: {e}")
        time.sleep(1)
    return ""

# --- Streamlit UI ---

# Initialize session state variables if they don't exist
if "messages" not in st.session_state: # Fix: corrected 'not not' to 'not in'
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a pitch deck PDF to get started, or ask me anything about venture capital."}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

st.title("VC Pitch Deck Analyzer & RAG Chat")

# Get OpenAI API Key
openai_api_key = get_openai_api_key()

# Initialize RAG components if not already done
if st.session_state.vector_index is None or st.session_state.chat_engine is None:
    st.session_state.vector_index, st.session_state.chat_engine = initialize_rag_components(openai_api_key)


# --- NEW: Tabbed Interface ---
tab1, tab2 = st.tabs(["Upload & Analyze Pitch Deck", "RAG Chat"])

with tab1:
    st.header("Upload & Analyze Pitch Deck")
    uploaded_file = st.file_uploader("Upload a Pitch Deck PDF", type="pdf")

    # Use a unique key for each upload (e.g., file name + size)
    file_key = None
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_file_key") != file_key:
            # New file uploaded, reset state
            st.session_state["last_file_key"] = file_key
            st.session_state["image_paths"] = None
            st.session_state["current_pitch_deck_documents"] = None

    if uploaded_file and st.session_state.get("image_paths") is None:
        st.session_state.messages.append({"role": "user", "content": f"Uploaded: {uploaded_file.name}"})
        st.write(f"Processing '{uploaded_file.name}'...")

        base_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_base_name = f"{base_name}_{timestamp}"

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info("Splitting PDF into images...")
            current_slides_output_dir = os.path.join(temp_dir, "current_slides")
            os.makedirs(current_slides_output_dir, exist_ok=True)
            image_paths = pdf_to_images(pdf_path, current_slides_output_dir, unique_base_name)
            st.success(f"PDF split into {len(image_paths)} images.")

            if image_paths:
                st.info("Generating descriptions for each slide using AI...")
                current_pitch_deck_documents = describe_slides_in_folder(current_slides_output_dir, openai_api_key)
                st.success(f"Generated descriptions for {len(current_pitch_deck_documents)} slides.")

                if current_pitch_deck_documents:
                    st.subheader("Slide Descriptions (First 100 chars):")
                    for i, doc in enumerate(current_pitch_deck_documents):
                        st.text(f"Slide {doc.metadata.get('page_number', i+1)}: {doc.text[:100]}...")

                    print(f"DEBUG: Number of documents generated for RAG: {len(current_pitch_deck_documents)}")

                    st.info("Adding new slide descriptions to the RAG knowledge base...")
                    try:
                        for doc in current_pitch_deck_documents:
                            st.session_state.vector_index.insert(doc)
                        st.session_state.vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
                        st.success("New slide descriptions successfully added to the RAG knowledge base!")

                        # --- Show evaluation criteria ---
                        st.markdown("""
**Automated Evaluation Criteria:**

1. **Funding Round**: What funding round is the startup seeking or currently in (e.g., Seed, Series A, Series B, etc.)? If not explicitly stated, infer based on the stage (e.g., early traction suggests Seed/Pre-Seed, significant revenue suggests Series A/B).
2. **Region**: What is the primary geographical region or target market of the startup (e.g., San Francisco, Europe, Global, specific countries)?
3. **Category**: What is the primary industry or category of the startup (e.g., SaaS, FinTech, AI, Healthcare, E-commerce, Deep Tech)?
4. **Excluded Fields**: List any areas or aspects that are explicitly stated as *not* being part of the startup's focus, product, or strategy. If none are explicitly excluded, state "None explicitly mentioned."
""")
                        # --- End criteria display ---

                        st.subheader("Automated Startup Evaluation")
                        aggregated_descriptions = "\n\n".join([doc.text for doc in current_pitch_deck_documents])
                        
                        raw_evaluation_response_text = ""

                        with st.spinner("Running AI evaluation..."):
                            raw_evaluation_response_text = evaluate_startup_criteria_raw(aggregated_descriptions, openai_api_key)
                        
                        if raw_evaluation_response_text:
                            try:
                                evaluation_result = json.loads(raw_evaluation_response_text)
                                st.json(evaluation_result)
                            except json.JSONDecodeError as e:
                                st.error(f"Failed to parse AI evaluation response as JSON: {e}")
                                st.text_area("Raw AI Evaluation Response (for debugging):", raw_evaluation_response_text, height=200)
                                st.warning("Could not complete automated evaluation due to JSON parsing error.")
                            except Exception as e:
                                st.error(f"An unexpected error occurred after evaluation: {e}")
                                st.text_area("Raw AI Evaluation Response (for debugging):", raw_evaluation_response_text, height=200)
                        else:
                            st.warning("AI evaluation returned an empty or invalid response.")
                            st.text_area("Raw AI Evaluation Response (for debugging):", raw_evaluation_response_text, height=100)
                    except Exception as e:
                        st.error(f"Error updating RAG knowledge base: {e}")
    else:
        st.info("Upload a PDF pitch deck to begin analysis and evaluation.")

with tab2:
    st.header("Cumulative RAG Chat")

    if st.session_state.chat_engine:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me about the pitch decks or venture capital..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.warning("RAG chat engine is not initialized. Please check API key and ensure index can be loaded/created.")

st.markdown("---")
st.markdown("Developed with ❤️ by cosmincbodea")