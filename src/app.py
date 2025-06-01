import streamlit as st
import os
import openai
import fitz
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import tempfile
import json
import re
import sys

from pitchdeck_splitter import pdf_to_images
from slide_description_gen import describe_image
from pitchdeck_evaluator import evaluate_startup

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- API Key Handling (Recommended: Use Streamlit secrets) ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.warning("OPENAI_API_KEY not found in Streamlit secrets. Attempting to use environment variable.")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# --- Streamlit Config ---
st.set_page_config(page_title="RAG Chat for VC Pitch decks", page_icon=" 💼 ", layout="wide")
st.title(" 💼  RAG Chat & Evaluation for VC Pitch decks")

# --- Constants ---
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT_CHAT = (
    "You are a helpful assistant for the investment team of a venture capital fund called UVC Partners. "
    "Answer with the most amount of information you can find in the provided context."
)

# --- Define Persistent Directories ---
DATA_DIR = "./data"
PITCHDECKS_DIR = "./data/uploaded_pitchdecks"
SLIDES_DIR = "./data/slides"
DOCS_DIR = "./data/docs"
PERSIST_INDEX_DIR = "./data/vector_store_index"

# --- Ensure persistent directories exist ---
for directory in [DATA_DIR, PITCHDECKS_DIR, SLIDES_DIR, DOCS_DIR, PERSIST_INDEX_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Initialize memory & multimodal LLM (cached for performance) ---
@st.cache_resource
def get_chat_memory_buffer():
    """Caches and returns a ChatMemoryBuffer for the chat engine."""
    return ChatMemoryBuffer.from_defaults(token_limit=10000)

@st.cache_resource
def get_openai_multimodal_llm():
    """Caches and returns an OpenAIMultiModal LLM instance."""
    return OpenAIMultiModal(model=MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])

memory = get_chat_memory_buffer()
openai_mm_llm = get_openai_multimodal_llm()

# --- Initialize session state for cumulative index and chat ---
if "cumulative_vector_index_object" not in st.session_state:
    st.session_state.cumulative_vector_index_object = None # The LlamaIndex object
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None # The chat engine derived from the index

def load_or_init_chat_engine_dynamic(msg_container):
    """
    Loads an existing LlamaIndex VectorStoreIndex from persistence, or initializes a new one.
    All status messages are displayed in the provided Streamlit container.
    """
    if st.session_state.cumulative_vector_index_object is None:
        # Attempt to load if not already loaded in session
        if os.path.exists(PERSIST_INDEX_DIR) and len(os.listdir(PERSIST_INDEX_DIR)) > 0:
            # msg_container.info(f"Loading existing chat engine from: {PERSIST_INDEX_DIR}")
            try:
                vec_ctx = StorageContext.from_defaults(persist_dir=PERSIST_INDEX_DIR)
                idx = load_index_from_storage(vec_ctx)
                st.session_state.cumulative_vector_index_object = idx
                st.session_state.chat_engine = idx.as_chat_engine(
                    chat_mode="context", memory=memory, llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
                    system_prompt=SYSTEM_PROMPT_CHAT, verbose=True, similarity_top_k=20,
                )
                msg_container.success("RAG chat engine loaded from persisted index.")
            except Exception as e:
                msg_container.error(f"Failed to load chat engine from '{PERSIST_INDEX_DIR}': {e}. Creating a new empty index.")
                st.session_state.cumulative_vector_index_object = VectorStoreIndex([])
                st.session_state.chat_engine = st.session_state.cumulative_vector_index_object.as_chat_engine(
                    chat_mode="context", memory=memory, llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
                    system_prompt=SYSTEM_PROMPT_CHAT, verbose=True, similarity_top_k=20
                )
        else:
            msg_container.info(f"No existing chat engine found at. Creating a new empty index.")
            st.session_state.cumulative_vector_index_object = VectorStoreIndex([])
            st.session_state.chat_engine = st.session_state.cumulative_vector_index_object.as_chat_engine(
                chat_mode="context", memory=memory, llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
                system_prompt=SYSTEM_PROMPT_CHAT, verbose=True, similarity_top_k=20
            )
    if st.session_state.cumulative_vector_index_object and st.session_state.chat_engine is None:
        st.session_state.chat_engine = st.session_state.cumulative_vector_index_object.as_chat_engine(
            chat_mode="context", memory=memory, llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
            system_prompt=SYSTEM_PROMPT_CHAT, verbose=True, similarity_top_k=20,
        )

# --- Initialize remaining session state variables ---
if "slides_info" not in st.session_state:
    st.session_state.slides_info = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "history" not in st.session_state:
    st.session_state.history = []
if "current_conditions_prompt_display" not in st.session_state:
    st.session_state.current_conditions_prompt_display = None

# --- Helper Functions for Processing ---
def process_pdf_to_images_app(pdf_path: str, slides_dir: str, base_name: str) -> list[dict]:
    """
    Converts a PDF to a series of images and saves them to the specified directory.
    Returns a list of dictionaries with 'path' and 'page' info for each slide.
    """
    os.makedirs(slides_dir, exist_ok=True)
    pdf_doc = fitz.open(pdf_path)
    current_slides_info = []
    if not pdf_doc.page_count:
        st.warning("The uploaded PDF has no pages.")
        return []
    for i in range(len(pdf_doc)):
        try:
            pix = pdf_doc.load_page(i).get_pixmap()
            # Ensure consistent naming for slides, especially for doc_id later
            slide_name_suffix = f"_slide_{i + 1:02d}.png" # e.g., _slide_01.png, _slide_10.png
            slide_path = os.path.join(slides_dir, f"{base_name}{slide_name_suffix}")
            pix.save(slide_path)
            current_slides_info.append({"path": slide_path, "page": i + 1, "base_name": base_name})
        except Exception as e_slide:
            st.error(f"Error processing slide {i+1}: {e_slide}")
            current_slides_info.append({"path": None, "desc": f"Error generating image for slide {i+1}.", "page": i+1, "error": True, "base_name": base_name})
    pdf_doc.close()
    return current_slides_info

def describe_slides_app(slides_info: list[dict], openai_mm_llm: OpenAIMultiModal) -> list[dict]:
    """
    Generates descriptions for each slide image within the Streamlit context using a multimodal LLM.
    Updates the 'slides_info' list with the generated descriptions.
    """
    progress_bar = st.progress(0, text="Generating slide descriptions...")
    total = len([info for info in slides_info if info["path"] and not info.get("error")])
    completed = 0
    for info in slides_info:
        if info["path"] and not info.get("error"):
            try:
                description = describe_image(info["path"], openai_mm_llm)
                info["desc"] = description
            except Exception as e_desc:
                st.error(f"Error generating description for slide {info['page']}: {e_desc}")
                info["desc"] = f"Error generating description for slide {info['page']}."
                info["error"] = True
            completed += 1
            progress_bar.progress(completed / total, text=f"Generating slide descriptions... ({completed}/{total})")
    progress_bar.empty()  # Remove the progress bar when done
    return slides_info

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📤 Upload & Analyze Pitch Deck", "💬 Chat with General Knowledge Base"])

with tab1:
    st.header(" 📤  Upload & Analyze Pitch Deck")
    uploaded_file = st.file_uploader("Select a PDF pitch deck", type="pdf", key="pdf_uploader")
    if uploaded_file:
        # Only process if a new file is uploaded
        if st.session_state.uploaded_file_name != uploaded_file.name:
            # Reset states for the new upload
            st.session_state.slides_info = []
            st.session_state.evaluation_results = None
            st.session_state.current_conditions_prompt_display = None # Clear old prompt display
            st.session_state.uploaded_file_name = uploaded_file.name

            with st.spinner(f"Processing {uploaded_file.name}..."):
                # 1. Save uploaded PDF to data/uploaded_pitchdecks/
                pdf_path = os.path.join(PITCHDECKS_DIR, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"Uploaded pitch deck!")

                base_file_name = os.path.splitext(uploaded_file.name)[0]

                # 2. Process PDF to images and save to data/slides/
                # Create a specific sub-folder for this pitch deck's slides to avoid naming conflicts
                current_slides_output_dir = os.path.join(SLIDES_DIR, base_file_name)
                os.makedirs(current_slides_output_dir, exist_ok=True)
                current_slides_info = process_pdf_to_images_app(pdf_path, current_slides_output_dir, base_file_name)
                st.session_state.slides_info = current_slides_info # Update session state

                # 3. Describe slides
                if st.session_state.slides_info:
                    st.session_state.slides_info = describe_slides_app(st.session_state.slides_info, openai_mm_llm)

                    # 4. Save generated descriptions to data/docs/
                    descriptions_list_for_json = []
                    for info in st.session_state.slides_info:
                        if not info.get("error") and info.get("desc"):
                            # Prepare a dictionary that can be easily saved as JSON
                            descriptions_list_for_json.append({
                                "text": info["desc"],
                                "doc_id": f"{base_file_name}_slide_{info['page']}",
                                "metadata": {
                                    "startup_name": base_file_name,
                                    "page_number": info['page'],
                                    "source_file": uploaded_file.name,
                                    "image_path": info['path'] # Include image path for reference
                                }
                            })
                    if descriptions_list_for_json:
                        descriptions_json_filename = f"{base_file_name}_descriptions.json"
                        descriptions_json_path = os.path.join(DOCS_DIR, descriptions_json_filename)
                        with open(descriptions_json_path, 'w') as f:
                            json.dump(descriptions_list_for_json, f, indent=4)
                        st.success(f"Generated and saved slide descriptions for {base_file_name}.")
                    else:
                        st.warning("No valid descriptions were generated to save.")

                # 5. Evaluate startup - calling the imported function
                all_descriptions = "\n\n---\n\n".join([
                    f"Content from Slide {info['page']}:\n{info['desc']}"
                    for info in st.session_state.slides_info if not info.get("error") and info.get("desc")
                ])
                if not all_descriptions.strip():
                    st.warning("No valid slide descriptions available to perform startup evaluation.")
                else:
                    st.subheader(" ⚙️  Performing Startup Evaluation...")
                    with st.spinner("Evaluating conditions based on slide descriptions..."):
                        # Calling the imported evaluate_startup function
                        st.session_state.evaluation_results = evaluate_startup(all_descriptions, uploaded_file.name, OPENAI_API_KEY)
                        st.session_state.current_conditions_prompt_display = None # Clear this as the new evaluation is dynamic

                    # --- ADDED: Make the index cumulative ---
                    if st.session_state.cumulative_vector_index_object:
                        # st.info("Adding new slide descriptions to the RAG knowledge base...")
                        new_documents_for_rag = []
                        for info in st.session_state.slides_info:
                            if not info.get("error") and info.get("desc"):
                                # Create a LlamaIndex Document for each slide description
                                # Ensure doc_id is unique across all pitch decks and slides
                                doc_id = f"{base_file_name}_slide_{info['page']}"
                                new_documents_for_rag.append(
                                    Document(
                                        text=info["desc"],
                                        doc_id=doc_id, # Unique ID for each slide
                                        metadata={
                                            "startup_name": base_file_name, # Original file name as startup name
                                            "page_number": info['page'],
                                            "source_file": uploaded_file.name # Original PDF filename
                                        }
                                    )
                                )
                        if new_documents_for_rag:
                            with st.spinner("Updating RAG index with new pitch deck data..."):
                                try:
                                    for doc_to_insert in new_documents_for_rag:
                                        # LlamaIndex's insert method adds new documents incrementally
                                        st.session_state.cumulative_vector_index_object.insert(doc_to_insert)
                                    
                                    # Persist the updated index to disk
                                    st.session_state.cumulative_vector_index_object.storage_context.persist(persist_dir=PERSIST_INDEX_DIR)
                                    st.success(f"Successfully added {len(new_documents_for_rag)} new slide descriptions to the RAG knowledge base and persisted the index.")
                                    
                                    # Re-create chat engine with the updated index object to ensure it reflects changes
                                    st.session_state.chat_engine = st.session_state.cumulative_vector_index_object.as_chat_engine(
                                        chat_mode="context", memory=memory, llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
                                        system_prompt=SYSTEM_PROMPT_CHAT, verbose=True, similarity_top_k=20,
                                    )
                                except Exception as e_insert_persist:
                                    st.error(f"Failed to update or persist the cumulative index: {e_insert_persist}")
                        else:
                            st.warning("No valid documents generated from this pitch deck to add to the RAG index.")
                    else:
                        st.warning("Cumulative index object is not initialized. Cannot add new documents to RAG.")
        # Display results (unchanged in logic, but now expects the new JSON structure)
        if st.session_state.slides_info:
            st.subheader(" 📄  Slide Previews & Descriptions")
            startup_name = st.session_state.slides_info[0].get("base_name", "Startup")
            slide_options = [
                f"Slide {info['page']}" for info in st.session_state.slides_info if info["path"] and not info.get("error")
            ]
            if slide_options:
                selected_slide = st.selectbox(
                    f"Select a slide from {startup_name} to preview and see its description:",
                    slide_options,
                    key="slide_selectbox"
                )
                selected_index = slide_options.index(selected_slide)
                info = [s for s in st.session_state.slides_info if s["path"] and not s.get("error")][selected_index]
                with st.expander(f"{startup_name} - {selected_slide}", expanded=True):
                    st.image(info["path"], use_container_width=True, caption=selected_slide)
                    st.markdown(f"**Description:**\n\n{info['desc']}")
            else:
                st.warning("No valid slides to display.")

        # Move the evaluation section here, after the slides
        if st.session_state.evaluation_results:
            st.subheader(" 📊  Startup Evaluation Results by LLM")
            results = st.session_state.evaluation_results
            # Define the criteria and their corresponding keys
            criteria = [
                ("Funding Round", "funding_round", "What funding round is the startup seeking or currently in?"),
                ("Region", "region", "What is the primary geographical region or target market of the startup?"),
                ("Category", "category", "What is the primary industry or category of the startup (e.g., SaaS, FinTech, AI, Healthcare, E-commerce, Deep Tech)?"),
                ("Excluded Fields", "excluded_fields", "List whether the startup is active in any of the following excluded fields: crypto development, cryptocurrencies, or drug development. If none are mentioned, state 'None explicitly mentioned.'"),
            ]
            if isinstance(results, dict) and "error" in results:
                st.error(f"Evaluation Error: {results.get('error', 'Could not generate evaluation.')}")
                if "raw_llm_response_eval" in results:
                    st.text_area("Problematic Raw Response (if available)", results["raw_llm_response_eval"], height=150)
            elif isinstance(results, dict):
                st.markdown(f"**Startup Name:** {results.get('startup_name', 'N/A')}")
                for idx, (title, key, desc) in enumerate(criteria, 1):
                    st.markdown(f"**{idx}. {title}:** {desc}")
                    st.markdown(f"**Answer:** {results.get(key, 'N/A')}\n")
            else:
                st.error("Evaluation results are in an unexpected format.")
                st.write(results)

with tab2:
    st.header(" 💬  Chat with General Knowledge Base")
    rag_status_container = st.container()
    load_or_init_chat_engine_dynamic(rag_status_container)

    if not st.session_state.chat_engine:
        rag_status_container.warning("Chat engine is not available. Please check configurations and persisted index path.")
    else:
        for msg in st.session_state.history:
            st.chat_message(msg["role"]).write(msg["content"])
        user_input = st.chat_input("Ask about general VC topics or indexed documents...")
        if user_input:
            st.chat_message("user").write(user_input)
            st.session_state.history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(user_input)
                    assistant_text = response.response
                    st.chat_message("assistant").write(assistant_text)
                    st.session_state.history.append({"role": "assistant", "content": assistant_text})
                except Exception as e_chat:
                    rag_status_container.error(f"Error during chat: {e_chat}")
                    st.session_state.history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e_chat}"})