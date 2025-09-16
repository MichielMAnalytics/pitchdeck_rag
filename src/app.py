import streamlit as st
import os
import openai
import fitz
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
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

# --- Feature Toggle Settings ---
SHOW_LLM_EVALUATION = st.secrets.get("SHOW_LLM_EVALUATION", True)  # Default to True if not set

# --- Streamlit Config ---
st.set_page_config(page_title="RAG Chat for VC Pitch decks", page_icon=" 💼 ", layout="wide")
st.title(" 💼  RAG Chat & Evaluation for VC Pitch decks")

# --- Constants ---
MODEL_NAME = "gpt-4o-mini"
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

# Clean up old cumulative vector index files if they exist
old_index_files = ["default__vector_store.json", "docstore.json", "graph_store.json", "image__vector_store.json", "index_store.json"]
for old_file in old_index_files:
    old_path = os.path.join(PERSIST_INDEX_DIR, old_file)
    if os.path.exists(old_path):
        try:
            os.remove(old_path)
        except:
            pass  # Silently ignore if we can't delete

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
    """Caches and returns an OpenAI LLM instance with vision capabilities."""
    return OpenAI(model=MODEL_NAME, api_key=os.environ["OPENAI_API_KEY"])

memory = get_chat_memory_buffer()
openai_mm_llm = get_openai_multimodal_llm()

# --- Helper functions for per-deck vector indexes ---
def create_deck_vector_index(deck_name: str):
    """
    Creates a vector index for a specific pitch deck from its description JSON file.
    """
    descriptions_path = os.path.join(DOCS_DIR, f"{deck_name}_descriptions.json")
    
    if not os.path.exists(descriptions_path):
        return None
    
    try:
        with open(descriptions_path, 'r') as f:
            descriptions_data = json.load(f)
        
        # Create documents for this specific deck
        documents = []
        for item in descriptions_data:
            documents.append(
                Document(
                    text=item["text"],
                    doc_id=item["doc_id"],
                    metadata=item["metadata"]
                )
            )
        
        # Create deck-specific vector index
        vector_index = VectorStoreIndex(documents)
        
        # Persist to deck-specific directory
        deck_index_dir = os.path.join(PERSIST_INDEX_DIR, deck_name)
        os.makedirs(deck_index_dir, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=deck_index_dir)
        
        return vector_index
        
    except Exception as e:
        st.error(f"Error creating vector index for {deck_name}: {e}")
        return None

def load_deck_vector_index(deck_name: str):
    """
    Loads a deck-specific vector index from persistence.
    """
    deck_index_dir = os.path.join(PERSIST_INDEX_DIR, deck_name)
    
    if not os.path.exists(deck_index_dir) or not os.listdir(deck_index_dir):
        return None
    
    try:
        vec_ctx = StorageContext.from_defaults(persist_dir=deck_index_dir)
        return load_index_from_storage(vec_ctx)
    except Exception as e:
        st.error(f"Error loading vector index for {deck_name}: {e}")
        return None

# --- Initialize session state variables ---
if "slides_info" not in st.session_state:
    st.session_state.slides_info = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
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

def describe_slides_app(slides_info: list[dict], openai_mm_llm: OpenAI) -> list[dict]:
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

def delete_pitch_deck(startup_name: str, deck_filename: str):
    """
    Deletes a pitch deck and all its associated files, and removes it from the vector database.
    
    Args:
        startup_name: The base name of the startup (without extension)
        deck_filename: The filename of the PDF file
    """
    import shutil
    
    with st.spinner(f"Deleting {startup_name} and removing from RAG..."):
        try:
            # 1. Delete the PDF file
            pdf_path = os.path.join(PITCHDECKS_DIR, deck_filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                st.success(f"✅ Deleted PDF: {deck_filename}")
            
            # 2. Delete the slides directory
            slides_dir = os.path.join(SLIDES_DIR, startup_name)
            if os.path.exists(slides_dir):
                shutil.rmtree(slides_dir)
                st.success(f"✅ Deleted slide images for {startup_name}")
            
            # 3. Delete the descriptions JSON file
            desc_file = os.path.join(DOCS_DIR, f"{startup_name}_descriptions.json")
            if os.path.exists(desc_file):
                os.remove(desc_file)
                st.success(f"✅ Deleted descriptions for {startup_name}")
            
            # 3b. Delete the evaluation JSON file if it exists
            eval_file = os.path.join(DOCS_DIR, f"{startup_name}_evaluation.json")
            if os.path.exists(eval_file):
                os.remove(eval_file)
                st.success(f"✅ Deleted evaluation for {startup_name}")
            
            # 4. Remove deck-specific vector index
            deck_index_dir = os.path.join(PERSIST_INDEX_DIR, startup_name)
            if os.path.exists(deck_index_dir):
                try:
                    shutil.rmtree(deck_index_dir)
                    st.success(f"✅ Deleted vector index for {startup_name}")
                except Exception as e:
                    st.error(f"Error deleting vector index: {e}")
                
            st.success(f"🎉 Successfully deleted {startup_name} and all associated data!")
            
        except Exception as e:
            st.error(f"Error during deletion: {e}")
            st.error("Some files may not have been deleted. Please check manually.")

# --- Main Content ---
st.header(" 📤  Upload & Analyze Pitch Deck")

# Display already uploaded pitch decks
existing_pitchdecks = []
if os.path.exists(PITCHDECKS_DIR):
    existing_pitchdecks = [f for f in os.listdir(PITCHDECKS_DIR) if f.endswith('.pdf')]

if existing_pitchdecks:
    st.subheader(" 📚  Already Uploaded Pitch Decks")
    # Create a more visually appealing display
    cols = st.columns(3)  # Create 3 columns for better layout
    for idx, deck in enumerate(sorted(existing_pitchdecks)):
        col = cols[idx % 3]
        with col:
            # Extract startup name from filename
            startup_name = os.path.splitext(deck)[0]
            # Check if descriptions exist for this deck
            desc_file = os.path.join(DOCS_DIR, f"{startup_name}_descriptions.json")
            has_descriptions = os.path.exists(desc_file)
            
            # Check if vector index exists for this deck (chat functionality)
            deck_index_dir = os.path.join(PERSIST_INDEX_DIR, startup_name)
            has_chat = os.path.exists(deck_index_dir) and os.listdir(deck_index_dir)
            
            # Create a card-like display
            with st.container():
                st.markdown(f"**📄 {startup_name}**")
                if has_descriptions and has_chat:
                    st.caption("✅ Ready with Chat")
                elif has_descriptions:
                    st.caption("⚠️ Descriptions only")
                else:
                    st.caption("⏳ Not processed")
                # Get file size
                file_size = os.path.getsize(os.path.join(PITCHDECKS_DIR, deck))
                st.caption(f"Size: {file_size / 1024:.1f} KB")
                
                # Add View and Delete buttons
                button_col1, button_col2 = st.columns(2)
                with button_col1:
                    if st.button(f"👁️ View", key=f"view_{startup_name}", 
                               type="primary", width="stretch",
                               help=f"View {startup_name} slides"):
                        # Store in session state and navigate
                        st.session_state.selected_deck = startup_name
                        st.switch_page("pages/2_📊_Pitch_Deck_Viewer.py")
                
                with button_col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{startup_name}", 
                               width="stretch",
                               help=f"Delete {startup_name} and remove from RAG"):
                        delete_pitch_deck(startup_name, deck)
                        st.rerun()
    st.divider()
else:
    st.info(" 📭  No pitch decks uploaded yet. Upload your first pitch deck below!")

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
            descriptions_list_for_json = []  # Initialize outside the if block
            if st.session_state.slides_info:
                st.session_state.slides_info = describe_slides_app(st.session_state.slides_info, openai_mm_llm)

                # 4. Save generated descriptions to data/docs/
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
            elif SHOW_LLM_EVALUATION:
                st.subheader(" ⚙️  Performing Startup Evaluation...")
                with st.spinner("Evaluating conditions based on slide descriptions..."):
                    # Calling the imported evaluate_startup function
                    st.session_state.evaluation_results = evaluate_startup(all_descriptions, uploaded_file.name, OPENAI_API_KEY)
                    st.session_state.current_conditions_prompt_display = None # Clear this as the new evaluation is dynamic
                    
                    # Save evaluation results to file for persistence
                    if st.session_state.evaluation_results and isinstance(st.session_state.evaluation_results, dict):
                        eval_filename = f"{base_file_name}_evaluation.json"
                        eval_path = os.path.join(DOCS_DIR, eval_filename)
                        with open(eval_path, 'w') as f:
                            json.dump(st.session_state.evaluation_results, f, indent=4)

            # --- Create deck-specific vector index ---
            if descriptions_list_for_json:
                with st.spinner("Creating vector index for this pitch deck..."):
                    try:
                        vector_index = create_deck_vector_index(base_file_name)
                        if vector_index:
                            st.success(f"✅ Created vector index for {base_file_name} - chat functionality available in deck viewer!")
                        else:
                            st.warning("Could not create vector index for this deck")
                    except Exception as e:
                        st.error(f"Error creating vector index: {e}")
    # Display a message about the newly uploaded deck with a button to view it
    if st.session_state.slides_info:
        startup_name = st.session_state.slides_info[0].get("base_name", "Startup")
        st.success(f"✅ Successfully processed {startup_name}!")
        if st.button(f"👁️ View {startup_name} Slides", type="primary", key="view_new_deck"):
            st.session_state.selected_deck = startup_name
            st.switch_page("pages/2_📊_Pitch_Deck_Viewer.py")

    # Display evaluation section
    if SHOW_LLM_EVALUATION and st.session_state.evaluation_results:
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

