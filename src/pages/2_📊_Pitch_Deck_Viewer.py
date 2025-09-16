import streamlit as st
import os
import json
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer

# Set page config
st.set_page_config(
    page_title="Pitch Deck Viewer",
    page_icon="📊",
    layout="wide"
)

# Define data directories (same as in main app)
DATA_DIR = "./data"
PITCHDECKS_DIR = "./data/uploaded_pitchdecks"
SLIDES_DIR = "./data/slides"
DOCS_DIR = "./data/docs"
PERSIST_INDEX_DIR = "./data/vector_store_index"

# --- Constants for chat ---
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT_CHAT = (
    "You are a helpful assistant for the investment team of a venture capital fund called UVC Partners. "
    "You have access to information about this specific pitch deck. "
    "Answer questions about this pitch deck with the most amount of information you can find in the provided context. "
    "Be specific and reference slide numbers when relevant."
)

# Get deck name from session state (fallback to query params for direct links)
deck_name = None
if "selected_deck" in st.session_state and st.session_state.selected_deck:
    deck_name = st.session_state.selected_deck
    # Update URL with the deck name for better UX
    st.query_params["deck"] = deck_name
else:
    # Try query parameters as fallback
    query_params = st.query_params
    deck_name = query_params.get("deck", None)

if not deck_name:
    st.error("❌ No pitch deck specified")
    st.info("Please select a pitch deck from the dashboard to view.")
    if st.button("← Back to Dashboard", type="primary"):
        st.switch_page("app.py")
    st.stop()

# Check if the deck exists
pdf_path = os.path.join(PITCHDECKS_DIR, f"{deck_name}.pdf")
slides_path = os.path.join(SLIDES_DIR, deck_name)
descriptions_path = os.path.join(DOCS_DIR, f"{deck_name}_descriptions.json")

if not os.path.exists(pdf_path):
    st.error(f"❌ Pitch deck '{deck_name}' not found")
    if st.button("← Back to Dashboard", type="primary"):
        st.switch_page("app.py")
    st.stop()

# Page Header
col1, col2 = st.columns([6, 1])
with col1:
    st.title(f"📊 {deck_name}")
    st.caption(f"Dashboard > Pitch Deck Viewer > {deck_name}")
with col2:
    if st.button("← Back", type="secondary", width="stretch"):
        st.switch_page("app.py")

st.divider()

# Load slide descriptions if available
descriptions = {}
if os.path.exists(descriptions_path):
    with open(descriptions_path, 'r') as f:
        desc_data = json.load(f)
        descriptions = {item['metadata']['page_number']: item['text'] for item in desc_data}

# Get all slide images
slide_images = []
if os.path.exists(slides_path):
    # Get all PNG files and sort them by slide number
    slide_files = sorted([f for f in os.listdir(slides_path) if f.endswith('.png')])
    slide_images = [os.path.join(slides_path, f) for f in slide_files]

if not slide_images:
    st.warning("⚠️ No slides found for this pitch deck. It may need to be processed.")
    st.info("Please upload the deck again from the dashboard to generate slides.")
    st.stop()

# Initialize session state for current slide
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0

# Ensure current slide is within bounds
st.session_state.current_slide = min(st.session_state.current_slide, len(slide_images) - 1)
st.session_state.current_slide = max(st.session_state.current_slide, 0)

# Load deck-specific chat engine
@st.cache_resource
def get_chat_memory_buffer():
    """Caches and returns a ChatMemoryBuffer for the chat engine."""
    return ChatMemoryBuffer.from_defaults(token_limit=10000)

def load_deck_chat_engine(deck_name: str):
    """Load chat engine for specific deck."""
    deck_index_dir = os.path.join(PERSIST_INDEX_DIR, deck_name)
    
    if not os.path.exists(deck_index_dir) or not os.listdir(deck_index_dir):
        return None
    
    try:
        vec_ctx = StorageContext.from_defaults(persist_dir=deck_index_dir)
        vector_index = load_index_from_storage(vec_ctx)
        
        memory = get_chat_memory_buffer()
        chat_engine = vector_index.as_chat_engine(
            chat_mode="context", 
            memory=memory, 
            llm=LI_OpenAI(temperature=0, model=MODEL_NAME),
            system_prompt=SYSTEM_PROMPT_CHAT, 
            verbose=True, 
            similarity_top_k=10
        )
        return chat_engine
    except Exception as e:
        st.error(f"Error loading chat engine for {deck_name}: {e}")
        return None

# Initialize chat session state for this deck
chat_history_key = f"chat_history_{deck_name}"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

# Load chat engine
deck_chat_engine = load_deck_chat_engine(deck_name)

# Create main layout
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    # Slide navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    
    with nav_col1:
        if st.button("← Previous", width="stretch", 
                    disabled=(st.session_state.current_slide == 0)):
            st.session_state.current_slide -= 1
            st.rerun()
    
    with nav_col2:
        # Slide selector
        selected_slide = st.selectbox(
            "Select slide:",
            options=list(range(len(slide_images))),
            index=st.session_state.current_slide,
            format_func=lambda x: f"Slide {x + 1} of {len(slide_images)}",
            label_visibility="collapsed"
        )
        if selected_slide != st.session_state.current_slide:
            st.session_state.current_slide = selected_slide
            st.rerun()
    
    with nav_col3:
        if st.button("Next →", width="stretch",
                    disabled=(st.session_state.current_slide == len(slide_images) - 1)):
            st.session_state.current_slide += 1
            st.rerun()
    
    # Display current slide
    st.image(slide_images[st.session_state.current_slide], 
             width="stretch",
             caption=f"Slide {st.session_state.current_slide + 1} of {len(slide_images)}")
    
    # Display slide description if available
    current_page_num = st.session_state.current_slide + 1
    if current_page_num in descriptions:
        with st.expander("📝 Slide Description", expanded=True):
            st.markdown(descriptions[current_page_num])
    
    # Add chat interface
    st.divider()
    st.subheader("💬 Chat about this Pitch Deck")
    
    if deck_chat_engine:
        # Display chat history
        for msg in st.session_state[chat_history_key]:
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Chat input
        user_input = st.chat_input(f"Ask questions about {deck_name}...")
        
        if user_input:
            # Add user message to chat history
            st.session_state[chat_history_key].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            # Generate AI response
            with st.spinner("Thinking..."):
                try:
                    response = deck_chat_engine.chat(user_input)
                    assistant_text = response.response
                    
                    # Add AI response to chat history
                    st.session_state[chat_history_key].append({"role": "assistant", "content": assistant_text})
                    st.chat_message("assistant").write(assistant_text)
                    
                except Exception as e_chat:
                    error_msg = f"Sorry, I encountered an error: {e_chat}"
                    st.session_state[chat_history_key].append({"role": "assistant", "content": error_msg})
                    st.chat_message("assistant").write(error_msg)
                    st.error(f"Chat error: {e_chat}")
    else:
        st.info(f"💡 Chat functionality not available for {deck_name}. The vector index may need to be rebuilt.")
        st.caption("Upload the pitch deck again to enable chat functionality.")

with sidebar_col:
    st.caption("Click to jump to a slide")
    
    # Create a scrollable container for thumbnails
    for idx, slide_path in enumerate(slide_images):
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"Slide {idx + 1}", 
                        key=f"thumb_{idx}",
                        width="stretch",
                        type="primary" if idx == st.session_state.current_slide else "secondary"):
                st.session_state.current_slide = idx
                st.rerun()
        with col2:
            if idx == st.session_state.current_slide:
                st.markdown("**→**")
    
    # Display deck information
    st.divider()
    st.subheader("ℹ️ Deck Information")
    st.markdown(f"**Total Slides:** {len(slide_images)}")
    st.markdown(f"**Descriptions Available:** {'Yes ✅' if descriptions else 'No ❌'}")
    
    # Load and display evaluation results if they exist for this deck
    # Check if there's a stored evaluation for this specific deck
    eval_file = os.path.join(DOCS_DIR, f"{deck_name}_evaluation.json")
    if os.path.exists(eval_file):
        st.divider()
        st.subheader("📊 Evaluation Summary")
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)
            if isinstance(results, dict):
                st.markdown(f"**Startup:** {results.get('startup_name', deck_name)}")
                st.markdown(f"**Category:** {results.get('category', 'N/A')}")
                st.markdown(f"**Region:** {results.get('region', 'N/A')}")
                st.markdown(f"**Funding Round:** {results.get('funding_round', 'N/A')}")
        except Exception as e:
            st.error(f"Could not load evaluation: {e}")