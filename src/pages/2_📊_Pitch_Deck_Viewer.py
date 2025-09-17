import streamlit as st
import os
import json
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer

# Initialize sidebar visibility state
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False

# Set page config with sidebar always expanded
st.set_page_config(
    page_title="Pitch Deck Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Toggle sidebar visibility function
def toggle_sidebar():
    st.session_state.show_sidebar = not st.session_state.show_sidebar


# CSS to control sidebar visibility and hide navigation
sidebar_display = "block" if st.session_state.show_sidebar else "none"
st.markdown(f"""
<style>
    /* Conditionally hide/show the sidebar */
    section[data-testid="stSidebar"][aria-expanded="true"] {{
        display: {sidebar_display} !important;
    }}
    
    /* Hide only the multipage navigation widget */
    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}
    
    /* Hide the specific navigation list but preserve other elements */
    .css-1544g2n ul {{
        display: none !important;
    }}
    
    /* Keep the collapse button visible when sidebar is hidden */
    [data-testid="collapsedControl"] {{
        display: block !important;
    }}
    
    /* Keep sidebar header controls visible */
    section[data-testid="stSidebar"] > div > div:first-child {{
        display: block !important;
    }}
</style>
""", unsafe_allow_html=True)

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

# Use full width for main content
# Slide navigation controls with chat toggle
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])

with nav_col1:
    # Chat toggle button all the way to the left
    if deck_chat_engine:
        if st.button("✨ Assistant", key="nav_toggle_btn", width="stretch"):
            toggle_sidebar()
            st.rerun()
    
with nav_col2:
    if st.button("← Previous", width="stretch", 
                disabled=(st.session_state.current_slide == 0)):
        st.session_state.current_slide -= 1
        st.rerun()

with nav_col3:
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

with nav_col4:
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
    

# Initialize chat panel state
if "show_assistant" not in st.session_state:
    st.session_state.show_assistant = False

# Error message if chat engine not available
if not deck_chat_engine:
    st.error("💡 Chat functionality not available for this deck. The vector index may need to be rebuilt.")

# Assistant sidebar - always render content (visibility controlled by CSS)
if deck_chat_engine:
    with st.sidebar:
        # Modern assistant styling
        st.markdown("""
        <style>
        /* Sidebar styling */
        .stSidebar {
            background-color: #1e1e1e !important;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background: transparent !important;
            border: none !important;
            padding: 8px 0 !important;
        }
        
        /* User message styling */
        .stChatMessage[data-testid="user"] {
            flex-direction: row-reverse !important;
        }
        
        .stChatMessage[data-testid="user"] .stChatMessageContent {
            background: #2563eb !important;
            color: white !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 12px 16px !important;
            margin-left: 40px !important;
            margin-right: 0 !important;
        }
        
        /* Assistant message styling */
        .stChatMessage[data-testid="assistant"] .stChatMessageContent {
            background: #374151 !important;
            color: #f3f4f6 !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 12px 16px !important;
            margin-right: 40px !important;
            margin-left: 0 !important;
        }
        
        /* Hide chat avatars */
        .stChatMessage img {
            display: none !important;
        }
        
        /* Position assistant header at absolute top right of sidebar */
        .assistant-header {
            position: absolute !important;
            top: 16px !important;
            right: 16px !important;
            background: transparent !important;
            z-index: 1002 !important;
            padding: 0 !important;
        }
        
        /* Make sure sidebar has relative positioning for absolute children */
        section[data-testid="stSidebar"] {
            position: relative !important;
        }
        
        /* Make sidebar content take full height */
        section[data-testid="stSidebar"] .main {
            height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
            padding-top: 20px !important;
        }
        
        /* Hide the specific close button in sidebar */
        .stSidebar .stButton > button[kind="secondary"] {
            display: none !important;
        }
        
        /* Chat input styling - natural positioning in sidebar */
        .stSidebar .stChatInput > div {
            background: #374151 !important;
            border: 1px solid #4b5563 !important;
            border-radius: 12px !important;
            height: 60px !important;
        }
        
        .stSidebar .stChatInput input {
            background: transparent !important;
            color: white !important;
            border: none !important;
            padding: 16px 20px !important;
            font-size: 14px !important;
        }
        
        .stSidebar .stChatInput input::placeholder {
            color: #9ca3af !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: #374151 !important;
            color: white !important;
            border: 1px solid #4b5563 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        
        .stButton > button:hover {
            background: #4b5563 !important;
            border-color: #6b7280 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        
        # Create a container that takes up all available space
        chat_container = st.container(height=600)
        with chat_container:
            # Display chat history with improved styling
            for i, msg in enumerate(st.session_state[chat_history_key]):
                if msg["role"] == "user":
                    st.chat_message("user", avatar="💭").write(msg["content"])
                else:
                    st.chat_message("assistant", avatar="🤖").write(msg["content"])
        
        # Chat input at the bottom - let Streamlit handle the positioning naturally
        has_messages = len(st.session_state[chat_history_key]) > 0
        placeholder = "Ask anything about this deck..." if has_messages else f"Ask about {deck_name}..."
        
        if sidebar_prompt := st.chat_input(placeholder, key="sidebar_chat_input"):
            # Add user message to history
            st.session_state[chat_history_key].append({"role": "user", "content": sidebar_prompt})
            
            # Generate AI response with modern spinner
            with st.spinner("🤖 Thinking..."):
                try:
                    response = deck_chat_engine.chat(sidebar_prompt)
                    assistant_text = response.response
                    
                    # Add AI response to history
                    st.session_state[chat_history_key].append({"role": "assistant", "content": assistant_text})
                    
                except Exception as e_chat:
                    error_msg = f"Sorry, I encountered an error: {e_chat}"
                    st.session_state[chat_history_key].append({"role": "assistant", "content": error_msg})
            
            st.rerun()