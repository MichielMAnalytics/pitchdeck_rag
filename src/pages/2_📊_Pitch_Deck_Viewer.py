import streamlit as st
import os
import json
from pathlib import Path

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
    if st.button("← Back", type="secondary", use_container_width=True):
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

# Create main layout
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    # Slide navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    
    with nav_col1:
        if st.button("← Previous", use_container_width=True, 
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
        if st.button("Next →", use_container_width=True,
                    disabled=(st.session_state.current_slide == len(slide_images) - 1)):
            st.session_state.current_slide += 1
            st.rerun()
    
    # Display current slide
    st.image(slide_images[st.session_state.current_slide], 
             use_container_width=True,
             caption=f"Slide {st.session_state.current_slide + 1} of {len(slide_images)}")
    
    # Display slide description if available
    current_page_num = st.session_state.current_slide + 1
    if current_page_num in descriptions:
        with st.expander("📝 Slide Description", expanded=True):
            st.markdown(descriptions[current_page_num])

with sidebar_col:
    st.caption("Click to jump to a slide")
    
    # Create a scrollable container for thumbnails
    for idx, slide_path in enumerate(slide_images):
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"Slide {idx + 1}", 
                        key=f"thumb_{idx}",
                        use_container_width=True,
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