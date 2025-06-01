# src/vector_index_builder.py
import json
import os
from llama_index.core import Document, VectorStoreIndex, StorageContext

# Custom deserialization for Document objects (matching the format from slide_description_gen)
def document_decoder(dct):
    if "text" in dct and "doc_id" in dct:
        return Document(
            text=dct["text"],
            doc_id=dct["doc_id"],
            metadata=dct.get("metadata", {}),
        )
    return dct

def build_and_persist_index(input_json_path: str, persistent_dir: str, openai_api_key: str):
    """
    Loads documents from a JSON file, builds a VectorStoreIndex, and persists it.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key # Ensure API key is set
    loaded_documents = []
    try:
        with open(input_json_path, "r") as file:
            loaded_documents = json.load(file, object_hook=document_decoder)
        print(f"Loaded {len(loaded_documents)} documents from {input_json_path}")
    except FileNotFoundError:
        print(f"Error: The file '{input_json_path}' does not exist.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading documents: {e}")
        return
    if not loaded_documents:
        print("No documents loaded to build the index. Exiting.")
        return
    print(f"Building VectorStoreIndex and persisting to {persistent_dir}...")
    try:
        # Create the persistence directory if it doesn't exist
        os.makedirs(persistent_dir, exist_ok=True)
        # Build the index from the loaded documents
        vector_index = VectorStoreIndex.from_documents(
            loaded_documents,
            show_progress=True # Shows progress during embedding
        )
        # Persist the index
        vector_index.storage_context.persist(persist_dir=persistent_dir)
        print(f"VectorStoreIndex successfully persisted to {persistent_dir}")
    except Exception as e:
        print(f"Error building or persisting VectorStoreIndex: {e}")

if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    # Or ensure it's set as an environment variable
    TEST_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not TEST_API_KEY:
        print("OPENAI_API_KEY environment variable not set. Cannot run index builder example.")
        print("Please set it or replace TEST_API_KEY with your key for testing.")
    else:
        # Example paths for building the index
        # This JSON file would contain descriptions from multiple pitch decks
        input_json_file = "./data/v2_slides_descriptions.json" # Adjust this path as needed
        
        # Make sure your slide_description_gen.py has created some data in this JSON first
        # Or manually create a dummy JSON file for testing:
        # with open(input_json_file, 'w') as f:
        #     json.dump([
        #         {"text": "This is a test document about startup A.", "doc_id": "test_doc_A", "metadata": {"startup_name": "Test Startup A"}},
        #         {"text": "This is another test document about startup B.", "doc_id": "test_doc_B", "metadata": {"startup_name": "Test Startup B"}}
        #     ], f, indent=4)
        index_persist_dir = "./data/vector_store_index" # Updated path
        if not os.path.exists(input_json_file):
            print(f"Warning: Input JSON file '{input_json_file}' not found.")
            print("Please run slide_description_gen.py first or create a dummy JSON file.")
        else:
            build_and_persist_index(input_json_file, index_persist_dir, TEST_API_KEY)
