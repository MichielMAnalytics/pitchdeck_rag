# src/vector_index_loader.py
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.core.memory import ChatMemoryBuffer

def load_existing_index(persistent_dir: str, openai_api_key: str, memory: ChatMemoryBuffer, system_prompt_chat: str, model_name: str):
    """
    Loads a VectorStoreIndex from the specified persistent directory.
    Returns the loaded index and a chat engine built from it, or None if loading fails.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key # Ensure API key is set
    if not os.path.exists(persistent_dir) or not os.path.isdir(persistent_dir) or not os.listdir(persistent_dir):
        print(f"Warning: No existing index found at '{persistent_dir}'. Returning None.")
        return None, None
    try:
        print(f"Loading VectorStoreIndex from {persistent_dir}...")
        vec_storage_context = StorageContext.from_defaults(persist_dir=persistent_dir)
        vector_index = load_index_from_storage(vec_storage_context)
        print("VectorStoreIndex loaded successfully.")
        chat_engine = vector_index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            llm=LI_OpenAI(temperature=0, model=model_name),
            system_prompt=system_prompt_chat,
            verbose=True,
            similarity_top_k=20,
        )
        return vector_index, chat_engine
    except Exception as e:
        print(f"Error loading VectorStoreIndex from '{persistent_dir}': {e}")
        return None, None

# Example usage (for testing this module independently)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    # Or ensure it's set as an environment variable
    TEST_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not TEST_API_KEY:
        print("OPENAI_API_KEY environment variable not set. Cannot run index loader example.")
        print("Please set it or replace TEST_API_KEY with your key for testing.")
    else:
        # This path should match where your vector_index_builder.py saved the index
        index_persist_dir = "./data/VectorStoreIndex/Rag" # Updated path
        # Dummy memory and prompt for testing loader
        dummy_memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        dummy_system_prompt = "You are a test assistant."
        
        loaded_idx, loaded_engine = load_existing_index(
            index_persist_dir, TEST_API_KEY, dummy_memory, dummy_system_prompt, "gpt-4o"
        )
        
        if loaded_idx and loaded_engine:
            print("\nSuccessfully loaded index and created chat engine.")
            # You can now test the engine, e.g.,
            # response = loaded_engine.chat("What information is in the index?")
            # print(response.response)
        else:
            print("\nFailed to load index. Ensure 'vector_index_builder.py' was run successfully.")
