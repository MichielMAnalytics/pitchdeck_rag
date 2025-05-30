# 💼 RAG Chat & Evaluation for VC Pitch Decks

A Streamlit application to analyze VC pitch decks, generate slide descriptions, and perform automated startup evaluation using a Retrieval Augmented Generation (RAG) based chat.

---
## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* An OpenAI API Key

### 📦 Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/cosmincbodea/vc-pitchdecks-analyzer.git](https://github.com/cosmincbodea/vc-pitchdecks-analyzer.git)
    cd vc-pitchdecks-analyzer
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your OpenAI API Key:**
    * Create a folder named `.streamlit` in the root of your project (`vc-pitchdecks-analyzer/.streamlit/`).
    * Inside this folder, create a file named `secrets.toml`. Add your API key like this:
        ```toml
        # .streamlit/secrets.toml
        OPENAI_API_KEY="sk-proj-YOUR_ACTUAL_OPENAI_API_KEY"
        ```

### 📂 Data Directory Setup
The application expects certain directory structures for data storage and persistence. These will be created automatically if they don't exist:
* `data/pitchdecks/`: (Optional) You can place sample PDF pitch decks here.
* `data/slides_output/`: (Created by `slide_description_gen.py` if run standalone)
* `data/VectorStoreIndex/RAG/`: (Created and managed by LlamaIndex) This directory will store the persistent vector index for your RAG knowledge base.

## 🏃 How to Run the Application
1.  **Ensure your virtual environment is active.**
2.  **Navigate to the root of your project (`vc-pitchdecks-analyzer`).**
3.  **Run the Streamlit application:**
    ```bash
    streamlit run src/app.py
    ```
    Your browser should automatically open to the Streamlit app.

## ✨ Features
### Left Pane: Pitch Deck Upload & Analysis
* **Upload PDF**: Select and upload a PDF pitch deck.
* **Slide Extraction**: The PDF is processed, and each slide is converted into an image.
* **AI-Powered Descriptions**: A multimodal LLM generates a detailed text description for each slide image.
* **Startup Evaluation**: The aggregated slide descriptions are used to evaluate the startup against four pre-defined investment criteria (funding round, region, category, excluded fields). The results are presented in a structured JSON format.

### Right Pane: Cumulative RAG Chat
* **General Knowledge Base**: The RAG chat is initialized with a persistent vector index. If you run `src/vector_index_builder.py` beforehand, it will load that initial data.
* **Cumulative Knowledge**: **Crucially, when you upload new pitch decks via the left pane, their generated slide descriptions are automatically added to this RAG knowledge base.** This means the chat can answer questions about all previously uploaded and indexed pitch decks, as well as general VC information.
* **Interactive Chat**: Ask questions about the startups you've analyzed or general venture capital concepts.

## 🛠️ Developer Notes
* **`src/pitchdeck_splitter.py`**: Contains the core logic for PDF to image conversion. This function is called directly within `app.py`.
* **`src/slide_description_gen.py`**: Contains the core logic for generating descriptions from images. Its `describe_image` function is utilized by `app.py`.
* **`src/vector_index_builder.py`**: A standalone script (can be run once) to build an initial RAG index from a pre-existing collection of documents (e.g., a `v2_slides_descriptions.json` created by `slide_description_gen.py` run in batch mode). This helps pre-populate your general knowledge base.
* **`src/vector_index_loader.py`**: A helper module to load the index. The primary loading/initialization for the cumulative index is now handled directly within `app.py` for dynamic updates.
* **Persistence**: The RAG vector index is persisted in `data/VectorStoreIndex/RAG/`. This means the knowledge base grows and is saved across application restarts.
* **Temporary Files**: Uploaded PDFs and generated images for the *current* session's processing are stored in a temporary directory and cleaned up (Streamlit usually handles this on rerun, but for large files or long-running processes, manual cleanup might be needed).

## ⚠️ Important Considerations
* **API Key Security**: For production deployments, always use environment variables or platform-specific secret management (like Streamlit Cloud Secrets) and **never hardcode API keys** directly in your code.
* **Scalability**: While the cumulative indexing (using `insert()`) is efficient for adding documents, extremely large datasets might eventually benefit from dedicated vector databases (e.g., Pinecone, Weaviate, Chroma) which LlamaIndex can integrate with. For hundreds or thousands of pitch decks, the local file-based persistence might still work well.
* **Error Handling**: The application includes basic error handling, but more robust logging and user feedback mechanisms could be added for production readiness.
* **Resource Usage**: Processing PDFs and generating descriptions can be memory and CPU intensive, especially for large pitch decks or many concurrent users.# vc-pitchdecks-analyzer
An application to analyze VC pitch decks, generate slide descriptions, and perform automated startup evaluation using a Retrieval Augmented Generation (RAG) based chat.
