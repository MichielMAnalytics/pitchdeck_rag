import os
import base64
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import Document 

def describe_image(image_path: str, openai_mm_llm: OpenAI) -> str:
    """
    Generates a description for a single image using OpenAI with vision capabilities.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Use the OpenAI client directly with proper message format
    client = openai.OpenAI(api_key=openai_mm_llm.api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This image is part of a slidedeck, describe the relevant content for analysis of the presentation. Please respond in full sentences and without empty lines."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content

def describe_slides_in_folder(image_directory: str, openai_api_key: str) -> list[Document]:
    """
    Processes all PNG images in a directory, generates descriptions,
    and returns them as a list of LlamaIndex Document objects.
    """
    os.environ["OPENAI_API_KEY"] = openai_api_key # Ensure API key is set for this context
    openai_mm_llm = OpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key
    )
    if not os.path.isdir(image_directory):
        print(f"Error: Directory '{image_directory}' does not exist.")
        return []
    image_files = sorted([
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if f.endswith(".png")
    ])
    documents = []
    print(f"Found {len(image_files)} images in {image_directory} for description.")
    for image_file in image_files:
        try:
            print(f"Processing image: {image_file}")
            # The describe_image function already uses ImageDocument internally now
            description = describe_image(image_file, openai_mm_llm)
            print(f"Response: {description[:100]}...") # Print first 100 chars
            print("---------------------------------")
            # Extract startup name from filename (assuming format like startup_name_slide_XX.png)
            # Adjust this if your naming convention is different
            try:
                base_filename = os.path.basename(image_file)
                startup_name = base_filename.split('_slide_')[0].replace("_", " ").title()
                page_number = int(base_filename.split('_slide_')[-1].replace('.png', ''))
            
            except Exception:
                startup_name = "Unknown Startup"
                page_number = -1 # Indicate unknown page
            document = Document(
                text=description,
                doc_id=os.path.basename(image_file),
            
                metadata={"startup_name": startup_name, "page_number": page_number, "source_image": image_file}
            )
            documents.append(document)
            print(f"Finished processing {image_file}.")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            # Optionally add an error document
            documents.append(
                Document(
                    text=f"Error processing image {os.path.basename(image_file)}: {e}",
                    doc_id=os.path.basename(image_file),
                    metadata={"error": True, "source_image": image_file}
               
                )
            )
    return documents

# Example usage (for testing this module independently)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual OpenAI API key for testing
    # Or ensure it's set as an environment variable
    TEST_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not TEST_API_KEY:
        print("OPENAI_API_KEY environment variable not set. Cannot run description generation example.")
        print("Please set it or replace TEST_API_KEY with your key for testing.")
    else:
        # Assuming you have some dummy images in a folder
        # Make sure 'data/slides/' exists and has some .png files
        dummy_image_dir = './data/slides/dummy_startup_slides/' # Updated path to a specific subfolder for consistency
        if not os.path.isdir(dummy_image_dir) or not any(f.endswith('.png') for f in os.listdir(dummy_image_dir)):
            print(f"No PNG images found in '{dummy_image_dir}'. Please ensure images are present for testing.")
            print("Run pitchdeck_splitter.py first to generate some dummy images.")
        else:
            print(f"Describing images in: {dummy_image_dir}")
            described_docs = describe_slides_in_folder(dummy_image_dir, TEST_API_KEY)
            print(f"\nGenerated {len(described_docs)} documents:")
            for doc in described_docs:
                print(f"Doc ID: {doc.doc_id}, Startup: {doc.metadata.get('startup_name')}, Page: {doc.metadata.get('page_number')}")
                print(f"Text (first 50 chars): {doc.text[:50]}...\n")
