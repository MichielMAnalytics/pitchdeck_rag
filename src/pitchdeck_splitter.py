from pdf2image import convert_from_path
import os
from PIL import Image

def pdf_to_images(pdf_path: str, output_folder: str, base_name: str) -> list[str]:
    """
    Converts a PDF to a series of images (PNG) and saves them to the output_folder.
    Returns a list of paths to the saved images.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Convert PDF to PIL Images
        pages = convert_from_path(pdf_path)
        image_paths = []
        
        if not pages:
            print(f"Warning: PDF '{pdf_path}' has no pages.")
            return []

        for page_num, page in enumerate(pages):
            # Format output image path consistently (e.g., startup_name_slide_01.png)
            if page_num < 9: # For slide numbers 1-9, use a leading zero
                output_image_path = os.path.join(output_folder, f"{base_name}_slide_0{page_num + 1}.png")
            else:
                output_image_path = os.path.join(output_folder, f"{base_name}_slide_{page_num + 1}.png")
            
            page.save(output_image_path, 'PNG')
            image_paths.append(output_image_path)
        
        return image_paths
        
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

# Example usage (for testing this module independently, not used directly by Streamlit)
if __name__ == "__main__":
    # Define dummy paths for demonstration
    # In a real scenario, these would come from user input or config
    dummy_pitchdecks_path = './data/uploaded_pitchdecks/' # Updated path
    dummy_slides_path = './data/slides/dummy_startup_slides/' # Updated path for specific slides
    os.makedirs(dummy_pitchdecks_path, exist_ok=True)
    os.makedirs(dummy_slides_path, exist_ok=True) # Ensure slides sub-directory exists

    # For testing, you would need to place a sample PDF in the dummy_pitchdecks_path
    # since we no longer use PyMuPDF to create test PDFs
    try:
        # Look for any PDF files in the dummy directory
        pdf_files = [f for f in os.listdir(dummy_pitchdecks_path) if f.endswith('.pdf')]
        if pdf_files:
            dummy_pdf_path = os.path.join(dummy_pitchdecks_path, pdf_files[0])
            print(f"Processing PDF: {dummy_pdf_path}")
            converted_images = pdf_to_images(dummy_pdf_path, dummy_slides_path, "dummy_startup")
            print(f"Converted {len(converted_images)} images: {converted_images}")
        else:
            print("No PDF files found in dummy directory. Place a sample PDF for testing.")
    except Exception as e:
        print(f"Could not process PDF: {e}")