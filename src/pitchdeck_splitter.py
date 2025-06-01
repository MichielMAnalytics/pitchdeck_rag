# src/pitchdeck_splitter.py
import fitz  # PyMuPDF
import os
from PIL import Image # Although PIL.Image isn't directly used for saving pix.save, it's a common dependency.

def pdf_to_images(pdf_path: str, output_folder: str, base_name: str) -> list[str]:
    """
    Converts a PDF to a series of images (PNG) and saves them to the output_folder.
    Returns a list of paths to the saved images.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    if not pdf_document.page_count:
        print(f"Warning: PDF '{pdf_path}' has no pages.")
        pdf_document.close()
        return []
 
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        # Format output image path consistently (e.g., startup_name_slide_01.png)
        if page_num < 9: # For slide numbers 1-9, use a leading zero
            output_image_path = os.path.join(output_folder, f"{base_name}_slide_0{page_num + 1}.png")
        else:
            output_image_path = os.path.join(output_folder, f"{base_name}_slide_{page_num + 1}.png")
        
        pix.save(output_image_path)
        image_paths.append(output_image_path)
    
    pdf_document.close()
    return image_paths

# Example usage (for testing this module independently, not used directly by Streamlit)
if __name__ == "__main__":
    # Define dummy paths for demonstration
    # In a real scenario, these would come from user input or config
    dummy_pitchdecks_path = './data/uploaded_pitchdecks/' # Updated path
    dummy_slides_path = './data/slides/dummy_startup_slides/' # Updated path for specific slides
    os.makedirs(dummy_pitchdecks_path, exist_ok=True)
    os.makedirs(dummy_slides_path, exist_ok=True) # Ensure slides sub-directory exists

    # Create a dummy PDF for testing (requires PyMuPDF)
    try:
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text(fitz.Point(50, 100), "Dummy Slide 1: Introduction", fontsize=20)
        page2 = doc.new_page()
        page2.insert_text(fitz.Point(50, 100), "Dummy Slide 2: Data", fontsize=20)
        dummy_pdf_path = os.path.join(dummy_pitchdecks_path, "dummy_startup.pdf")
        doc.save(dummy_pdf_path)
     
        doc.close()
        print(f"Created dummy PDF at: {dummy_pdf_path}")
        print("Processing dummy PDF...")
        converted_images = pdf_to_images(dummy_pdf_path, dummy_slides_path, "dummy_startup")
        print(f"Converted {len(converted_images)} images: {converted_images}")
        # Clean up dummy files (optional)
        # os.remove(dummy_pdf_path)
        # for img_path in converted_images:
        #     os.remove(img_path)
     
        # os.rmdir(dummy_slides_path) # Only if empty
    except Exception as e:
        print(f"Could not create dummy PDF or process it: {e}")
