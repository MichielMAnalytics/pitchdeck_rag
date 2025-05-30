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
    dummy_pitchdecks_path = './data/pitchdecks/'
    dummy_slides_path = './data/slides_output/'
    os.makedirs(dummy_pitchdecks_path, exist_ok=True)
    os.makedirs(dummy_slides_path, exist_ok=True)

    # Process all PDFs in the pitchdecks folder
    pdf_files = [f for f in os.listdir(dummy_pitchdecks_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{dummy_pitchdecks_path}'. Please add some PDFs for splitting.")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dummy_pitchdecks_path, pdf_file)
            base_name = os.path.splitext(pdf_file)[0]
            print(f"Processing PDF: {pdf_path}")
            try:
                converted_images = pdf_to_images(pdf_path, dummy_slides_path, base_name)
                print(f"Converted {len(converted_images)} images for '{pdf_file}': {converted_images}")
            except Exception as e:
                print(f"Error processing '{pdf_file}': {e}")