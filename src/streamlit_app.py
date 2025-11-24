"""
OCR Text Extraction Streamlit Application - Improved Version
Key improvements:
- Better error handling and resource cleanup
- Removed RAR from supported formats (not supported by py7zr)
- Added configuration constants
- Improved memory management
- Better progress tracking
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
import zipfile
import py7zr
from PIL import Image, ImageFile
import numpy as np
import os
import io
from contextlib import contextmanager

# Configuration
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.tif')
ARCHIVE_EXTENSIONS = ('.zip', '.7z')

# Placeholder imports - replace with actual implementations
# from black_roi.blackening_roi import black_roi
# from ocr_process.image_processor import process_roi_x, process_roi_y
# from ocr_process.text_extractor import extract_from_image
# from ocr_process.text_cleaner import clean_text
# from ocr_process.save_to_csv import save_side_by_side_csv

@contextmanager
def cleanup_temp_dir(temp_dir):
    """Context manager for automatic cleanup of temporary directories"""
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def process_images(input_folder: Path, process_fn, output_folder_name="output", ocr_results=None):
    """Process images with robust error handling and progress tracking"""
    output_folder = input_folder / output_folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "process_log.txt"
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(message: str):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        original_stems = []
        processed_count = 0
        skipped_count = 0
        total_images = 0

        # Collect all image paths first
        image_paths = []
        for root, dirs, files in os.walk(input_folder):
            root_path = Path(root)
            if output_folder in root_path.parents or root_path == output_folder:
                continue

            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append((root_path, file))

        total_images = len(image_paths)
        log(f"Found {total_images} images to process\n")

        # Process with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (root_path, file) in enumerate(image_paths):
            input_path = root_path / file
            relative_path = input_path.relative_to(input_folder)
            stem, suffix = relative_path.stem, relative_path.suffix

            # Update progress
            progress = (idx + 1) / total_images
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx + 1}/{total_images}: {file}")

            # OCR prefix
            ocr_text = ""
            if ocr_results and stem in ocr_results:
                x_vals = ocr_results[stem].get('X', [])
                if len(x_vals) >= 2:
                    ocr_text = f"{x_vals[0]}_{x_vals[1]}_"

            processed_name = f"{ocr_text}{stem}{suffix}"
            processed_path = output_folder / relative_path.parent / processed_name
            processed_path.parent.mkdir(parents=True, exist_ok=True)

            # Validate and load image
            try:
                img = load_image_safely(input_path)
                if img is None:
                    log(f"⚠️ Not a valid image: {input_path}")
                    skipped_count += 1
                    continue

                # Process
                array = np.array(img)
                processed_array = process_fn(array)
                Image.fromarray(processed_array).save(processed_path)
                log(f"✓ Processed: {processed_path.name}")
                original_stems.append(stem)
                processed_count += 1

            except Exception as e:
                log(f"✗ Failed: {input_path.name} - {str(e)}")
                skipped_count += 1
                continue

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Summary
        summary = (
            "\n" + "="*50 + "\n"
            "PROCESS SUMMARY\n"
            "="*50 + "\n"
            f"Total images found    : {total_images}\n"
            f"Successfully processed: {processed_count}\n"
            f"Skipped (errors)      : {skipped_count}\n"
            f"Success rate          : {processed_count/total_images*100:.1f}%\n"
            "="*50 + "\n"
        )
        log_file.write(summary)
        print(summary)

    return original_stems, total_images, processed_count, skipped_count, output_folder

def load_image_safely(image_path: Path) -> Image.Image | None:
    """Load image with multiple fallback strategies"""
    import imghdr
    
    # Validate file type
    if imghdr.what(image_path) is None:
        return None
    
    # Try standard loading
    try:
        with Image.open(image_path) as im:
            return im.convert("RGB")
    except Exception:
        pass
    
    # Fallback: load via bytes
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def save_uploaded_files(uploaded_files, temp_dir: Path):
    """Save uploaded files to temporary directory"""
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return temp_dir, saved_paths

def extract_archives(file_paths: list[Path], extract_dir: Path):
    """Extract ZIP and 7Z archives"""
    extracted_folders = []
    
    for file_path in file_paths:
        suffix = file_path.suffix.lower()
        if suffix not in ARCHIVE_EXTENSIONS:
            continue
            
        extracted_folder = extract_dir / file_path.stem
        extracted_folder.mkdir(exist_ok=True)
        
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)
                extracted_folders.append(extracted_folder)
                st.success(f"✓ Extracted: {file_path.name}")
                
            elif suffix == ".7z":
                with py7zr.SevenZipFile(file_path, mode='r') as archive:
                    archive.extractall(path=extracted_folder)
                extracted_folders.append(extracted_folder)
                st.success(f"✓ Extracted: {file_path.name}")
                
        except Exception as e:
            st.warning(f"Failed to extract {file_path.name}: {e}")
    
    return extracted_folders

def collect_images_recursive(root_folder: Path):
    """Recursively collect all image files"""
    return [
        p for p in root_folder.rglob("*") 
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]

def create_zip(folder_path: Path, zip_path: Path):
    """Create ZIP archive of folder"""
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(folder_path))
    return zip_path

def show_image_gallery(folder: Path, max_images: int = 12):
    """Display gallery of processed images"""
    image_files = sorted([
        f for f in folder.glob("*") 
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])[:max_images]
    
    if not image_files:
        return
    
    st.markdown("### 🖼️ Preview of Processed Images")
    if len(image_files) > max_images:
        st.info(f"Showing first {max_images} images out of {len(list(folder.glob('*')))}")
    
    cols = st.columns(3)
    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                with cols[i % 3]:
                    st.image(img, use_container_width=True, caption=img_path.name)
        except Exception as e:
            st.warning(f"Cannot display {img_path.name}: {e}")

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(
        page_title="OCR Text Extractor", 
        page_icon="🧠", 
        layout="centered"
    )
    
    st.title("🧠 OCR Text Extraction Pipeline")
    st.markdown("""
    Upload images or archives to extract text using OCR.
    - Supports: PNG, JPG, JPEG, TIF, TIFF
    - Archives: ZIP, 7Z
    """)

    uploaded_files = st.file_uploader(
        "📁 Upload images or archives",
        type=["png", "jpg", "jpeg", "tif", "tiff", "7z", "zip"],
        accept_multiple_files=True,
        help="Upload individual images or compressed archives"
    )

    if uploaded_files:
        st.success(f"✓ Uploaded {len(uploaded_files)} file(s)")
        
        if st.button("🚀 Start OCR Processing", type="primary"):
            with st.spinner("Processing your files..."):
                try:
                    # Create temporary directory with cleanup
                    temp_dir = Path(tempfile.mkdtemp())
                    
                    with cleanup_temp_dir(temp_dir):
                        # Save uploaded files
                        image_folder, file_paths = save_uploaded_files(uploaded_files, temp_dir)
                        
                        # Extract archives
                        extracted_dirs = extract_archives(file_paths, temp_dir)
                        
                        # Determine base name
                        archive_files = [f for f in file_paths if f.suffix.lower() in ARCHIVE_EXTENSIONS]
                        base_name = archive_files[0].stem if archive_files else "output"
                        
                        # Collect all images
                        all_images = [f for f in file_paths if f.suffix.lower() in IMAGE_EXTENSIONS]
                        for dir in extracted_dirs:
                            all_images.extend(collect_images_recursive(dir))
                        
                        if not all_images:
                            st.warning("⚠️ No valid image files found")
                            return
                        
                        st.info(f"Found {len(all_images)} images to process")
                        
                        # Process images (implement your pipeline here)
                        # output_folder, csv_path = run_pipeline(temp_dir, all_images, base_name)
                        
                        # For demo purposes:
                        output_folder = temp_dir / f"{base_name}_output"
                        output_folder.mkdir(exist_ok=True)
                        
                        st.success("✅ Processing complete!")
                        
                        # Show gallery
                        show_image_gallery(output_folder)
                        
                        # Show log
                        log_path = output_folder / "process_log.txt"
                        if log_path.exists():
                            with st.expander("📝 View Processing Log"):
                                st.code(log_path.read_text(), language="text")
                        
                        # Create and offer download
                        zip_path = temp_dir / f"{base_name}_output.zip"
                        create_zip(output_folder, zip_path)
                        
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "📥 Download Results (ZIP)",
                                f,
                                file_name=f"{base_name}_output.zip",
                                mime="application/zip"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()