from PIL import Image
from pathlib import Path
from tqdm import tqdm
from pillow_heif import register_heif_opener


def convert_heic_to_jpeg(image_path: Path, output_path: Path, quality: int = 95) -> bool:
    """
    Convert a single HEIC image to JPEG format.
    
    Parameters:
    -----------
    image_path : Path
        Path to the HEIC image
    output_path : Path
        Path where the JPEG will be saved
    quality : int
        JPEG quality (1-100). Default is 95.
    
    Returns:
    --------
    bool
        True if conversion was successful, False otherwise
    """
    try:
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if necessary (JPEG doesn't support transparency)
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        
        # Save as JPEG
        image.save(output_path, 'JPEG', quality=quality)
        return True
    except Exception as e:
        print(f"Error converting {image_path.name}: {e}")
        return False


def convert_heic_folder_to_jpeg(input_folder: Path, output_folder: Path, quality: int = 95) -> dict:
    """
    Convert all HEIC images in a folder to JPEG format.
    
    Parameters:
    -----------
    input_folder : Path
        Folder containing HEIC images
    output_folder : Path
        Folder where JPEG files will be saved
    quality : int
        JPEG quality (1-100). Default is 95.
    
    Returns:
    --------
    dict
        Statistics about the conversion process
    """
    register_heif_opener()

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all HEIC images
    image_files = sorted(input_folder.glob("*.HEIC")) + sorted(input_folder.glob("*.heic"))
    
    print(f"Found {len(image_files)} HEIC images in {input_folder}")
    print(f"Converting to JPEG with quality={quality}...")
    
    stats = {
        "total": len(image_files),
        "successful": 0,
        "failed": 0
    }
    
    for image_path in tqdm(image_files, desc="Converting HEIC to JPEG"):
        output_path = output_folder / (image_path.stem + ".jpg")
        if convert_heic_to_jpeg(image_path, output_path, quality):
            stats["successful"] += 1
        else:
            stats["failed"] += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"Total: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output folder: {output_folder}")
    
    return stats


if __name__ == "__main__":
    # Get paths
    script_dir = Path(__file__).parent.parent.parent
    input_folder = script_dir / "data" / "images" / "collected_data"
    output_folder = script_dir / "data" / "images" / "converted_jpeg"
    
    print(f"Converting HEIC images to JPEG...")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}\n")
    
    stats = convert_heic_folder_to_jpeg(input_folder, output_folder, quality=95)
