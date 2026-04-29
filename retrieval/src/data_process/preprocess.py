from PIL import Image
from pathlib import Path
import piexif
import pandas as pd
from tqdm import tqdm

def read_heic_with_location(image_path: Path) -> dict:
    image_path = Path(image_path)

    from pillow_heif import register_heif_opener
    register_heif_opener()
    
    image = Image.open(image_path)

    try:
        exif_dict = piexif.load(image_path)
        gps_data = exif_dict.get("GPS", {})

        if gps_data:
            lat = gps_data.get(piexif.GPSIFD.GPSLatitude)
            lon = gps_data.get(piexif.GPSIFD.GPSLongitude)

            if lat and lon:
                lat_decimal = dms_to_decimal(lat)
                lon_decimal = dms_to_decimal(lon)
                return {
                    "image": image,
                    "latitude": lat_decimal,
                    "longitude": lon_decimal
                }
            
    except Exception as e:
        print(f"Could not extract GPS data from {image_path.name}: {e}")

    
    return {"image": image, "latitude": None, "longitude": None}

    
def dms_to_decimal(dms):
    """Convert GPS DMS (degrees, minutes, seconds) to decimal degrees."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    return degrees + (minutes / 60) + (seconds / 3600)


def extract_gps_from_images(image_folder: str | Path, output_csv: str | Path = None) -> pd.DataFrame:
    """
    Extract GPS metadata from all images in a folder and save to CSV.
    
    Parameters:
    -----------
    image_folder : str | Path
        Path to folder containing HEIC images
    output_csv : str | Path, optional
        Path to save the CSV file. If None, saves as 'gps_data.csv' in the image folder's parent
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: image_name, latitude, longitude
    """
    image_folder = Path(image_folder)
    
    if output_csv is None:
        output_csv = image_folder.parent / "gps_data.csv"
    else:
        output_csv = Path(output_csv)
    
    # Get all HEIC images
    image_files = sorted(image_folder.glob("*.HEIC")) + sorted(image_folder.glob("*.heic"))
    
    print(f"Found {len(image_files)} HEIC images in {image_folder}")
    
    gps_records = []
    
    for image_path in tqdm(image_files, desc="Extracting GPS data"):
        data = read_heic_with_location(image_path)
        
        gps_records.append({
            "image_name": image_path.name,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude")
        })
    
    # Create DataFrame
    df = pd.DataFrame(gps_records)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nGPS data saved to: {output_csv}")
    print(f"Total images processed: {len(df)}")
    print(f"Images with GPS data: {df['latitude'].notna().sum()}")
    print(f"Images without GPS data: {df['latitude'].isna().sum()}")
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # Get the parent directory of this script
    script_dir = Path(__file__).parent.parent
    image_folder = script_dir / "data" / "images" / "collected_data"
    output_csv = script_dir / "data" / "images" / "gps_data.csv"
    
    print(f"Processing images from: {image_folder}")
    df = extract_gps_from_images(image_folder, output_csv)

