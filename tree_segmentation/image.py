from PIL import Image
from PIL.ExifTags import TAGS

def get_gps_data(image_path, tag_id=34853) -> dict:
    """Extracts gps metadata from an image.
    This function is based on DJI FC 6310 camera, it might need some modification
    to be able to extract gps-metadata from images taken with another camera.

    @param image_path: Image to extract from.
    @param tag_id: Tag id for gps-metadata.
    @return: {"latitude": (0,0,0), "longitude": (0,0,0), "altitude": 0.0)} or None if fail.
    """

    try:
        metadata = Image.open(image_path).getexif()
        gps_data = metadata.get(tag_id)
        tag = TAGS.get(tag_id, tag_id)

        if tag != "GPSInfo":
            raise TypeError("GPSInfo was not loaded.", tag, " was loaded.")

    except Exception as e:
        print(e)
        return None
    else:
        return {"latitude": gps_data[2], "longitude": gps_data[4], "altitude": float(gps_data[6])}

