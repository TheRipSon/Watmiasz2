import requests
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables (specifically MAPBOX_API_KEY) from .env
load_dotenv()
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

# --------------- LOGGING SETUP ---------------
logger = logging.getLogger("GeoFetcher")
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler("my_geo.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# Console handler (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --------------- LIST OF 50 POLISH CITIES ---------------
cities = [
    "Warszawa", "Łódź", "Kraków", "Wrocław", "Poznań", "Gdańsk", "Bydgoszcz", "Lublin", "Białystok", "Katowice", "Gorzów Wielkopolski", "Toruń", "Zielona Góra", "Opole", "Rzeszów", "Olsztyn", "Szczecin", "Kielce", "Częstochowa", "Radom", "Płock", "Gliwice", "Rybnik", "Bielsko-Biała", "Koszalin", "Legnica", "Kalisz", "Piła", "Leszno", "Tarnów", "Przemyśl", "Ostrołęka", "Słupsk", "Konin", "Piotrków Trybunalski", "Inowrocław", "Ostrów Wielkopolski", "Zamość", "Puławy", "Jelenia Góra", "Nowy Sącz", "Suwałki", "Gniezno", "Wałbrzych", "Świnoujście", "Chełm", "Zakopane", "Malbork", "Kołobrzeg", "Grudziądz"
]
def get_city_coordinates(city_name: str):
    """
    Fetch coordinates (longitude, latitude) from Mapbox Geocoding API for a given city name.
    Returns a tuple (lon, lat) or None if not found/error.
    """
    # You can adjust query parameters like `limit` or `language`
    url = (
        f"https://api.mapbox.com/geocoding/v5/mapbox.places/"
        f"{city_name}.json?access_token={MAPBOX_API_KEY}&limit=1&language=pl"
    )
    logger.debug(f"Requesting coordinates for city: {city_name}, URL: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'features' in data and len(data['features']) > 0:
            # Usually features[0] is the best match
            lon, lat = data['features'][0]['geometry']['coordinates']
            logger.debug(f"Got coords for {city_name}: lon={lon}, lat={lat}")
            return (lon, lat)
        else:
            logger.warning(f"No features returned for city: {city_name}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching city {city_name}: {e}")
        return None

def main():
    logger.info("Starting coordinate fetch for 50 Polish cities...")
    city_coords = {}

    for city in cities:
        coords = get_city_coordinates(city)
        if coords is not None:
            # Store in dictionary as "CityName": [lon, lat]
            city_coords[city] = [coords[0], coords[1]]
        else:
            logger.warning(f"Skipping city {city}, no valid coordinates found.")

    # Save all fetched coordinates to JSON
    with open("cities_coordinates.json", "w", encoding="utf-8") as f:
        json.dump(city_coords, f, ensure_ascii=False, indent=4)

    logger.info(f"Finished. Coordinates saved to cities_coordinates.json. "
                f"Total valid entries: {len(city_coords)}")

if __name__ == "__main__":
    main()
