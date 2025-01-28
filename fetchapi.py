import requests
import json
import os
import math
import logging
from dotenv import load_dotenv

load_dotenv()  # Load MAPBOX_API_KEY from .env

MAPBOX_API_KEY = os.getenv('MAPBOX_API_KEY')

# ---------------- LOGGING SETUP ----------------
logger = logging.getLogger("RouteGenerator")
logger.setLevel(logging.DEBUG)  

file_handler = logging.FileHandler("my_routes.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------- Utility Functions ----------------

def haversine_distance(coord1, coord2):
    """Calculate Haversine distance (in meters) between two (lon, lat) points."""
    R = 6371000  # Earth radius in meters
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def parse_road_summary(summary):
    """
    Assign speed limit and lanes based on route summary.
    Examples:
      A  => 140 km/h and ~2.5 lanes (autostrada)
      S  => 120 km/h and 2 lanes  (ekspresowa)
      GP => 100 km/h and 2 lanes  (główna przyspieszona)
      G  => 90 km/h  and 1 lanes  (główna)
      Z  => 70 km/h  and 1 lane   (zbiorcza)
      L  => 60 km/h  and 1 lane   (lokalna)
      D  => 50 km/h  and 1 lane   (dojazdowa)
    """
    summary = summary.upper()  # just to simplify checks

    if 'GP' in summary:
        return 100, 2
    elif 'A' in summary:
        return 140, 2.5
    elif 'S' in summary:
        return 120, 2
    elif 'G' in summary:
        return 90, 1
    elif 'Z' in summary:
        return 70, 1
    elif 'L' in summary:
        return 60, 1
    elif 'D' in summary:
        return 50, 1
    elif 'E' in summary:
        return 90, 1
    else:
        # Fallback if we find none of the letters above
        return 50, 1


def city_is_near_route(city_coord, route_coords, threshold=25000):
    """
    Returns True if city_coord is within `threshold` (meters)
    of ANY coordinate in route_coords.
    route_coords is route['geometry']['coordinates'], i.e., a list of [lon, lat].
    """
    for rc in route_coords:
        dist = haversine_distance(rc, city_coord)
        if dist <= threshold:
            return True
    return False


# ---------------- Core Logic ----------------

def get_route_between_cities(city1_name, city2_name, city1_coords, city2_coords):
    """
    Calls the Mapbox Directions API to get the route from city1 -> city2.
    Returns the entire JSON response for further parsing (multiple routes, steps, etc.).
    """
    url = (
        f"https://api.mapbox.com/directions/v5/mapbox/driving/"
        f"{city1_coords[0]},{city1_coords[1]};{city2_coords[0]},{city2_coords[1]}"
        f"?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token={MAPBOX_API_KEY}"
    )
    logger.debug(f"Requesting route from {city1_name} to {city2_name} via: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        logger.debug(f"Received route data for {city1_name} -> {city2_name}.")
        return data
    else:
        logger.error(f"Mapbox Directions API returned {response.status_code} for {city1_name} -> {city2_name}")
        return None


def parse_route_and_split_on_cities(
    route_data,
    start_city_name,
    end_city_name,
    cities_coordinates,
    city_index_map,
    threshold=20000.0  # e.g. 20 km
):
    route = route_data['routes'][0]
    steps = route['legs'][0]['steps']
    route_coords = route['geometry']['coordinates']  # entire path coordinates

    route_summary = route['legs'][0]['summary']
    speed_limit, number_of_lanes = parse_road_summary(route_summary)

    # Distances
    total_distance = route['legs'][0]['distance']  
    accumulated_distance = 0.0  # from the very start
    segment_distance = 0.0      # from the last cut

    segments = []
    current_start_city = start_city_name
    visited_intermediate_cities = set()

    for step in steps:
        step_distance = step['distance']
        intersections = step.get('intersections', [])

        accumulated_distance += step_distance
        segment_distance     += step_distance

        for intersection in intersections:
            location = intersection.get('location')
            if not location:
                continue

            for city_name, city_coord in cities_coordinates.items():
                # skip if city is start or final or already visited
                if city_name in (current_start_city, end_city_name, *visited_intermediate_cities):
                    continue

                dist_to_city = haversine_distance(location, city_coord)

                if dist_to_city <= threshold:
                    # Also check city is truly “on the route”
                    if not city_is_near_route(city_coord, route_coords, threshold):
                        continue

                    # --- NEW LOGIC: Compare segment_distance to center-distance ---
                    center_distance = haversine_distance(
                        cities_coordinates[current_start_city],
                        city_coord
                    )
                    # If the route distance from last cut is < e.g. 70% of the direct center distance,
                    # then skip it because it’s too soon or a false partial cut near the outskirts.
                    if segment_distance < (0.7 * center_distance) or segment_distance > (1.3 * center_distance):
                        continue
                    else:
                        logger.debug(
                            f"Cut between {current_start_city} and {city_name} "
                            f"during {start_city_name}->{end_city_name}. "
                            f"segment_dist={segment_distance:.1f}m center_dist={center_distance:.1f}m"
                        )

                    segments.append({
                        "start_city": current_start_city,
                        "end_city": city_name,
                        "distance": segment_distance,
                        "road_class": route_summary,
                        "speed_limit": speed_limit,
                        "number_of_lanes": number_of_lanes
                    })

                    visited_intermediate_cities.add(city_name)

                    current_start_city = city_name
                    segment_distance = 0.0
                    break  # city loop

            if city_name == current_start_city:
                break

    # Add final
    if current_start_city != end_city_name and segment_distance > 0:
        segments.append({
            "start_city": current_start_city,
            "end_city": end_city_name,
            "distance": segment_distance,
            "road_class": route_summary,
            "speed_limit": speed_limit,
            "number_of_lanes": number_of_lanes
        })

    return segments


def main():
    # 1) Load city coordinates
    with open("cities_coordinates.json", "r", encoding="utf-8") as f:
        cities_coordinates = json.load(f)

    city_names = list(cities_coordinates.keys())
    city_index_map = {name: i for i, name in enumerate(city_names)}

    # Prepare final data structure
    city_data = {"cites": {}}
    for city_name in city_names:
        city_data["cites"][city_name] = {
            "id": city_index_map[city_name],
            "routes": []
        }

    n = len(city_names)
    logger.info(f"Starting route generation for {n} cities...")

    # 2) Generate routes between all unique pairs of cities (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            cityA = city_names[i]
            cityB = city_names[j]
            coordA = cities_coordinates[cityA]
            coordB = cities_coordinates[cityB]

            # 3) Call Mapbox to get route
            route_data = get_route_between_cities(cityA, cityB, coordA, coordB)
            if not route_data or 'routes' not in route_data or len(route_data['routes']) == 0:
                logger.warning(f"No valid route for {cityA} -> {cityB}. Skipping.")
                continue

            # 4) Split route if we pass near other large cities
            segments_AtoB = parse_route_and_split_on_cities(
                route_data, 
                cityA, 
                cityB, 
                cities_coordinates, 
                city_index_map,
                threshold=25000
            )

            # 5) Record each segment in city_data 
            for seg in segments_AtoB:
                s_city = seg["start_city"]
                e_city = seg["end_city"]
                dist   = seg["distance"]
                rc     = seg["road_class"]     # The route summary as 'road_class' 
                sl     = seg["speed_limit"]
                ln     = seg["number_of_lanes"]

                city_data["cites"][s_city]["routes"].append({
                    "name": rc,
                    "road_length": dist,
                    "speed_limit": sl,
                    "end_node_name": e_city,
                    "end_node_id": city_index_map[e_city],
                    "number_of_lanes": ln
                })

                # Also add the reverse route (2-way)
                city_data["cites"][e_city]["routes"].append({
                    "name": rc,
                    "road_length": dist,
                    "speed_limit": sl,
                    "end_node_name": s_city,
                    "end_node_id": city_index_map[s_city],
                    "number_of_lanes": ln
                })

    # 7) Save final data
    with open("cities_data.json", "w", encoding="utf-8") as jf:
        json.dump(city_data, jf, ensure_ascii=False, indent=4)

    logger.info("Finished generating routes. Results saved to cities_data.json.")


if __name__ == "__main__":
    main()
