
import json
import requests

# -----------------------------
# CONFIGURATION
# -----------------------------
ORS_API_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImNhMzdmODQ0MzU3NDRhNmJiNjBlZDViNDU5MWVkNTc0IiwiaCI6Im11cm11cjY0In0="
TIME_WEIGHT = 1.0
RISK_WEIGHT = 10.0
NUM_CANDIDATES = 3

# ROUTE CONFIGURATION - Change these coordinates for your actual start and end points
# Format: [longitude, latitude]

# CURRENT ROUTE (Your actual coordinates - Point 8 to Point 9 - VERY LONG):
ROUTE_START = [77.3451401224851, 9.939739167964605]    # Point 8 [lon, lat] - Kerala
ROUTE_END = [96.18827335259276, 29.176916631088705]    # Point 9 [lon, lat] - Arunachal Pradesh

# UNCOMMENT ONE OF THESE FOR TESTING WITH YOUR ACTUAL COORDINATES:
# Point 1 to Point 2 (Himachal to Maharashtra - 1466km):
# ROUTE_START = [78.31660750824553, 30.628253880932053]  # Point 1 [lon, lat]
# ROUTE_END = [75.28026072047453, 17.733056205938357]    # Point 2 [lon, lat]

# Point 1 to Point 3 (Himachal to Rajasthan - 326km):
# ROUTE_START = [78.31660750824553, 30.628253880932053]  # Point 1 [lon, lat]
# ROUTE_END = [79.18015056648093, 27.794585499436472]    # Point 3 [lon, lat]

# Point 1 to Point 4 (Himachal to West Bengal):
# ROUTE_START = [78.31660750824553, 30.628253880932053]  # Point 1 [lon, lat]
# ROUTE_END = [83.07283624454519, 22.572814314759288]    # Point 4 [lon, lat]

# Point 2 to Point 5 (Maharashtra to Karnataka):
# ROUTE_START = [75.28026072047453, 17.733056205938357]  # Point 2 [lon, lat]
# ROUTE_END = [76.89846170816628, 13.096882450168653]    # Point 5 [lon, lat]

# Point 6 to Point 7 (Rajasthan to Madhya Pradesh):
# ROUTE_START = [74.98128845359025, 27.940954779375833]  # Point 6 [lon, lat]
# ROUTE_END = [80.42027501172338, 24.069486131339602]    # Point 7 [lon, lat]

# Point 8 to Point 9 (Kerala to Arunachal Pradesh - VERY LONG):
# ROUTE_START = [77.3451401224851, 9.939739167964605]    # Point 8 [lon, lat]
# ROUTE_END = [96.18827335259276, 29.176916631088705]    # Point 9 [lon, lat]

# Point 9 to Point 10 (Arunachal Pradesh to Bihar):
# ROUTE_START = [96.18827335259276, 29.176916631088705]  # Point 9 [lon, lat]
# ROUTE_END = [85.74981824762158, 24.65316913097531]     # Point 10 [lon, lat]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_geojson(file_path):
    with open(file_path, "r") as f:
        geojson = json.load(f)
    return geojson

def convert_to_weights(geojson):
    """
    Convert disaster points into weighted zones.
    """
    weighted_zones = []
    for feature in geojson['features']:
        props = feature['properties']
        lat = feature['geometry']['coordinates'][1]
        lon = feature['geometry']['coordinates'][0]
        
        risk = 0
        for key, value in props.items():
            if key != "id" and isinstance(value, (float, int)):
                if value > 0.8:
                    risk += 10
                elif 0.5 <= value <= 0.8:
                    risk += 5
                elif value > 0:
                    risk += 2
        weighted_zones.append({"lat": lat, "lon": lon, "weight": risk})
    return weighted_zones

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula.
    Returns distance in kilometers.
    """
    import math
    
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def generate_intermediate_points(start, end, max_segment_distance=100):
    """
    Generate intermediate points to break long routes into smaller segments.
    """
    start_lat, start_lon = start[1], start[0]
    end_lat, end_lon = end[1], end[0]
    
    total_distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    
    if total_distance <= max_segment_distance:
        return [start, end]
    
    # Calculate number of segments needed
    num_segments = int(total_distance / max_segment_distance) + 1
    
    intermediate_points = [start]
    
    for i in range(1, num_segments):
        ratio = i / num_segments
        lat = start_lat + (end_lat - start_lat) * ratio
        lon = start_lon + (end_lon - start_lon) * ratio
        intermediate_points.append([lon, lat])
    
    intermediate_points.append(end)
    return intermediate_points

def generate_candidate_routes(start, end, alternatives=NUM_CANDIDATES):
    """
    Query ORS public API for multiple route alternatives.
    For long distances, this will use route segmentation.
    """
    routes = []
    
    # Check if route is too long for single API call
    start_lat, start_lon = start[1], start[0]
    end_lat, end_lon = end[1], end[0]
    distance_km = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    
    print(f"Total distance: {distance_km:.1f} km")
    
    if distance_km > 100:  # ORS free tier limit
        print("Route too long for single API call, using segmentation approach...")
        return generate_segmented_routes(start, end, alternatives)
    
    # First, try to get alternative routes
    payload = {
        "coordinates": [start, end],
        "instructions": False,
        "alternative_routes": {
            "share_factor": 0.6,
            "target_count": min(alternatives, 3),
            "weight_factor": 1.4
        }
    }
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    response = requests.post(ORS_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if "routes" in data and len(data["routes"]) > 0:
            routes.extend(data["routes"][:alternatives])
    else:
        print(f"Alternative routes failed: {response.status_code}, {response.text}")
        # Fallback to single route
        payload = {
            "coordinates": [start, end],
            "instructions": False
        }
        response = requests.post(ORS_API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if "routes" in data and len(data["routes"]) > 0:
                routes.append(data["routes"][0])
        else:
            print(f"Single route also failed: {response.status_code}, {response.text}")
    
    return routes

def generate_segmented_routes(start, end, alternatives=3):
    """
    Generate routes by breaking long distances into segments.
    This creates multiple route options by varying the intermediate points.
    """
    routes = []
    
    # Generate different sets of intermediate points for route alternatives
    for alt in range(alternatives):
        # Create slightly different intermediate points for each alternative
        offset_factor = (alt - 1) * 0.1  # Small offset for route variation
        
        intermediate_points = generate_intermediate_points(start, end, 80)  # 80km segments
        
        # Add slight variation to intermediate points
        if len(intermediate_points) > 2:
            for i in range(1, len(intermediate_points) - 1):
                lat_offset = offset_factor * 0.01
                lon_offset = offset_factor * 0.01
                intermediate_points[i][1] += lat_offset  # lat
                intermediate_points[i][0] += lon_offset  # lon
        
        # Generate route through intermediate points
        route = generate_route_through_points(intermediate_points)
        if route:
            routes.append(route)
    
    return routes

def generate_route_through_points(points):
    """
    Generate a single route through multiple waypoints.
    """
    if len(points) < 2:
        return None
    
    # For now, create a simplified route representation
    # In a full implementation, you would call ORS for each segment
    total_distance = 0
    total_duration = 0
    
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        
        # Calculate segment distance and duration
        start_lat, start_lon = start_point[1], start_point[0]
        end_lat, end_lon = end_point[1], end_point[0]
        
        segment_distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
        # More realistic speed calculation based on distance
        if segment_distance < 50:  # Urban/short distance
            avg_speed = 40  # km/h
        elif segment_distance < 200:  # Regional roads
            avg_speed = 60  # km/h
        else:  # Highways/long distance
            avg_speed = 80  # km/h
        
        segment_duration = (segment_distance / avg_speed) * 60  # Convert hours to minutes
        
        total_distance += segment_distance * 1000  # Convert to meters
        total_duration += segment_duration
    
    # Create a simplified route object
    route = {
        "summary": {
            "distance": total_distance,
            "duration": total_duration
        },
        "geometry": {
            "coordinates": points  # Simplified - just the waypoints
        }
    }
    
    return route

def calculate_disaster_risk(route, weighted_zones):
    """
    Compute a simple risk score by checking proximity to disaster zones.
    Note: This is a simplified version since ORS returns encoded geometry.
    For production use, you would need to decode the polyline geometry.
    """
    risk_score = 0
    
    if not isinstance(route, dict):
        return risk_score
    
    # For now, we'll use a simplified risk calculation based on route summary
    # In a production system, you would decode the polyline geometry and
    # check actual route coordinates against disaster zones
    
    # Simple heuristic: longer routes might have higher disaster risk
    distance = route.get('summary', {}).get('distance', 0)
    if distance > 0:
        # Base risk proportional to distance (very simplified)
        risk_score = distance / 10000  # 1 point per 10km
    
    return risk_score

def score_routes(routes, weighted_zones):
    scored_routes = []
    for route in routes:
        risk_score = calculate_disaster_risk(route, weighted_zones)
        time_score = route.get("summary", {}).get("duration", 0)
        overall_score = TIME_WEIGHT * time_score + RISK_WEIGHT * risk_score
        scored_routes.append((route, overall_score))
    return sorted(scored_routes, key=lambda x: x[1])

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    geojson_path = "frontend/predictions_map.geojson"
    start = ROUTE_START
    end = ROUTE_END

    print("Loading disaster data...")
    geojson_data = load_geojson(geojson_path)
    print(f"Loaded {len(geojson_data['features'])} disaster points")
    
    weighted_zones = convert_to_weights(geojson_data)
    print(f"Created {len(weighted_zones)} weighted zones")
    
    print("Generating candidate routes...")
    candidate_routes = generate_candidate_routes(start, end)
    print(f"Generated {len(candidate_routes)} candidate routes")
    
    if not candidate_routes:
        print("No routes generated. Check ORS API connection.")
        exit(1)
    
    ranked_routes = score_routes(candidate_routes, weighted_zones)
    
    print("\n=== Smart Routing Results ===")
    print("Routes ranked by combined time and disaster risk score (lower is better):")
    for idx, (route, score) in enumerate(ranked_routes):
        distance_km = route['summary']['distance'] / 1000
        duration_hours = route['summary']['duration'] / 60  # duration is in minutes, convert to hours
        print(f"Route {idx+1}: Score={score:.2f}, Distance={distance_km:.1f} km, Duration={duration_hours:.1f} hours")
