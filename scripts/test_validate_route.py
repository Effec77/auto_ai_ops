"""
Enhanced Smart Route Validation & Rerouting Test
Implements all recommended improvements for comprehensive disaster-aware routing
"""

import json
import math
import os
import random
import requests
import base64
import polyline
import time
from shapely.geometry import LineString, Point
from datetime import datetime

# ---------- CONFIG ----------
ENCODED_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImNhMzdmODQ0MzU3NDRhNmJiNjBlZDViNDU5MWVkNTc0IiwiaCI6Im11cm11cjY0In0="

# Decode the API key
try:
    decoded_bytes = base64.b64decode(ENCODED_API_KEY)
    decoded_json = json.loads(decoded_bytes.decode('utf-8'))
    ORS_API_KEY = decoded_json.get('id', '')
    print(f"🔑 Decoded API Key: {ORS_API_KEY}")
except Exception as e:
    print(f"❌ Error decoding API key: {e}")
    ORS_API_KEY = "5b3ce3597851110001cf62448"

GEOJSON_PATH = "frontend/predictions_map.geojson"
CRITICAL_THRESHOLD = 15
MAX_SEGMENT_KM = 800
OUTPUT_DIR = "route_exports"  # Directory for exported routes

# Enhanced test configurations
BUFFER_SIZES = [2.0, 5.0, 10.0]  # Test different disaster impact radii
DISASTER_TEST_SCENARIOS = [
    {"weight": 25, "description": "Minor road damage", "radius_km": 3.0},
    {"weight": 75, "description": "Major flooding", "radius_km": 8.0},
    {"weight": 150, "description": "Critical bridge collapse", "radius_km": 15.0}
]

# Sample routes
TEST_ROUTES = {
    "Himachal to Maharashtra": [[77.1734, 31.1048], [72.8777, 19.0760]],
    "Himachal to Rajasthan": [[77.1734, 31.1048], [75.7873, 26.9124]], 
    "Kerala to Arunachal": [[76.2711, 9.9312], [93.6167, 27.1004]]
}

# Inland routing hubs (approximate) to keep stitched fallbacks on land
INDIA_ROUTING_HUBS = [
    [72.8777, 19.0760],   # Mumbai
    [73.8567, 18.5204],   # Pune
    [77.1025, 28.7041],   # Delhi
    [76.2711, 9.9312],    # Kochi
    [80.2707, 13.0827],   # Chennai
    [78.4867, 17.3850],   # Hyderabad
    [77.5946, 12.9716],   # Bengaluru
    [75.8577, 30.9000],   # Ludhiana (approx)
    [75.7873, 26.9124],   # Jaipur
    [79.0882, 21.1458],   # Nagpur
    [88.3639, 22.5726],   # Kolkata
    [91.7362, 26.1445],   # Guwahati
    [85.1376, 25.5941],   # Patna
    [81.8463, 25.4358]    # Prayagraj
]

# ---------- ENHANCED HELPERS ----------
def setup_output_directory():
    """Create output directory for route exports"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

def load_geojson(path):
    with open(path, "r") as f:
        return json.load(f)

def convert_to_weights(geojson):
    weighted_zones = []
    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        coords = feature.get('geometry', {}).get('coordinates', [None, None])
        lon, lat = coords[0], coords[1]
        risk = 0
        disaster_types = []
        
        for key, value in props.items():
            if key != "id" and isinstance(value, (float, int)):
                if value > 0.8:
                    risk += 10
                elif 0.5 <= value <= 0.8:
                    risk += 5
                elif value > 0:
                    risk += 2
                
                if value > 0.5:
                    disaster_types.append(key.replace('_', ' ').title())
        
        weighted_zones.append({
            "lat": lat, "lon": lon, "weight": risk, 
            "types": disaster_types, "radius_km": 5.0
        })
    return weighted_zones

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def interpolate_line(start, end, n_points=20):
    lon1, lat1 = start
    lon2, lat2 = end
    pts = []
    for i in range(n_points + 1):
        t = i / n_points
        lon = lon1 + (lon2 - lon1) * t
        lat = lat1 + (lat2 - lat1) * t
        pts.append([lon, lat])
    return pts

# ---------- AVOIDANCE HELPERS ----------
def make_circle_polygon(lon, lat, radius_km=2.0, num_points=8):
    """Approximate a circle as a simple polygon for ORS avoid_polygons."""
    # Use smaller radius and fewer points for better ORS compatibility
    radius_km = min(radius_km, 3.0)  # Cap at 3km for better success rate
    deg = radius_km * 0.009  # Rough conversion: 1 degree ≈ 111km
    
    pts = []
    for i in range(num_points):
        t = 2 * math.pi * (i / num_points)
        pts.append([lon + deg * math.cos(t), lat + deg * math.sin(t)])
    pts.append(pts[0])  # Close the polygon
    
    return {
        "type": "Polygon",
        "coordinates": [pts]
    }

def build_avoid_multipolygon(disaster_points, default_radius_km=2.0):
    """Create a MultiPolygon around disaster points for ORS options.avoid_polygons."""
    polygons = []
    max_polygons = 3  # Limit number of polygons to avoid ORS complexity issues
    
    for i, p in enumerate(disaster_points[:max_polygons]):  # Limit to first few zones
        lon, lat = p.get("location", (None, None))
        if lon is None or lat is None:
            continue
        
        # Use smaller radius for better ORS compatibility
        radius = min(p.get("radius_km", default_radius_km), 5.0)  # Cap at 5km
        poly = make_circle_polygon(lon, lat, radius)
        polygons.append(poly["coordinates"][0])
    
    if not polygons:
        return None
    
    # For single polygon, return Polygon instead of MultiPolygon
    if len(polygons) == 1:
        return {
            "type": "Polygon",
            "coordinates": polygons
        }
    
    # Convert list of rings to MultiPolygon
    return {
        "type": "MultiPolygon",
        "coordinates": [[ring] for ring in polygons]
    }

# ---------- ENHANCED RISK CALCULATION ----------
def calculate_segment_risk_shapely(linestring, weighted_zones, buffer_km=2.0):
    """Enhanced risk calculation with detailed logging"""
    deg_buffer = buffer_km * 0.009
    buffered = linestring.buffer(deg_buffer)
    score = 0
    affected_zones = []
    
    for z in weighted_zones:
        if z["lat"] is None or z["lon"] is None:
            continue
        p = Point(z["lon"], z["lat"])
        if buffered.contains(p):
            score += z["weight"]
            affected_zones.append({
                "weight": z["weight"],
                "types": z.get("types", []),
                "location": (z["lon"], z["lat"])
            })
    
    return score, affected_zones

def calculate_route_risk(route, weighted_zones, buffer_km=2.0):
    """Calculate risk with detailed analysis"""
    coords = route.get('geometry', {}).get('coordinates', [])
    if not coords:
        return 0.0, []
    
    ls = LineString(coords)
    score, affected_zones = calculate_segment_risk_shapely(ls, weighted_zones, buffer_km)
    return score, affected_zones

def validate_route_realtime(route, weighted_zones, critical_threshold=CRITICAL_THRESHOLD, buffer_km=2.0):
    """Enhanced validation with detailed reporting"""
    score, affected_zones = calculate_route_risk(route, weighted_zones, buffer_km)
    
    status_details = {
        "risk_score": score,
        "is_safe": score < critical_threshold,
        "affected_zones": affected_zones,
        "buffer_km": buffer_km,
        "critical_threshold": critical_threshold
    }
    
    if score >= critical_threshold:
        return False, f"Compromised (risk={score:.1f}, zones={len(affected_zones)})", score, status_details
    return True, f"Clear (risk={score:.1f}, buffer={buffer_km}km)", score, status_details

# ---------- VISUAL DEBUGGING FUNCTIONS ----------
def export_route_geojson(route, filename, risk_level=0, rerouted=False, route_info=None):
    """
    Smart GeoJSON export with risk-based styling and metadata for frontend integration.
    
    Args:
        route: Route dictionary with geometry and summary
        filename: Output filename
        risk_level: Risk score (0-150+)
        rerouted: Whether this is a rerouted path
        route_info: Additional route metadata
    """
    route_info = route_info or {}
    
    # Determine color based on risk level and reroute status
    def get_route_color(risk, is_rerouted):
        if is_rerouted:
            if risk <= 25:
                return "#fbbf24"  # Yellow - low risk reroute
            elif risk <= 75:
                return "#f97316"  # Orange - medium risk reroute
            else:
                return "#dc2626"  # Red - high risk reroute
        else:
            if risk == 0:
                return "#2563eb"  # Blue - safe primary route
            elif risk <= 25:
                return "#16a34a"  # Green - low risk primary
            elif risk <= 75:
                return "#f59e0b"  # Amber - medium risk primary
            else:
                return "#dc2626"  # Red - high risk primary
    
    # Determine route style
    color = get_route_color(risk_level, rerouted)
    dash_array = "10,5" if rerouted else None
    weight = 5 if rerouted else 4
    
    # Enhanced properties for frontend styling and tooltips
    properties = {
        "name": route_info.get("name", "Route"),
        "distance_km": route["summary"].get("distance", 0),
        "duration_hours": route["summary"].get("duration", 0),
        "risk_level": risk_level,
        "risk_score": risk_level,  # Alias for backward compatibility
        "is_safe": risk_level < CRITICAL_THRESHOLD,
        "rerouted": rerouted,
        "route_type": route_info.get("route_type", "rerouted" if rerouted else "original"),
        "color": color,
        "weight": weight,
        "dashArray": dash_array,
        "opacity": 0.8,
        "affected_zones": route_info.get("affected_zones", 0),
        "disaster_scenario": route_info.get("disaster_scenario", ""),
        "reroute_success": route_info.get("reroute_success", rerouted),
        "export_time": datetime.now().isoformat(),
        **route_info
    }
    
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": route["geometry"]["coordinates"]
            },
            "properties": properties
        }]
    }
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    route_type_emoji = "🔄" if rerouted else "📍"
    risk_emoji = "🚨" if risk_level >= CRITICAL_THRESHOLD else "✅"
    print(f"📄 {route_type_emoji} Exported {properties['route_type']} route: {filepath}")
    print(f"    {risk_emoji} Risk: {risk_level}, Color: {color}, Zones: {properties['affected_zones']}")
    
    return filepath

# Backward compatibility wrapper
def export_route_to_geojson(route, filename, route_info=None):
    """Legacy function for backward compatibility"""
    route_info = route_info or {}
    risk_level = route_info.get("risk_score", 0)
    rerouted = route_info.get("reroute_success", False)
    return export_route_geojson(route, filename, risk_level, rerouted, route_info)

def export_all_alternatives_to_geojson(routes, base_filename, route_name):
    """Export all route alternatives for comparison with enhanced styling"""
    for i, route in enumerate(routes):
        filename = f"{base_filename}_alt_{i+1}.geojson"
        
        # Calculate risk for this route (assuming no disasters initially)
        risk_score = 0  # Will be calculated if disasters are present
        
        export_route_geojson(
            route=route,
            filename=filename,
            risk_level=risk_score,
            rerouted=False,
            route_info={
                "name": f"{route_name} - Alternative {i+1}",
                "alternative_number": i+1,
                "total_alternatives": len(routes),
                "route_type": "original"
            }
        )

# ---------- ENHANCED ORS ROUTING ----------
def test_ors_api_key():
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        'Accept': 'application/json, application/geo+json',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
    }
    test_body = {"coordinates": [[77.1734, 31.1048], [77.2734, 31.1048]], "instructions": False}
    try:
        response = requests.post(url, json=test_body, headers=headers, timeout=10)
        print(f"🧪 API Test Response: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return False
        else:
            print(f"   ✅ API Key working!")
            return True
    except Exception as e:
        print(f"🚨 API Test Failed: {e}")
        return False

def fetch_all_ors_routes(start, end, max_alternatives=3):
    """Fetch ALL available route alternatives from ORS"""
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    
    headers = {
        'Accept': 'application/json, application/geo+json',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    def request_single_route(coord_a, coord_b):
        """Request a single route (no alternatives) between two points"""
        single_body = {
            "coordinates": [coord_a, coord_b],
            "instructions": False
        }
        resp = requests.post(url, json=single_body, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None, resp
        data = resp.json()
        route_data = (data.get('routes') or data.get('features') or [])
        if not route_data:
            return None, resp
        rd = route_data[0]
        # geometry may be dict (geojson), encoded polyline string, or list
        geom = rd.get('geometry', {}) if isinstance(rd, dict) else {}
        coords = None
        if isinstance(geom, dict):
            coords = geom.get('coordinates')
        elif isinstance(geom, str):
            try:
                decoded = polyline.decode(geom)
                coords = [[lon, lat] for (lat, lon) in decoded]
            except Exception:
                coords = None
        if not coords:
            # fallback: straight line if no geometry
            coords = interpolate_line(coord_a, coord_b, 50)
        summary = rd.get('summary', {}) if isinstance(rd, dict) else {}
        route = {
            "geometry": {"coordinates": coords},
            "summary": {
                "distance": summary.get('distance', 0) / 1000,
                "duration": summary.get('duration', 0) / 3600
            },
            "alternative_id": 1
        }
        return route, resp

    def compute_reroute_with_avoid(coord_a, coord_b, avoid_geojson):
        """Request a single route while avoiding polygons (ORS options.avoid_polygons)."""
        body = {
            "coordinates": [coord_a, coord_b],
            "instructions": False,
            "options": {
                "avoid_polygons": avoid_geojson
            }
        }
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        routes_list = data.get('routes', [])
        if not routes_list:
            return None
        rd = routes_list[0]
        geom = rd.get('geometry', {})
        coords = None
        if isinstance(geom, dict):
            coords = geom.get('coordinates')
        elif isinstance(geom, str):
            try:
                decoded = polyline.decode(geom)
                coords = [[lon, lat] for (lat, lon) in decoded]
            except Exception:
                coords = None
        if not coords:
            return None
        summary = rd.get('summary', {})
        return {
            "geometry": {"coordinates": coords},
            "summary": {
                "distance": summary.get('distance', 0) / 1000,
                "duration": summary.get('duration', 0) / 3600
            },
            "alternative_id": 1
        }
    
    def build_chunked_route(coord_start, coord_end):
        """For very long distances, split into chunks and stitch single routes.
        If server rejects segment size (error 2004), increase chunk count and retry.
        """
        total_distance_km = haversine(coord_start[1], coord_start[0], coord_end[1], coord_end[0])
        base_steps = max(1, int(total_distance_km // MAX_SEGMENT_KM) + 1)
        max_retries = 3
        attempt = 0
        steps = base_steps
        while attempt <= max_retries:
            waypoints = [
                [
                    coord_start[0] + (coord_end[0] - coord_start[0]) * (i / steps),
                    coord_start[1] + (coord_end[1] - coord_start[1]) * (i / steps)
                ]
                for i in range(steps + 1)
            ]
            stitched_coords = []
            sum_distance = 0.0
            sum_duration = 0.0
            need_retry_smaller_segments = False
            for i in range(len(waypoints) - 1):
                seg_route, seg_resp = request_single_route(waypoints[i], waypoints[i+1])
                if not seg_route:
                    # Check if it's a size error (2004). If so, retry with more steps.
                    try:
                        err_json = seg_resp.json() if seg_resp is not None else {}
                        err_code = err_json.get('error', {}).get('code')
                    except Exception:
                        err_code = None
                    if seg_resp is not None and seg_resp.status_code == 400 and err_code == 2004:
                        need_retry_smaller_segments = True
                        break
                    return None
                seg_coords = seg_route["geometry"]["coordinates"]
                if i > 0 and seg_coords:
                    seg_coords = seg_coords[1:]
                stitched_coords.extend(seg_coords)
                sum_distance += seg_route["summary"].get("distance", 0)
                sum_duration += seg_route["summary"].get("duration", 0)
            if need_retry_smaller_segments:
                attempt += 1
                steps *= 2  # more segments → smaller per-segment distance
                print(f"  ⚠️  Segment too large, increasing chunks to {steps} and retrying")
                continue
            return {
                "geometry": {"coordinates": stitched_coords},
                "summary": {"distance": sum_distance, "duration": sum_duration},
                "alternative_id": 1
            }
        return None

    def build_corridor_route(coord_start, coord_end):
        """Corridor-based fallback: choose inland hubs roughly along the great-circle
        and request single ORS legs between them to avoid over-water segments.
        """
        # Pick up to 3 hubs by proximity to the straight line between start and end
        line = LineString([coord_start, coord_end])
        scored = []
        for hub in INDIA_ROUTING_HUBS:
            p = Point(hub[0], hub[1])
            try:
                dist = line.distance(p)
            except Exception:
                dist = 999
            scored.append((dist, hub))
        scored.sort(key=lambda x: x[0])
        corridor = [coord_start] + [h for _, h in scored[:3]] + [coord_end]

        stitched_coords = []
        sum_distance = 0.0
        sum_duration = 0.0
        for i in range(len(corridor) - 1):
            leg_route, leg_resp = request_single_route(corridor[i], corridor[i+1])
            if not leg_route:
                return None
            leg_coords = leg_route["geometry"]["coordinates"]
            if i > 0 and leg_coords:
                leg_coords = leg_coords[1:]
            stitched_coords.extend(leg_coords)
            sum_distance += leg_route["summary"].get("distance", 0)
            sum_duration += leg_route["summary"].get("duration", 0)
        return {
            "geometry": {"coordinates": stitched_coords},
            "summary": {"distance": sum_distance, "duration": sum_duration},
            "alternative_id": 1
        }
    
    # For very long distances, avoid alternatives and stitch chunks
    direct_distance_km = haversine(start[1], start[0], end[1], end[0])
    if direct_distance_km > MAX_SEGMENT_KM:
        print("  📡 Long route detected — using chunked single-route stitching")
        route = build_chunked_route(start, end)
        if route:
            return [route]
        print("  ⚠️  Chunked stitching failed — trying corridor fallback")
        corridor_route = build_corridor_route(start, end)
        return [corridor_route] if corridor_route else []
    
    body = {
        "coordinates": [start, end],
        "instructions": False,
        "alternative_routes": {
            "target_count": max_alternatives,
            "weight_factor": 1.4,
            "share_factor": 0.6
        }
    }
    
    try:
        print(f"  📡 Requesting {max_alternatives} ORS route alternatives")
        response = requests.post(url, json=body, headers=headers, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            routes = []
            
            for i, route_data in enumerate(data.get('routes', [])):
                geom = route_data.get('geometry', {})
                coords = None
                if isinstance(geom, dict):
                    coords = geom.get('coordinates')
                elif isinstance(geom, str):
                    try:
                        decoded = polyline.decode(geom)
                        coords = [[lon, lat] for (lat, lon) in decoded]
                    except Exception:
                        coords = None
                
                if not coords:
                    coords = interpolate_line(start, end, 50)
                
                summary = route_data.get('summary', {})
                routes.append({
                    "geometry": {"coordinates": coords},
                    "summary": {
                        "distance": summary.get('distance', 0) / 1000,
                        "duration": summary.get('duration', 0) / 3600
                    },
                    "alternative_id": i + 1
                })
            
            # Flag very short/undersampled routes
            for r in routes:
                coords = r["geometry"].get("coordinates", [])
                if len(coords) < 10:
                    r["summary"]["note"] = "undersampled_geometry"
            print(f"  ✅ Retrieved {len(routes)} route alternatives")
            return routes
        else:
            # If request too large (code 2004), retry without alternatives
            try:
                err_json = response.json()
                err_code = err_json.get('error', {}).get('code')
            except Exception:
                err_code = None
            if response.status_code == 400 and err_code == 2004:
                print("  ⚠️  Request too large — retrying with single route (no alternatives)")
                single_route, single_resp = request_single_route(start, end)
                if single_route:
                    return [single_route]
                print("  ⚠️  Single-route request failed — trying corridor fallback")
                corridor_route = build_corridor_route(start, end)
                if corridor_route:
                    return [corridor_route]
            print(f"  ❌ ORS Error {response.status_code}: {response.text[:100]}")
            return []
    except Exception as e:
        print(f"  💥 ORS Request failed: {e}")
        return []

# ---------- PUBLIC REROUTE WRAPPER ----------
def compute_reroute_with_avoid(start, end, avoid_geojson):
    """Compute reroute avoiding polygons with improved error handling.
    Returns a stitched route dict or None.
    """
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        'Accept': 'application/json, application/geo+json',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
    }

    def request_avoid_leg(a, b, attempt_num=1):
        # Try different body formats for better compatibility
        body_formats = [
            {
                "coordinates": [a, b],
                "instructions": False,
                "options": {"avoid_polygons": avoid_geojson}
            },
            {
                "coordinates": [a, b],
                "instructions": False,
                "options": {
                    "avoid_polygons": {
                        "type": "FeatureCollection",
                        "features": [{"type": "Feature", "geometry": avoid_geojson}]
                    }
                }
            }
        ]
        
        for body_idx, body in enumerate(body_formats):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=30)
                
                if resp.status_code == 200:
                    data = resp.json()
                    routes_list = data.get('routes', [])
                    if routes_list:
                        rd = routes_list[0]
                        geom = rd.get('geometry', {})
                        coords = None
                        
                        if isinstance(geom, dict):
                            coords = geom.get('coordinates')
                        elif isinstance(geom, str):
                            try:
                                decoded = polyline.decode(geom)
                                coords = [[lon, lat] for (lat, lon) in decoded]
                            except Exception:
                                coords = None
                        
                        if coords:
                            summary = rd.get('summary', {})
                            return {
                                "coords": coords,
                                "distance": summary.get('distance', 0) / 1000,
                                "duration": summary.get('duration', 0) / 3600
                            }
                else:
                    # Log the error for debugging
                    try:
                        error_data = resp.json()
                        print(f"        ORS Error {resp.status_code}: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    except:
                        print(f"        ORS Error {resp.status_code}: {resp.text[:100]}")
                        
            except requests.exceptions.Timeout:
                print(f"        Request timeout on attempt {attempt_num}, body format {body_idx + 1}")
            except Exception as e:
                print(f"        Request failed: {str(e)[:100]}")
        
        return None

    # For very long routes, try direct routing first (simpler)
    total_km = haversine(start[1], start[0], end[1], end[0])
    
    if total_km <= 500:  # For shorter routes, try direct routing with avoidance
        print(f"        Attempting direct route with avoidance ({total_km:.1f}km)")
        direct_result = request_avoid_leg(start, end, 1)
        if direct_result:
            return {
                "geometry": {"coordinates": direct_result["coords"]},
                "summary": {"distance": direct_result["distance"], "duration": direct_result["duration"]},
                "alternative_id": 1
            }
    
    # For longer routes or if direct failed, use chunking
    print(f"        Using chunked routing ({total_km:.1f}km)")
    # ORS has 150km limit with avoid_polygons, so use much smaller chunks
    max_chunk_km = 100  # Conservative limit well below 150km
    steps = max(2, int(total_km // max_chunk_km) + 1)
    waypoints = [
        [
            start[0] + (end[0] - start[0]) * (i / steps),
            start[1] + (end[1] - start[1]) * (i / steps)
        ]
        for i in range(steps + 1)
    ]
    
    stitched = []
    sum_distance = 0.0
    sum_duration = 0.0
    
    for i in range(len(waypoints) - 1):
        leg = request_avoid_leg(waypoints[i], waypoints[i+1], i + 1)
        if not leg:
            print(f"        Leg {i+1}/{len(waypoints)-1} failed")
            return None
        
        seg = leg["coords"]
        if i > 0 and seg:
            seg = seg[1:]  # Remove duplicate point
        stitched.extend(seg)
        sum_distance += leg["distance"]
        sum_duration += leg["duration"]

    if stitched:
        return {
            "geometry": {"coordinates": stitched},
            "summary": {"distance": sum_distance, "duration": sum_duration},
            "alternative_id": 1
        }
    
    return None

def validate_route_quality(route, start, end):
    """Enhanced route quality validation for rerouted paths"""
    coords = route["geometry"]["coordinates"]
    if len(coords) < 2:
        return False, "Route has insufficient coordinates"

    route_start = coords[0]
    route_end = coords[-1]

    # Check endpoint accuracy
    start_deviation = haversine(start[1], start[0], route_start[1], route_start[0])
    end_deviation = haversine(end[1], end[0], route_end[1], route_end[0])
    if start_deviation > 50:
        return False, f"Route start deviates {start_deviation:.1f}km from expected"
    if end_deviation > 50:
        return False, f"Route end deviates {end_deviation:.1f}km from expected"

    # Check route distance reasonableness
    route_distance = route["summary"].get("distance", 0)
    direct_distance = haversine(start[1], start[0], end[1], end[0])
    
    if route_distance > direct_distance * 4:  # More than 4x direct distance is suspicious
        return False, f"Route too long: {route_distance:.1f}km vs {direct_distance:.1f}km direct (ratio: {route_distance/direct_distance:.1f}x)"
    
    # Over-water heuristic: check for suspiciously straight long segments
    ls = LineString(coords)
    total_len = ls.length
    if total_len > 5.0:  # crude degree scale
        minx, miny, maxx, maxy = ls.bounds
        bbox_diag = math.hypot(maxx - minx, maxy - miny)
        if bbox_diag > 8.0 and total_len / bbox_diag < 1.2:
            return False, "Route appears too straight over long distance (likely over water)"

    # Check for reasonable coordinate density (avoid undersampled routes)
    if len(coords) < 10 and route_distance > 100:
        return False, f"Route undersampled: only {len(coords)} points for {route_distance:.1f}km"

    return True, "Route quality validated"

def validate_reroute_quality(rerouted_route, original_route, start, end):
    """Specific validation for rerouted paths to ensure they're meaningful alternatives"""
    # Basic quality check
    quality_ok, quality_msg = validate_route_quality(rerouted_route, start, end)
    if not quality_ok:
        return False, f"Quality check failed: {quality_msg}"
    
    # Check that reroute is actually different from original
    reroute_distance = rerouted_route["summary"].get("distance", 0)
    original_distance = original_route["summary"].get("distance", 0)
    
    distance_diff = abs(reroute_distance - original_distance)
    if distance_diff < original_distance * 0.05:  # Less than 5% difference
        return False, f"Reroute too similar to original: {distance_diff:.1f}km difference"
    
    # Check that reroute isn't excessively longer
    if reroute_distance > original_distance * 2.5:  # More than 2.5x original
        return False, f"Reroute too long: {reroute_distance:.1f}km vs {original_distance:.1f}km original"
    
    return True, f"Reroute validated: {reroute_distance:.1f}km vs {original_distance:.1f}km original"

# ---------- ENHANCED SMART REROUTING CLASS ----------
class EnhancedSmartReroutingEngine:
    def __init__(self, geojson_path):
        self.geojson_path = geojson_path
        self.disaster_zones = []
        self.load_disaster_zones()
    
    def load_disaster_zones(self):
        try:
            geo = load_geojson(self.geojson_path)
            self.disaster_zones = convert_to_weights(geo)
            print(f"🗺️  Loaded {len(self.disaster_zones)} disaster zones")
        except Exception as e:
            print(f"❌ Error loading disaster zones: {e}")
            self.disaster_zones = []
    
    def simulate_meaningful_disaster_injection(self, route, scenario):
        """Properly inject disaster into disaster zones for realistic testing"""
        coords_list = route["geometry"]["coordinates"]
        mid_idx = len(coords_list) // 2
        mid_coord = coords_list[mid_idx]
        
        # Create new disaster zone that affects risk calculation
        new_disaster = {
            "lat": mid_coord[1],
            "lon": mid_coord[0],
            "weight": scenario["weight"],
            "types": [scenario["description"]],
            "radius_km": scenario["radius_km"]
        }
        
        # Add to disaster zones (this affects risk calculation)
        original_zones = len(self.disaster_zones)
        self.disaster_zones.append(new_disaster)
        
        print(f"💥 Injected {scenario['description']} (weight={scenario['weight']}) at {mid_coord}")
        print(f"   Disaster zones: {original_zones} → {len(self.disaster_zones)}")
        
        return new_disaster
    
    def test_all_route_alternatives(self, start, end):
        """Test and compare risk scores of all available alternatives"""
        routes = fetch_all_ors_routes(start, end, max_alternatives=3)
        
        if not routes:
            print("  ⚠️  No routes from ORS, using fallback")
            routes = [self.create_smart_fallback_route(start, end)]
        
        print(f"\n  📊 Analyzing {len(routes)} route alternatives:")
        
        analyzed_routes = []
        for i, route in enumerate(routes):
            # Validate route quality
            quality_ok, quality_msg = validate_route_quality(route, start, end)
            
            # Test with different buffer sizes
            buffer_results = {}
            for buffer_km in BUFFER_SIZES:
                ok, msg, score, details = validate_route_realtime(route, self.disaster_zones, buffer_km=buffer_km)
                buffer_results[buffer_km] = {
                    "risk_score": score,
                    "is_safe": ok,
                    "affected_zones": len(details["affected_zones"])
                }
            
            analyzed_routes.append({
                "route": route,
                "alternative_id": i + 1,
                "quality_check": quality_ok,
                "quality_message": quality_msg,
                "buffer_analysis": buffer_results,
                "recommended_buffer": self.get_recommended_buffer(buffer_results)
            })
            
            print(f"    Alt {i+1}: {route['summary']['distance']:.1f}km, "
                  f"Quality: {'✅' if quality_ok else '❌'}")
            for buffer_km in BUFFER_SIZES:
                result = buffer_results[buffer_km]
                safety = '✅ Safe' if result["is_safe"] else '🚨 Risk'
                print(f"      {buffer_km}km buffer: {safety} (score={result['risk_score']:.1f})")
        
        return analyzed_routes
    
    def get_recommended_buffer(self, buffer_results):
        """Recommend optimal buffer size based on results"""
        for buffer_km in sorted(BUFFER_SIZES):
            if buffer_results[buffer_km]["affected_zones"] > 0:
                return buffer_km
        return BUFFER_SIZES[0]  # Default to smallest if no zones affected
    
    def create_smart_fallback_route(self, start, end):
        """Create fallback route when ORS fails"""
        distance = haversine(start[1], start[0], end[1], end[0])
        coords = interpolate_line(start, end, max(50, int(distance/10)))
        duration_hours = distance / 60
        
        return {
            "geometry": {"coordinates": coords},
            "summary": {"distance": distance, "duration": duration_hours},
            "alternative_id": 0  # Fallback route
        }
    
    def comprehensive_disaster_test(self, route_name, start, end):
        """Run comprehensive disaster testing with all scenarios"""
        print(f"\n🧪 COMPREHENSIVE DISASTER TESTING: {route_name}")
        print("="*70)
        
        # Get all route alternatives
        analyzed_routes = self.test_all_route_alternatives(start, end)
        
        if not analyzed_routes:
            print("❌ No routes available for testing")
            return
        
        # Export all alternatives for visual debugging
        routes_for_export = [ar["route"] for ar in analyzed_routes]
        export_all_alternatives_to_geojson(routes_for_export, 
                                         route_name.lower().replace(" ", "_"), 
                                         route_name)
        
        # Test each disaster scenario on the best route
        best_route_analysis = analyzed_routes[0]  # Assume first is primary
        best_route = best_route_analysis["route"]
        recommended_buffer = best_route_analysis["recommended_buffer"]
        
        print(f"\n🎯 Testing disaster scenarios on primary route (buffer={recommended_buffer}km):")
        
        for scenario in DISASTER_TEST_SCENARIOS:
            print(f"\n  💥 Scenario: {scenario['description']}")
            
            # Save original disaster zones
            original_zones = self.disaster_zones.copy()
            
            # Inject disaster
            injected_disaster = self.simulate_meaningful_disaster_injection(best_route, scenario)
            
            # Test route with new disaster
            ok, msg, score, details = validate_route_realtime(
                best_route, self.disaster_zones, buffer_km=recommended_buffer
            )
            
            print(f"    Result: {'✅ Safe' if ok else '🚨 REROUTE NEEDED'} — {msg}")
            
            if not ok:
                print(f"    Affected zones: {len(details['affected_zones'])}")
                print(f"    🔄 TRIGGERING EMERGENCY REROUTING...")
                
                # Build avoid polygons from affected zones
                avoid_geojson = build_avoid_multipolygon(details['affected_zones'], default_radius_km=scenario.get('radius_km', 2.0))
                rerouted = None
                
                # Attempt disaster-aware rerouting
                rerouted, reroute_success, reroute_message = self.compute_disaster_aware_reroute(
                    start, end, details['affected_zones'], scenario['description']
                )
                
                if reroute_success and rerouted:
                    print(f"    ✅ REROUTE SUCCESS: {reroute_message}")
                    
                    # Validate reroute quality
                    reroute_quality_ok, reroute_quality_msg = validate_reroute_quality(
                        rerouted, best_route, start, end
                    )
                    
                    if not reroute_quality_ok:
                        print(f"    ⚠️  Reroute quality issue: {reroute_quality_msg}")
                    
                    # Validate the rerouted path doesn't still hit disasters
                    reroute_ok, reroute_msg, reroute_score, reroute_details = validate_route_realtime(
                        rerouted, self.disaster_zones, buffer_km=recommended_buffer
                    )
                    
                    rerouted_filename = f"{route_name.lower().replace(' ', '_')}_rerouted_{scenario['description'].lower().replace(' ', '_')}.geojson"
                    export_route_geojson(
                        route=rerouted,
                        filename=rerouted_filename,
                        risk_level=reroute_score,
                        rerouted=True,
                        route_info={
                            "name": f"{route_name} - Rerouted avoiding {scenario['description']}",
                            "disaster_scenario": scenario['description'],
                            "affected_zones": len(reroute_details['affected_zones']),
                            "original_risk": score,
                            "reroute_success": True,
                            "route_type": "rerouted",
                            "quality_validated": reroute_quality_ok,
                            "quality_message": reroute_quality_msg
                        }
                    )
                else:
                    print(f"    ❌ REROUTE FAILED: {reroute_message}")
                    # Export compromised route if reroute failed
                    compromised_filename = f"{route_name.lower().replace(' ', '_')}_compromised_{scenario['description'].lower().replace(' ', '_')}.geojson"
                    export_route_geojson(
                        route=best_route,
                        filename=compromised_filename,
                        risk_level=score,
                        rerouted=False,
                        route_info={
                            "name": f"{route_name} - Compromised by {scenario['description']}",
                            "disaster_scenario": scenario['description'],
                            "affected_zones": len(details['affected_zones']),
                            "warning": "Reroute not possible",
                            "reroute_success": False,
                            "route_type": "compromised"
                        }
                    )
            
            # Restore original zones for next test
            self.disaster_zones = original_zones
    
    def compute_disaster_aware_reroute(self, start, end, affected_zones, scenario_description):
        """
        Compute a disaster-aware reroute avoiding affected zones with multiple fallback strategies.
        Returns (rerouted_route, success_flag, message)
        """
        if not affected_zones:
            return None, False, "No affected zones to avoid"
        
        print(f"    🔄 Computing reroute avoiding {len(affected_zones)} disaster zones...")
        
        # Strategy 1: Alternative route selection (pick safest existing alternative)
        try:
            print(f"    📍 Strategy 1: Alternative route selection")
            rerouted = self._find_safest_alternative_route(start, end)
            if rerouted:
                return self._validate_and_return_reroute(rerouted, "Strategy 1 (alternatives)")
        except Exception as e:
            print(f"    ⚠️  Strategy 1 failed: {e}")
        
        # Strategy 2: Waypoint-based rerouting (go around disaster zones)
        try:
            print(f"    📍 Strategy 2: Waypoint-based rerouting")
            rerouted = self._compute_waypoint_reroute(start, end, affected_zones)
            if rerouted:
                return self._validate_and_return_reroute(rerouted, "Strategy 2 (waypoints)")
        except Exception as e:
            print(f"    ⚠️  Strategy 2 failed: {e}")
        
        # Strategy 3: Simple offset route (manual rerouting)
        try:
            print(f"    📍 Strategy 3: Manual offset routing")
            rerouted = self._create_offset_route(start, end, affected_zones)
            if rerouted:
                return self._validate_and_return_reroute(rerouted, "Strategy 3 (offset)")
        except Exception as e:
            print(f"    ⚠️  Strategy 3 failed: {e}")
        
        # Strategy 4: Try ORS avoid_polygons for shorter routes only
        total_km = haversine(start[1], start[0], end[1], end[0])
        if total_km <= 300:  # Only try avoidance for shorter routes
            try:
                print(f"    📍 Strategy 4: Avoidance polygons (short route)")
                avoid_geojson = build_avoid_multipolygon(affected_zones, default_radius_km=1.0)
                if avoid_geojson:
                    rerouted = compute_reroute_with_avoid(start, end, avoid_geojson)
                    if rerouted:
                        return self._validate_and_return_reroute(rerouted, "Strategy 4 (avoidance)")
            except Exception as e:
                print(f"    ⚠️  Strategy 4 failed: {e}")
        
        return None, False, "All rerouting strategies failed"
    
    def _validate_and_return_reroute(self, rerouted, strategy_name):
        """Validate a rerouted path and return results"""
        reroute_ok, reroute_msg, reroute_score, reroute_details = validate_route_realtime(
            rerouted, self.disaster_zones, buffer_km=5.0
        )
        
        success_msg = f"{strategy_name}: {rerouted['summary']['distance']:.1f}km, risk={reroute_score:.1f}"
        return rerouted, True, success_msg
    
    def _compute_waypoint_reroute(self, start, end, affected_zones):
        """Compute reroute using waypoints to go around disaster zones"""
        if not affected_zones:
            return None
        
        # Find the centroid of affected zones
        avg_lat = sum(zone.get("location", [0, 0])[1] for zone in affected_zones) / len(affected_zones)
        avg_lon = sum(zone.get("location", [0, 0])[0] for zone in affected_zones) / len(affected_zones)
        
        # Create waypoints that go around the disaster area
        # Calculate perpendicular offset from the direct line
        direct_bearing = math.atan2(end[1] - start[1], end[0] - start[0])
        
        # Try different offset distances
        offset_distances = [0.3, 0.5, 0.8]  # degrees (~30km, 50km, 80km)
        
        for offset_distance in offset_distances:
            # Create two waypoints: one on each side of the disaster zone
            waypoint1 = [
                avg_lon + offset_distance * math.cos(direct_bearing + math.pi/2),
                avg_lat + offset_distance * math.sin(direct_bearing + math.pi/2)
            ]
            waypoint2 = [
                avg_lon + offset_distance * math.cos(direct_bearing - math.pi/2),
                avg_lat + offset_distance * math.sin(direct_bearing - math.pi/2)
            ]
            
            # Try both waypoint routes
            for i, waypoint in enumerate([waypoint1, waypoint2]):
                try:
                    print(f"      Trying waypoint {i+1} (offset {offset_distance:.1f}°): [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
                    
                    # Route: start -> waypoint -> end
                    leg1_routes = fetch_all_ors_routes(start, waypoint, max_alternatives=1)
                    if not leg1_routes:
                        print(f"        Leg 1 failed")
                        continue
                        
                    leg2_routes = fetch_all_ors_routes(waypoint, end, max_alternatives=1)
                    if not leg2_routes:
                        print(f"        Leg 2 failed")
                        continue
                    
                    leg1 = leg1_routes[0]
                    leg2 = leg2_routes[0]
                    
                    # Combine the two legs
                    combined_coords = leg1["geometry"]["coordinates"] + leg2["geometry"]["coordinates"][1:]
                    combined_distance = leg1["summary"]["distance"] + leg2["summary"]["distance"]
                    combined_duration = leg1["summary"]["duration"] + leg2["summary"]["duration"]
                    
                    print(f"        Success! Combined route: {combined_distance:.1f}km")
                    
                    return {
                        "geometry": {"coordinates": combined_coords},
                        "summary": {"distance": combined_distance, "duration": combined_duration},
                        "alternative_id": 1
                    }
                except Exception as e:
                    print(f"        Waypoint {i+1} failed: {e}")
                    continue
        
        return None
    
    def _find_safest_alternative_route(self, start, end):
        """Find the safest alternative route by testing multiple ORS alternatives"""
        try:
            # For long routes, ORS typically returns only one route, so try waypoint approach
            total_km = haversine(start[1], start[0], end[1], end[0])
            
            if total_km > 500:
                print(f"      Long route ({total_km:.1f}km) - using waypoint alternatives")
                return self._generate_alternative_via_waypoints(start, end)
            
            # Get multiple route alternatives for shorter routes
            routes = fetch_all_ors_routes(start, end, max_alternatives=5)
            
            if not routes or len(routes) == 1:
                print(f"      Only one route available - trying waypoint alternatives")
                return self._generate_alternative_via_waypoints(start, end)
            
            # Score each route by safety (lower risk = better)
            scored_routes = []
            for route in routes:
                _, _, risk_score, _ = validate_route_realtime(route, self.disaster_zones, buffer_km=5.0)
                scored_routes.append((risk_score, route))
            
            # Sort by risk score (ascending - lower is better)
            scored_routes.sort(key=lambda x: x[0])
            
            # Return the safest route that's different from the first
            for risk_score, route in scored_routes:
                print(f"      Alternative route with risk score: {risk_score:.1f}")
                return route
            
            return None
            
        except Exception as e:
            print(f"      Alternative route selection failed: {e}")
            return None
    
    def _generate_alternative_via_waypoints(self, start, end):
        """Generate alternative routes using different waypoint strategies"""
        try:
            # Strategy: Route via major cities to create natural alternatives
            mid_lat = (start[1] + end[1]) / 2
            mid_lon = (start[0] + end[0]) / 2
            
            # Try routing via geographically logical hubs
            # Select waypoints based on route geography
            route_bearing = math.atan2(end[1] - start[1], end[0] - start[0])
            
            # Determine route type and select appropriate waypoints
            start_lat, start_lon = start[1], start[0]
            end_lat, end_lon = end[1], end[0]
            
            # Route-specific waypoint selection
            if start_lat < 15 and end_lat > 25:  # South to North/Northeast (like Kerala to Arunachal)
                potential_waypoints = [
                    [77.5946, 12.9716],   # Bengaluru (South India hub)
                    [78.4867, 17.3850],   # Hyderabad (Central India)
                    [80.2707, 13.0827],   # Chennai (East coast)
                    [88.3639, 22.5726],   # Kolkata (East India gateway)
                    [91.7362, 26.1445],   # Guwahati (Northeast hub)
                ]
            elif start_lat > 25 and end_lat > 25:  # North to North (like Himachal to Rajasthan)
                potential_waypoints = [
                    [77.1025, 28.7041],   # Delhi
                    [75.8577, 30.9000],   # Ludhiana (Punjab)
                    [76.7794, 30.7333],   # Chandigarh
                    [74.7973, 31.6340],   # Amritsar
                    [75.7873, 26.9124],   # Jaipur (Rajasthan)
                ]
            else:  # General India-wide routes
                potential_waypoints = [
                    [77.1025, 28.7041],   # Delhi
                    [72.8777, 19.0760],   # Mumbai
                    [77.5946, 12.9716],   # Bengaluru
                    [78.4867, 17.3850],   # Hyderabad
                    [88.3639, 22.5726],   # Kolkata
                    [91.7362, 26.1445],   # Guwahati
                ]
            
            # Find waypoints that are reasonably close AND in the right direction
            best_waypoint = None
            min_distance = float('inf')
            
            for waypoint in potential_waypoints:
                dist = haversine(mid_lat, mid_lon, waypoint[1], waypoint[0])
                
                # Check if waypoint is in reasonable corridor (not too far off the route)
                waypoint_bearing = math.atan2(waypoint[1] - start[1], waypoint[0] - start[0])
                bearing_diff = abs(route_bearing - waypoint_bearing)
                
                # Adjust constraints based on route length
                route_distance = haversine(start[1], start[0], end[1], end[0])
                max_distance = 300 if route_distance > 1500 else 200  # Longer routes allow farther waypoints
                max_bearing_diff = math.pi/2 if route_distance > 1500 else math.pi/3  # 90° vs 60°
                
                # Only consider waypoints within distance and bearing constraints
                if dist < max_distance and bearing_diff < max_bearing_diff:
                    if dist < min_distance:
                        min_distance = dist
                        best_waypoint = waypoint
            
            if best_waypoint:
                print(f"      Trying alternative via major hub: [{best_waypoint[0]:.3f}, {best_waypoint[1]:.3f}]")
                
                leg1_routes = fetch_all_ors_routes(start, best_waypoint, max_alternatives=1)
                leg2_routes = fetch_all_ors_routes(best_waypoint, end, max_alternatives=1)
                
                if leg1_routes and leg2_routes:
                    leg1 = leg1_routes[0]
                    leg2 = leg2_routes[0]
                    
                    combined_coords = leg1["geometry"]["coordinates"] + leg2["geometry"]["coordinates"][1:]
                    combined_distance = leg1["summary"]["distance"] + leg2["summary"]["distance"]
                    combined_duration = leg1["summary"]["duration"] + leg2["summary"]["duration"]
                    
                    return {
                        "geometry": {"coordinates": combined_coords},
                        "summary": {"distance": combined_distance, "duration": combined_duration},
                        "alternative_id": 1
                    }
            
            return None
            
        except Exception as e:
            print(f"      Waypoint alternative generation failed: {e}")
            return None
    
    def _create_offset_route(self, start, end, affected_zones):
        """Create a simple offset route that manually avoids disaster zones"""
        try:
            # Calculate the centroid of disaster zones
            if not affected_zones:
                return None
            
            avg_lat = sum(zone.get("location", [0, 0])[1] for zone in affected_zones) / len(affected_zones)
            avg_lon = sum(zone.get("location", [0, 0])[0] for zone in affected_zones) / len(affected_zones)
            
            # Create a logical detour around the disaster area
            mid_lat = (start[1] + end[1]) / 2
            mid_lon = (start[0] + end[0]) / 2
            
            # Use smaller, more reasonable offset
            offset_distance = 0.3  # degrees (~30km) - much more reasonable
            
            # Calculate direction from disaster to midpoint
            dx = mid_lon - avg_lon
            dy = mid_lat - avg_lat
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0.1:  # Only if disaster is not too close to midpoint
                # Normalize and apply offset
                dx /= length
                dy /= length
                offset_point = [mid_lon + dx * offset_distance, mid_lat + dy * offset_distance]
            else:
                # Fallback: offset perpendicular to start-end line (smaller offset)
                bearing = math.atan2(end[1] - start[1], end[0] - start[0])
                offset_point = [
                    mid_lon + offset_distance * math.cos(bearing + math.pi/2),
                    mid_lat + offset_distance * math.sin(bearing + math.pi/2)
                ]
            
            print(f"      Creating offset route via [{offset_point[0]:.3f}, {offset_point[1]:.3f}]")
            
            # Get routes for both legs
            leg1_routes = fetch_all_ors_routes(start, offset_point, max_alternatives=1)
            leg2_routes = fetch_all_ors_routes(offset_point, end, max_alternatives=1)
            
            if leg1_routes and leg2_routes:
                leg1 = leg1_routes[0]
                leg2 = leg2_routes[0]
                
                # Combine the legs
                combined_coords = leg1["geometry"]["coordinates"] + leg2["geometry"]["coordinates"][1:]
                combined_distance = leg1["summary"]["distance"] + leg2["summary"]["distance"]
                combined_duration = leg1["summary"]["duration"] + leg2["summary"]["duration"]
                
                return {
                    "geometry": {"coordinates": combined_coords},
                    "summary": {"distance": combined_distance, "duration": combined_duration},
                    "alternative_id": 1
                }
            
            return None
            
        except Exception as e:
            print(f"      Offset route creation failed: {e}")
            return None

# ---------- MAIN ENHANCED TEST ----------
def main():
    print("🚀 ENHANCED Smart Route Validation & Rerouting Test")
    print("="*60)
    
    setup_output_directory()
    
    if not test_ors_api_key():
        print("⚠️  Warning: ORS API issues detected - will use fallback routing")
    
    engine = EnhancedSmartReroutingEngine(GEOJSON_PATH)
    
    for route_name, coords in TEST_ROUTES.items():
        start, end = coords
        
        try:
            engine.comprehensive_disaster_test(route_name, start, end)
        except Exception as e:
            print(f"❌ Error testing {route_name}: {e}")
            continue
    
    print(f"\n🏁 Enhanced routing test completed!")
    print(f"📁 Check {OUTPUT_DIR}/ for exported route GeoJSON files")
    print(f"💡 Import these files into your Leaflet map for visualization")

if __name__ == "__main__":
    main()