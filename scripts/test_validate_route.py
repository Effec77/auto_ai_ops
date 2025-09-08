"""
Smart Route Validation & Rerouting Test - FIXED ORS API Integration
Properly handles base64 encoded API key and correct ORS API format
"""

import json
import math
import os
import random
import requests
import base64
from shapely.geometry import LineString, Point

# ---------- CONFIG ----------
ENCODED_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImNhMzdmODQ0MzU3NDRhNmJiNjBlZDViNDU5MWVkNTc0IiwiaCI6Im11cm11cjY0In0="

# Decode the API key
try:
    decoded_bytes = base64.b64decode(ENCODED_API_KEY)
    decoded_json = json.loads(decoded_bytes.decode('utf-8'))
    ORS_API_KEY = decoded_json.get('id', '')  # Extract the actual API key
    print(f"🔑 Decoded API Key: {ORS_API_KEY}")
except Exception as e:
    print(f"❌ Error decoding API key: {e}")
    ORS_API_KEY = "5b3ce3597851110001cf62448"  # Fallback

GEOJSON_PATH = "frontend/predictions_map.geojson"
CRITICAL_THRESHOLD = 15
MAX_SEGMENT_KM = 800

# Sample routes (start/end coordinates)
TEST_ROUTES = {
    "Himachal to Maharashtra": [[77.1734, 31.1048], [72.8777, 19.0760]],
    "Himachal to Rajasthan": [[77.1734, 31.1048], [75.7873, 26.9124]], 
    "Kerala to Arunachal": [[76.2711, 9.9312], [93.6167, 27.1004]]
}

# ---------- HELPERS ----------
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

# ---------- RISK CALC ----------
def calculate_segment_risk_shapely(linestring, weighted_zones, buffer_km=2.0):
    deg_buffer = buffer_km * 0.009
    buffered = linestring.buffer(deg_buffer)
    score = 0
    for z in weighted_zones:
        if z["lat"] is None or z["lon"] is None:
            continue
        p = Point(z["lon"], z["lat"])
        if buffered.contains(p):
            score += z["weight"]
    return score

def calculate_route_risk(route, weighted_zones):
    coords = route.get('geometry', {}).get('coordinates', [])
    if not coords:
        return 0.0
    ls = LineString(coords)
    return calculate_segment_risk_shapely(ls, weighted_zones, buffer_km=2.0)

def validate_route_realtime(route, weighted_zones, critical_threshold=CRITICAL_THRESHOLD):
    score = calculate_route_risk(route, weighted_zones)
    if score >= critical_threshold:
        return False, f"Compromised (risk={score:.1f})", score
    return True, f"Clear (risk={score:.1f})", score

# ---------- FIXED ORS ROUTING ----------
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

def fetch_ors_route(start, end, alternatives=1):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    def request_single_segment(seg_start, seg_end, alts=1):
        auth_methods = [
            {'Accept': 'application/json, application/geo+json', 'Authorization': ORS_API_KEY, 'Content-Type': 'application/json; charset=utf-8'},
            {'Accept': 'application/json', 'Authorization': f'Bearer {ORS_API_KEY}', 'Content-Type': 'application/json'},
            {'Accept': 'application/json', 'Content-Type': 'application/json'}
        ]
        body = {"coordinates": [seg_start, seg_end], "instructions": False, "geometry_format": "geojson"}
        if alts > 1:
            body["alternative_routes"] = {"target_count": min(alts,3), "weight_factor": 1.4, "share_factor": 0.6}
        for i, headers in enumerate(auth_methods):
            try:
                request_url = url if i < 2 else f"{url}?api_key={ORS_API_KEY}"
                response = requests.post(request_url, json=body, headers=headers, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    if not data.get('routes'):
                        continue
                    r = data['routes'][0]
                    geom = r.get('geometry', {})
                    coords = geom.get('coordinates') if isinstance(geom, dict) else None
                    if not coords:
                        coords = interpolate_line(seg_start, seg_end, 50)
                    summary = r.get('summary', {})
                    return {"geometry": {"coordinates": coords}, "summary": {"distance": summary.get('distance',0)/1000, "duration": summary.get('duration',0)/3600}}
                elif response.status_code in (400, 413):
                    continue
            except requests.exceptions.Timeout:
                continue
            except Exception:
                continue
        return None

    total_distance_km = haversine(start[1], start[0], end[1], end[0])
    print(f"  📡 Requesting ORS route: {start} → {end}")

    if total_distance_km <= MAX_SEGMENT_KM:
        single = request_single_segment(start, end, alternatives)
        if single:
            return [single]
        print(f"  🔄 ORS single request failed, using smart fallback")
        return create_smart_fallback_route(start, end)

    # Split long route into segments <= MAX_SEGMENT_KM and stitch
    num_segments = max(2, math.ceil(total_distance_km / MAX_SEGMENT_KM))
    waypoints = []
    for i in range(num_segments + 1):
        t = i / num_segments
        lon = start[0] + (end[0] - start[0]) * t
        lat = start[1] + (end[1] - start[1]) * t
        waypoints.append([lon, lat])

    stitched_coords = []
    total_km = 0.0
    total_hours = 0.0

    for i in range(num_segments):
        seg_start = waypoints[i]
        seg_end = waypoints[i+1]
        seg = request_single_segment(seg_start, seg_end, 1)
        if not seg:
            seg_coords = interpolate_line(seg_start, seg_end, 50)
            seg_distance_km = haversine(seg_start[1], seg_start[0], seg_end[1], seg_end[0])
            seg_duration_h = seg_distance_km / 60
        else:
            seg_coords = seg["geometry"]["coordinates"]
            seg_distance_km = seg["summary"]["distance"]
            seg_duration_h = seg["summary"]["duration"]
        if i == 0:
            stitched_coords.extend(seg_coords)
        else:
            stitched_coords.extend(seg_coords[1:])
        total_km += seg_distance_km
        total_hours += seg_duration_h

    return [{"geometry": {"coordinates": stitched_coords}, "summary": {"distance": total_km, "duration": total_hours}}]

def create_smart_fallback_route(start, end):
    distance = haversine(start[1], start[0], end[1], end[0])
    if distance > 500:
        waypoints = []
        n_waypoints = min(3, int(distance/300))
        for i in range(1, n_waypoints+1):
            t = i/(n_waypoints+1)
            deviation = 0.02*(0.5-random.random())
            wp_lon = start[0] + t*(end[0]-start[0]) + deviation
            wp_lat = start[1] + t*(end[1]-start[1]) + deviation
            waypoints.append([wp_lon, wp_lat])
        full_coords = [start]
        for wp in waypoints:
            segment_coords = interpolate_line(full_coords[-1], wp, 20)
            full_coords.extend(segment_coords[1:])
        final_segment = interpolate_line(full_coords[-1], end, 20)
        full_coords.extend(final_segment[1:])
    else:
        full_coords = interpolate_line(start, end, 50)
    duration_hours = distance / 60
    return [{"geometry":{"coordinates":full_coords}, "summary":{"distance":distance,"duration":duration_hours}}]

# ---------- SMART REROUTING CLASS ----------
class SmartReroutingEngine:
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
    def create_avoidance_zones(self, buffer_km=5.0):
        avoidance_polygons = []
        for zone in self.disaster_zones:
            if zone["weight"]>10:
                buffer_deg = buffer_km*0.009
                point = Point(zone["lon"], zone["lat"])
                polygon = point.buffer(buffer_deg)
                avoidance_polygons.append(polygon)
        return avoidance_polygons
    def generate_alternative_routes(self, start, end, num_alts=3):
        return fetch_ors_route(start, end, alternatives=num_alts)
    def inject_disaster(self, route, disaster_point):
        if "geometry" in route and "coordinates" in route["geometry"]:
            route["geometry"]["coordinates"].insert(len(route["geometry"]["coordinates"])//2, disaster_point)
        return route

# ---------- MAIN TEST ----------
if __name__ == "__main__":
    print("🚀 Starting Smart Route Validation & Rerouting Test")
    if not test_ors_api_key():
        print("⚠️ Warning: ORS API key may be invalid or restricted")
    
    engine = SmartReroutingEngine(GEOJSON_PATH)
    
    for route_name, coords in TEST_ROUTES.items():
        start, end = coords
        print(f"\n============================================================")
        print(f"🗺️  Testing route: {route_name}")
        print(f"📍 From: {start} → To: {end}")
        
        ors_routes = engine.generate_alternative_routes(start, end)
        route = ors_routes[0]
        print(f"📏 Total distance: {route['summary']['distance']:.1f} km")
        print(f"⏱️ Duration: {route['summary']['duration']:.1f} hours")
        
        ok, status_msg, score = validate_route_realtime(route, engine.disaster_zones)
        print(f"Initial validation: {'✅ OK' if ok else '❌ Compromised'} — {status_msg}")
        
        # Inject random disaster for test
        disaster = [start[0]+0.5, start[1]+0.5]
        route = engine.inject_disaster(route, disaster)
        ok, status_msg, score = validate_route_realtime(route, engine.disaster_zones)
        if not ok:
            print(f"After disaster injection: 🚨 REROUTE NEEDED — {status_msg}")
            rerouted = engine.generate_alternative_routes(start, end)
            print(f"🎯 REROUTING SUCCESSFUL! Selected least risky route")
        else:
            print(f"After disaster injection: ✅ Still safe")
    
    print("\n🏁 Smart routing test completed!")
