import sys
import time
import requests

# Your ORS key (already present/used in the repo)
ORS_API_KEY = "ca37f84435744a6bb60ed5b4591ed574"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

HEADERS = {
    "Authorization": ORS_API_KEY,
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/geo+json, application/json"
}

# A tiny sanity route: Delhi (Connaught Place) to India Gate
BODY = {
    "coordinates": [[77.2090, 28.6139], [77.2295, 28.6129]],
    "instructions": False
}


def check_once(timeout_sec: int = 30) -> bool:
    try:
        resp = requests.post(ORS_URL, json=BODY, headers=HEADERS, timeout=timeout_sec)
        if resp.status_code != 200:
            print(f"❌ ORS HTTP {resp.status_code}: {resp.text[:160]}")
            return False
        # Validate basic shape
        data = resp.json()
        routes = data.get("routes") or data.get("features") or []
        if not routes:
            print("❌ ORS responded but no routes returned")
            return False
        print("✅ ORS is up and returned a route")
        return True
    except requests.Timeout:
        print("⏳ Timeout waiting for ORS (increase timeout or retry)")
        return False
    except Exception as e:
        print(f"💥 Request failed: {e}")
        return False


def main():
    watch = "--watch" in sys.argv
    interval = 30
    timeout = 30
    for arg in sys.argv:
        if arg.startswith("--interval="):
            try:
                interval = int(arg.split("=", 1)[1])
            except Exception:
                pass
        if arg.startswith("--timeout="):
            try:
                timeout = int(arg.split("=", 1)[1])
            except Exception:
                pass

    if not watch:
        ok = check_once(timeout_sec=timeout)
        sys.exit(0 if ok else 1)

    # Poll until available
    print(f"Watching ORS availability every {interval}s (timeout {timeout}s)... Press Ctrl+C to stop.")
    while True:
        ok = check_once(timeout_sec=timeout)
        if ok:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()


