import pandas as pd
import requests
import joblib
import heapq
import os
from itertools import permutations
import google.generativeai as genai

# ==============================
# API KEYS
# ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

print(f"Gemini API Key loaded: {bool(GEMINI_API_KEY)}")
print(f"Google API Key loaded: {bool(GOOGLE_API_KEY)}")
print(f"Weather API Key loaded: {bool(WEATHER_API_KEY)}")

# ==============================
# LOAD MODEL
# ==============================
ml_model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# ==============================
# GEMINI SETUP
# ==============================
genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel("gemini-2.5-flash")

# Test the model
try:
    response = client.generate_content("Hello")
    print("Gemini API working:", response.text[:50])
except Exception as e:
    print(f"Gemini API Error: {e}")

# ==============================
# MODE ENCODING
# ==============================
def encode_mode(mode):
    return {"road": 1, "rail": 2, "sea": 3}[mode]

# ==============================
# SAFE API REQUEST
# ==============================
def safe_request(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        print(f"URL: {url}")
        return None
    except ValueError as e:
        print(f"JSON Parse Error: {e}")
        return None

# ==============================
# DISTANCE FUNCTION
# ==============================
def get_distance(src, dst):
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={src}&destinations={dst}&key={GOOGLE_API_KEY}"
    print(f"Fetching distance from {src} to {dst}...")
    res = safe_request(url)
    
    if res is None:
        print("Distance Matrix API failed, using default value")
        return 50
    
    try:
        distance_value = res["rows"][0]["elements"][0]["distance"]["value"] / 1000
        print(f"Distance obtained: {distance_value} km")
        return distance_value
    except (KeyError, IndexError, TypeError) as e:
        print(f"Distance parsing error: {e}")
        print(f"Response: {res}")
        return 50

# ==============================
# WEATHER FUNCTION
# ==============================
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    print(f"Fetching weather for {city}...")
    res = safe_request(url)
    
    if res is None:
        print("Weather API failed, using default value")
        return 0
    
    try:
        w = res["weather"][0]["main"]
        weather_code = 3 if w == "Rain" else 1 if w == "Clouds" else 0
        print(f"Weather: {w} (code: {weather_code})")
        return weather_code
    except (KeyError, IndexError, TypeError) as e:
        print(f"Weather parsing error: {e}")
        print(f"Response: {res}")
        return 0

# ==============================
# OPTIMIZE STOPS
# ==============================
def optimize_stops(source, stops, destination):
    """
    Reorders intermediate stops optimally without changing source/destination.
    Uses nearest neighbor heuristic to find optimal order.
    Returns ordered full route list: [source, stop1, stop2, ..., destination]
    """
    if not stops or len(stops) == 0:
        return [source, destination]
    
    if len(stops) == 1:
        return [source, stops[0], destination]
    
    unvisited = set(stops)
    current = source
    ordered_route = [source]
    
    print(f"Optimizing stop order: {stops}")
    
    while unvisited:
        nearest = None
        min_distance = float('inf')
        
        for stop in unvisited:
            distance = get_distance(current, stop)
            print(f"Distance from {current} to {stop}: {distance} km")
            
            if distance < min_distance:
                min_distance = distance
                nearest = stop
        
        if nearest is not None:
            ordered_route.append(nearest)
            current = nearest
            unvisited.remove(nearest)
    
    ordered_route.append(destination)
    print(f"Optimized route: {ordered_route}")
    
    return ordered_route

# ==============================
# PREDICT DELIVERY TIME
# ==============================
def predict_delivery_time(src, dst, mode, params, goal):
    distance = get_distance(src, dst)
    
    mode_multipliers = {
        "road": {"traffic": 0.15, "speed_factor": 1.0},
        "rail": {"traffic": 0.05, "speed_factor": 1.2},
        "sea": {"traffic": 0.02, "speed_factor": 0.8}
    }
    
    multiplier = mode_multipliers.get(mode, {"traffic": 0.1, "speed_factor": 1.0})
    
    data = {
        "distance": distance,
        "traffic_delay": distance * multiplier["traffic"],
        "traffic_level": multiplier["traffic"],
        "weight": params["weight"],
        "price": params["price"],
        "freight": params["freight"],
        "payment_value": params["payment"],
        "departure_hour": params["hour"],
        "priority": params["priority"],
        "mode": encode_mode(mode)
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    time = ml_model.predict(df)[0]
    time *= multiplier["speed_factor"]

    if goal == "time":
        time *= 0.8
    elif goal == "balanced":
        time *= 1

    return time

# ==============================
# GRAPH BUILDER
# ==============================
def build_graph(locations, mode, params, goal):
    graph = {}
    for i in locations:
        graph[i] = {}
        for j in locations:
            if i != j:
                graph[i][j] = predict_delivery_time(i, j, mode, params, goal)
    return graph

# ==============================
# DIJKSTRA
# ==============================
def dijkstra(graph, start):
    q = [(0, start)]
    dist = {n: float('inf') for n in graph}
    dist[start] = 0

    while q:
        d, node = heapq.heappop(q)
        for nei, w in graph[node].items():
            nd = d + w
            if nd < dist[nei]:
                dist[nei] = nd
                heapq.heappush(q, (nd, nei))

    return dist

# ==============================
# TSP 
# ==============================
def tsp(graph, start):
    nodes = list(graph.keys())
    nodes.remove(start)
    
    if len(nodes) > 10:
        return tsp_nearest_neighbor(graph, start)
    
    best_path = None
    best_cost = float('inf')
    total_perms = 1
    for i in range(1, len(nodes) + 1):
        total_perms *= i
    
    print(f"Calculating {total_perms} routes...")
    
    current = 0
    for perm in permutations(nodes):
        cost = 0
        cur = start

        for n in perm:
            cost += graph[cur][n]
            cur = n

        cost += graph[cur][start]

        if cost < best_cost:
            best_cost = cost
            best_path = (start,) + perm + (start,)
        
        current += 1
        if current % max(1, total_perms // 10) == 0:
            print(f"  Progress: {(current/total_perms)*100:.0f} percent")

    return best_path, best_cost

def tsp_nearest_neighbor(graph, start):
    unvisited = set(graph.keys())
    unvisited.remove(start)
    current = start
    path = [start]
    total_cost = 0

    while unvisited:
        nearest = min(unvisited, key=lambda x: graph[current][x])
        total_cost += graph[current][nearest]
        current = nearest
        path.append(current)
        unvisited.remove(current)

    total_cost += graph[current][start]
    path.append(start)
    
    return tuple(path), total_cost

# ==============================
# GOOGLE MAP LINK (FIXED)
# ==============================
def generate_map_link(path):
    """
    Generates Google Maps link without creating a closed loop.
    Path format: (source, stop1, stop2, ..., destination)
    Returns: source -> stop1 -> stop2 -> ... -> destination
    """
    base = "https://www.google.com/maps/dir/"
    route_string = "/".join(path)
    return base + route_string

# ==============================
# GEMINI EXPLANATION
# ==============================
def explain_route(time, mode):
    try:
        prompt = f"Delivery time: {time} days using {mode}. Explain why this is optimal."
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "AI explanation unavailable."

# ==============================
# CHATBOT
# ==============================
def chatbot():
    print("\n AI Chatbot (type 'exit' to stop)\n")

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            print("Bot: Goodbye")
            break

        try:
            response = client.generate_content(
                f"""
                You are a logistics AI assistant.
                Answer based on this system:
                - Uses ML for prediction
                - Uses Dijkstra and TSP for optimization
    
                User: {user}
                """
            )
            print("Bot:", response.text)
        except Exception as e:
            print(f"Bot: Error in response - {e}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("Logistics AI System\n")

    src = input("Source: ")
    dst = input("Destination: ")
    stops_input = input("Stops (comma separated, optional): ")

    weight = float(input("Weight: "))
    price = float(input("Price: "))
    freight = float(input("Freight: "))
    payment = float(input("Payment: "))
    hour = int(input("Departure Hour: "))
    priority = int(input("Priority (1-3): "))
    goal = input("Optimization Goal (cost/time/balanced): ")

    params = {
        "weight": weight,
        "price": price,
        "freight": freight,
        "payment": payment,
        "hour": hour,
        "priority": priority
    }

    stops = []
    if stops_input.strip():
        stops = [stop.strip() for stop in stops_input.split(",")]

    optimized_route = optimize_stops(src, stops, dst)

    print("\n Mode Comparison:\n")

    results = []
    for mode in ["road", "rail", "sea"]:
        time = predict_delivery_time(src, dst, mode, params, goal)
        print(f"{mode.upper()} -> {round(time, 2)} days")
        results.append((mode, time))

    best_mode = min(results, key=lambda x: x[1])[0]
    print(f"\n BEST MODE: {best_mode}")

    graph = build_graph(optimized_route, best_mode, params, goal)

    dijkstra_result = dijkstra(graph, src)
    dijkstra_result = {k: round(float(v), 2) for k, v in dijkstra_result.items()}
    print(f"\nDijkstra: {dijkstra_result}")

    path, total_time = tsp(graph, src)
    path_without_loop = path[:-1]

    print("\n FINAL RESULT")
    print(f"Best Path: {path_without_loop}")
    print(f"Total Delivery Time: {round(total_time, 2)} days")
    print(f"Distance: {get_distance(src, dst)} km")

    map_link = generate_map_link(path_without_loop)
    print("\n Google Maps Link:")
    print(map_link)

    print("\n AI Explanation:\n")
    print(explain_route(total_time, best_mode))

    chatbot()