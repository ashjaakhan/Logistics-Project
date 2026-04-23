import streamlit as st
import pandas as pd
import requests
import joblib
import heapq
import os
from itertools import permutations
import google.generativeai as genai
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Logistics AI System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        [data-testid="stMainBlockContainer"] {
            background: transparent;
            padding: 0;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            background: transparent;
        }
        
        .nav-header {
            color: white;
            font-size: 24px;
            font-weight: 900;
            padding: 20px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
        }
        
        .nav-item {
            color: white;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .nav-item:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(4px);
        }
        
        .nav-item.active {
            background: rgba(255, 255, 255, 0.3);
            border-left: 4px solid white;
            padding-left: 12px;
        }
        
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 40px;
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .header-title {
            font-size: 42px;
            font-weight: 900;
            margin: 0;
            letter-spacing: -1px;
        }
        
        .header-subtitle {
            font-size: 16px;
            color: rgba(255, 255, 255, 0.85);
            margin-top: 8px;
            font-weight: 300;
        }
        
        .content-wrapper {
            padding: 0 40px 40px 40px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
        }
        
        .left-panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            overflow: hidden;
        }
        
        .right-panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            overflow: hidden;
        }
        
        .form-section {
            padding: 24px;
            border-bottom: 2px solid #f0f2f5;
        }
        
        .form-section:last-child {
            border-bottom: none;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 16px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #667eea;
        }
        
        .form-input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ecf0f1;
            border-radius: 8px;
            font-size: 14px;
            margin-bottom: 12px;
            transition: all 0.3s ease;
        }
        
        .form-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }
        
        .form-label {
            font-size: 13px;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 6px;
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .slider-container {
            margin-bottom: 16px;
        }
        
        .radio-group {
            display: flex;
            gap: 12px;
            margin-top: 12px;
        }
        
        .radio-option {
            flex: 1;
            padding: 12px;
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 12px;
        }
        
        .radio-option.selected {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
        }
        
        .optimize-btn {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            border: none;
            padding: 16px 40px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            margin-top: 16px;
        }
        
        .optimize-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.6);
        }
        
        .results-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 24px;
            font-size: 16px;
            font-weight: 700;
        }
        
        .results-content {
            padding: 24px;
        }
        
        .mode-cards {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .mode-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ecf0f1 100%);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 2px solid #ecf0f1;
            transition: all 0.3s ease;
        }
        
        .mode-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        
        .mode-card.best {
            background: linear-gradient(135deg, #d5f4e6 0%, #abebc6 100%);
            border-color: #27ae60;
        }
        
        .mode-name {
            font-size: 14px;
            font-weight: 700;
            color: #2c3e50;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .mode-time {
            font-size: 28px;
            font-weight: 900;
            color: #667eea;
            margin-bottom: 4px;
        }
        
        .mode-card.best .mode-time {
            color: #27ae60;
        }
        
        .mode-unit {
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .best-badge {
            display: inline-block;
            background: #27ae60;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            margin-top: 8px;
            text-transform: uppercase;
        }
        
        .recommendation-box {
            background: #fff9e6;
            border-left: 4px solid #f39c12;
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        
        .recommendation-text {
            color: #7d6608;
            font-weight: 600;
            font-size: 14px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .metric-item {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 900;
            color: #667eea;
            margin-bottom: 4px;
        }
        
        .metric-label {
            font-size: 11px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .route-box {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 16px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #2c3e50;
            word-break: break-all;
        }
        
        .maps-link {
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 700;
            font-size: 13px;
            transition: all 0.3s ease;
        }
        
        .maps-link:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }
        
        .empty-state {
            padding: 60px 24px;
            text-align: center;
            color: #7f8c8d;
        }
        
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .empty-state-title {
            font-size: 18px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        
        .empty-state-text {
            font-size: 14px;
            line-height: 1.6;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }
        
        .info-table th {
            background: #f0f2f5;
            padding: 12px;
            text-align: left;
            font-weight: 700;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
        }
        
        .info-table td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .info-table tr:hover {
            background: #f8f9fa;
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            margin-bottom: 20px;
        }
        
        .stat-value {
            font-size: 32px;
            font-weight: 900;
            color: #667eea;
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            margin-bottom: 20px;
        }
        
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

try:
    ml_model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
except FileNotFoundError:
    st.error("Error: model.pkl or features.pkl not found")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    client = None

def encode_mode(mode):
    return {"road": 1, "rail": 2, "sea": 3}[mode]

def safe_request(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def get_distance(src, dst):
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={src}&destinations={dst}&key={GOOGLE_API_KEY}"
    res = safe_request(url)
    
    if res is None:
        return 50
    
    try:
        return res["rows"][0]["elements"][0]["distance"]["value"] / 1000
    except:
        return 50

def optimize_stops(source, stops, destination):
    if not stops or len(stops) == 0:
        return [source, destination]
    
    if len(stops) == 1:
        return [source, stops[0], destination]
    
    unvisited = set(stops)
    current = source
    ordered_route = [source]
    
    while unvisited:
        nearest = None
        min_distance = float('inf')
        
        for stop in unvisited:
            distance = get_distance(current, stop)
            if distance < min_distance:
                min_distance = distance
                nearest = stop
        
        if nearest is not None:
            ordered_route.append(nearest)
            current = nearest
            unvisited.remove(nearest)
    
    ordered_route.append(destination)
    return ordered_route

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

def build_graph(locations, mode, params, goal):
    graph = {}
    for i in locations:
        graph[i] = {}
        for j in locations:
            if i != j:
                graph[i][j] = predict_delivery_time(i, j, mode, params, goal)
    return graph

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

def tsp(graph, start):
    nodes = list(graph.keys())
    nodes.remove(start)
    
    if len(nodes) > 10:
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
    
    best_path = None
    best_cost = float('inf')
    
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

    return best_path, best_cost

def generate_map_link(path):
    base = "https://www.google.com/maps/dir/"
    return base + "/".join(path)

def explain_route(time, mode):
    if client is None:
        return "AI explanation unavailable (Gemini API not configured)"
    
    try:
        prompt = f"Delivery time: {time} days using {mode}. Explain why this is optimal."
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI explanation unavailable: {e}"

with st.sidebar:
    st.markdown('<div class="nav-header">📦 LOGISTICS AI</div>', unsafe_allow_html=True)
    
    nav_pages = {
        "Route Optimizer": "home",
        "Analytics": "analytics",
        "AI Assistant": "assistant",
        "Settings": "settings"
    }
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    for page_name, page_key in nav_pages.items():
        is_active = st.session_state.current_page == page_key
        css_class = "nav-item active" if is_active else "nav-item"
        
        if st.button(page_name, key=page_key, use_container_width=True):
            st.session_state.current_page = page_key

st.markdown('<div class="header-container"><h1 class="header-title">📦 Logistics AI</h1><p class="header-subtitle">Intelligent Route Optimization & Delivery Management System</p></div>', unsafe_allow_html=True)

st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

if st.session_state.current_page == "home":
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section"><div class="section-header">📍 Location Details</div>', unsafe_allow_html=True)
        src = st.text_input("Source Location", placeholder="e.g., Lucknow", key="src_input")
        dst = st.text_input("Destination Location", placeholder="e.g., Delhi", key="dst_input")
        stops_input = st.text_area("Additional Stops", placeholder="e.g., Agra, Mathura", height=60, key="stops_input")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section"><div class="section-header">📦 Cargo Details</div>', unsafe_allow_html=True)
        
        col_w, col_p = st.columns(2)
        with col_w:
            st.markdown('<label class="form-label">Weight (kg)</label>', unsafe_allow_html=True)
            weight = st.number_input("Weight", min_value=0.0, value=100.0, step=10.0, label_visibility="collapsed")
        with col_p:
            st.markdown('<label class="form-label">Price (INR)</label>', unsafe_allow_html=True)
            price = st.number_input("Price", min_value=0.0, value=1000.0, step=100.0, label_visibility="collapsed")
        
        col_f, col_pm = st.columns(2)
        with col_f:
            st.markdown('<label class="form-label">Freight Rate</label>', unsafe_allow_html=True)
            freight = st.number_input("Freight Rate", min_value=0.0, value=5.0, step=0.5, label_visibility="collapsed")
        with col_pm:
            st.markdown('<label class="form-label">Payment (INR)</label>', unsafe_allow_html=True)
            payment = st.number_input("Payment", min_value=0.0, value=500.0, step=100.0, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-section"><div class="section-header">⚙️ Route Settings</div>', unsafe_allow_html=True)
        
        st.markdown('<label class="form-label">Departure Hour</label>', unsafe_allow_html=True)
        hour = st.slider("Departure Hour", min_value=0, max_value=23, value=8, label_visibility="collapsed")
        
        st.markdown('<label class="form-label">Priority Level</label>', unsafe_allow_html=True)
        priority = st.slider("Priority Level", min_value=1, max_value=3, value=2, label_visibility="collapsed")
        
        st.markdown('<label class="form-label">Optimization Goal</label>', unsafe_allow_html=True)
        goal_option = st.radio(
            "Optimization Goal",
            options=["time", "cost", "balanced"],
            format_func=lambda x: {"time": "Minimize Time", "cost": "Minimize Cost", "balanced": "Balanced"}[x],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Optimize Route", use_container_width=True, key="optimize_btn"):
            st.session_state.optimize_clicked = True
        else:
            if "optimize_clicked" not in st.session_state:
                st.session_state.optimize_clicked = False
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        
        if st.session_state.get("optimize_clicked", False):
            if not src or not dst:
                st.markdown('<div class="results-header">Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="results-content"><p style="color: #e74c3c; font-weight: bold;">Please enter Source and Destination locations</p></div>', unsafe_allow_html=True)
            else:
                with st.spinner("Processing route optimization..."):
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
                    
                    road_time = predict_delivery_time(src, dst, "road", params, goal_option)
                    rail_time = predict_delivery_time(src, dst, "rail", params, goal_option)
                    sea_time = predict_delivery_time(src, dst, "sea", params, goal_option)
                    
                    results = [("road", road_time), ("rail", rail_time), ("sea", sea_time)]
                    best_mode = min(results, key=lambda x: x[1])[0]
                    
                    st.markdown('<div class="results-header">Mode Comparison</div>', unsafe_allow_html=True)
                    st.markdown('<div class="results-content">', unsafe_allow_html=True)
                    
                    st.markdown('<div class="mode-cards">', unsafe_allow_html=True)
                    
                    col_road, col_rail, col_sea = st.columns(3)
                    
                    with col_road:
                        css_class = "mode-card best" if best_mode == "road" else "mode-card"
                        st.markdown(f'''
                            <div class="{css_class}">
                                <div class="mode-name">Road</div>
                                <div class="mode-time">{round(road_time, 2)}</div>
                                <div class="mode-unit">Days</div>
                                {"<div class='best-badge'>Recommended</div>" if best_mode == "road" else ""}
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_rail:
                        css_class = "mode-card best" if best_mode == "rail" else "mode-card"
                        st.markdown(f'''
                            <div class="{css_class}">
                                <div class="mode-name">Rail</div>
                                <div class="mode-time">{round(rail_time, 2)}</div>
                                <div class="mode-unit">Days</div>
                                {"<div class='best-badge'>Recommended</div>" if best_mode == "rail" else ""}
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_sea:
                        css_class = "mode-card best" if best_mode == "sea" else "mode-card"
                        st.markdown(f'''
                            <div class="{css_class}">
                                <div class="mode-name">Sea</div>
                                <div class="mode-time">{round(sea_time, 2)}</div>
                                <div class="mode-unit">Days</div>
                                {"<div class='best-badge'>Recommended</div>" if best_mode == "sea" else ""}
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="recommendation-box"><div class="recommendation-text">Recommended Mode: {best_mode.upper()} ({round(eval(f"{best_mode}_time"), 2)} days)</div></div>', unsafe_allow_html=True)
                    
                    graph = build_graph(optimized_route, best_mode, params, goal_option)
                    dijkstra_result = dijkstra(graph, src)
                    path, total_time = tsp(graph, src)
                    path_without_loop = path[:-1]
                    
                    st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f'''
                            <div class="metric-item">
                                <div class="metric-value">{round(total_time, 2)}</div>
                                <div class="metric-label">Total Days</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown(f'''
                            <div class="metric-item">
                                <div class="metric-value">{round(get_distance(src, dst), 0)}</div>
                                <div class="metric-label">Total KM</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown(f'''
                            <div class="metric-item">
                                <div class="metric-value">{len(path_without_loop) - 1}</div>
                                <div class="metric-label">Stops</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<h4 style="color: #2c3e50; margin: 20px 0 12px 0;">Route Details</h4>', unsafe_allow_html=True)
                    st.markdown(f'<div class="route-box">{" → ".join(path_without_loop)}</div>', unsafe_allow_html=True)
                    
                    map_link = generate_map_link(path_without_loop)
                    st.markdown(f'<a href="{map_link}" target="_blank" class="maps-link">Open in Google Maps</a>', unsafe_allow_html=True)
                    
                    st.markdown('<h4 style="color: #2c3e50; margin: 20px 0 12px 0;">AI Explanation</h4>', unsafe_allow_html=True)
                    explanation = explain_route(total_time, best_mode)
                    st.markdown(f'<div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-left: 4px solid #667eea; padding: 16px; border-radius: 8px; color: #2c3e50; line-height: 1.6; font-size: 14px;">{explanation}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="results-header">Results</div>', unsafe_allow_html=True)
            st.markdown('''
                <div class="results-content">
                    <div class="empty-state">
                        <div class="empty-state-icon">📊</div>
                        <div class="empty-state-title">Ready to Optimize?</div>
                        <div class="empty-state-text">
                            Fill in the location and cargo details on the left,<br>
                            then click the "Optimize Route" button to see<br>
                            AI-powered route recommendations.
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "analytics":
    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('''
            <div class="stat-card">
                <div class="stat-value">127</div>
                <div class="stat-label">Total Routes</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
            <div class="stat-card">
                <div class="stat-value">12.5</div>
                <div class="stat-label">Avg Days</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
            <div class="stat-card">
                <div class="stat-value">4,850</div>
                <div class="stat-label">Total KM</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown('''
            <div class="stat-card">
                <div class="stat-value">94%</div>
                <div class="stat-label">On-Time Rate</div>
            </div>
        ''', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #2c3e50; margin-bottom: 20px;">Routes by Mode</h3>', unsafe_allow_html=True)
        
        mode_data = {
            'Mode': ['Road', 'Rail', 'Sea'],
            'Count': [65, 42, 20]
        }
        fig_mode = px.pie(
            pd.DataFrame(mode_data),
            values='Count',
            names='Mode',
            color_discrete_sequence=['#667eea', '#764ba2', '#f39c12']
        )
        fig_mode.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_mode, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #2c3e50; margin-bottom: 20px;">Delivery Performance</h3>', unsafe_allow_html=True)
        
        perf_data = {
            'Status': ['On Time', 'Delayed'],
            'Count': [119, 8]
        }
        fig_perf = px.bar(
            pd.DataFrame(perf_data),
            x='Status',
            y='Count',
            color='Status',
            color_discrete_sequence=['#27ae60', '#e74c3c']
        )
        fig_perf.update_layout(
            showlegend=False,
            height=400,
            xaxis_title='',
            yaxis_title='Number of Deliveries',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #2c3e50; margin-bottom: 20px;">Average Delivery Time Trend</h3>', unsafe_allow_html=True)
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        trend_data = {
            'Date': dates,
            'Days': np.random.uniform(10, 15, 30)
        }
        fig_trend = px.line(
            pd.DataFrame(trend_data),
            x='Date',
            y='Days',
            markers=True
        )
        fig_trend.update_traces(line=dict(color='#667eea', width=3), marker=dict(size=6))
        fig_trend.update_layout(
            height=400,
            xaxis_title='Date',
            yaxis_title='Delivery Days',
            hovermode='x unified',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #2c3e50; margin-bottom: 20px;">Top Routes</h3>', unsafe_allow_html=True)
        
        routes_data = {
            'Route': ['Delhi to Mumbai', 'Bangalore to Chennai', 'Lucknow to Delhi', 'Mumbai to Pune', 'Hyderabad to Bangalore'],
            'Count': [24, 18, 15, 12, 10]
        }
        fig_routes = px.bar(
            pd.DataFrame(routes_data),
            y='Route',
            x='Count',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_routes.update_layout(
            height=400,
            xaxis_title='Number of Deliveries',
            yaxis_title='',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_routes, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #2c3e50; margin-bottom: 20px;">Detailed Route Analytics</h3>', unsafe_allow_html=True)
    
    analytics_data = {
        'Route': ['Delhi to Mumbai', 'Bangalore to Chennai', 'Lucknow to Delhi', 'Mumbai to Pune', 'Hyderabad to Bangalore'],
        'Total Routes': [24, 18, 15, 12, 10],
        'Avg Days': [12.4, 11.8, 8.5, 6.3, 9.2],
        'Total KM': [1420, 1210, 680, 160, 580],
        'On-Time %': [96, 94, 98, 100, 90]
    }
    
    df_analytics = pd.DataFrame(analytics_data)
    st.dataframe(df_analytics, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "assistant":
    st.markdown('<div style="background: white; border-radius: 12px; padding: 40px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);"><h2 style="color: #2c3e50; margin-bottom: 20px;">AI Logistics Assistant</h2>', unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")
    
    user_input = st.text_input("Ask me anything about logistics...")
    
    if user_input:
        if client is None:
            st.error("Gemini API not configured")
        else:
            try:
                response = client.generate_content(
                    f"You are a logistics AI assistant. Answer: {user_input}"
                )
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "bot", "content": response.text})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "settings":
    st.markdown('<div style="margin-bottom: 30px;"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown('''
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); overflow: hidden;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px;">
                    <h2 style="margin: 0; font-size: 28px; font-weight: 900;">System Settings</h2>
                    <p style="margin: 8px 0 0 0; color: rgba(255, 255, 255, 0.85); font-size: 14px;">Manage your Logistics AI system configuration</p>
                </div>
                
                <div style="padding: 30px;">
                    <h3 style="color: #2c3e50; margin: 0 0 20px 0; font-size: 18px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">API Configuration</h3>
        ''', unsafe_allow_html=True)
        
        api_configs = {
            'Gemini API': {'status': bool(GEMINI_API_KEY), 'icon': '🤖'},
            'Google Maps API': {'status': bool(GOOGLE_API_KEY), 'icon': '🗺️'},
            'Weather API': {'status': bool(WEATHER_API_KEY), 'icon': '🌤️'}
        }
        
        for api_name, config in api_configs.items():
            status_text = "Configured" if config['status'] else "Not Configured"
            status_color = "#27ae60" if config['status'] else "#e74c3c"
            status_bg = "#d5f4e6" if config['status'] else "#fadbd8"
            
            st.markdown(f'''
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px; margin-bottom: 12px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {status_color};">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">{config['icon']}</span>
                        <div>
                            <div style="font-weight: 700; color: #2c3e50; font-size: 14px;">{api_name}</div>
                            <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">API Key Status</div>
                        </div>
                    </div>
                    <div style="background: {status_bg}; color: {status_color}; padding: 8px 16px; border-radius: 20px; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                        {status_text}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('''
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); padding: 30px; margin-top: 20px;">
                <h3 style="color: #2c3e50; margin: 0 0 20px 0; font-size: 18px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">Model Information</h3>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">Model Status</div>
                    <div style="font-size: 20px; font-weight: 900; color: #27ae60;">Loaded</div>
                </div>
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">Features Count</div>
                    <div style="font-size: 20px; font-weight: 900; color: #667eea;">{len(features)}</div>
                </div>
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">Optimization Methods</div>
                    <div style="font-size: 14px; font-weight: 700; color: #2c3e50;">Dijkstra, TSP</div>
                </div>
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px;">
                    <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 8px;">Transport Modes</div>
                    <div style="font-size: 14px; font-weight: 700; color: #2c3e50;">Road, Rail, Sea</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('''
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); padding: 30px; margin-top: 20px;">
                <h3 style="color: #2c3e50; margin: 0 0 20px 0; font-size: 18px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">Optimization Settings</h3>
                
                <div style="margin-bottom: 24px;">
                    <label style="display: block; font-size: 14px; font-weight: 700; color: #34495e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Maximum Stops for Exact TSP</label>
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <input type="range" min="5" max="15" value="10" style="flex: 1; height: 6px; border-radius: 3px; background: #667eea; outline: none; -webkit-appearance: none;" />
                        <span style="font-weight: 700; color: #667eea; min-width: 40px; text-align: right;">10 stops</span>
                    </div>
                    <p style="font-size: 12px; color: #7f8c8d; margin-top: 8px;">Beyond this limit, nearest neighbor heuristic will be used</p>
                </div>
                
                <div style="margin-bottom: 24px;">
                    <label style="display: block; font-size: 14px; font-weight: 700; color: #34495e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">API Request Timeout</label>
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <input type="range" min="3" max="10" value="5" style="flex: 1; height: 6px; border-radius: 3px; background: #667eea; outline: none; -webkit-appearance: none;" />
                        <span style="font-weight: 700; color: #667eea; min-width: 40px; text-align: right;">5s</span>
                    </div>
                    <p style="font-size: 12px; color: #7f8c8d; margin-top: 8px;">Maximum time to wait for API responses</p>
                </div>
                
                <div style="display: flex; gap: 12px; padding: 16px; background: #f0f2f5; border-radius: 8px; border-left: 4px solid #3498db;">
                    <span style="font-size: 16px;">ℹ️</span>
                    <div>
                        <div style="font-weight: 700; color: #2c3e50; font-size: 13px;">Pro Tip</div>
                        <p style="color: #7f8c8d; font-size: 12px; margin: 4px 0 0 0;">Increase TSP limit for smaller problems to get exact optimal solutions. For large problems (>10 stops), heuristic provides fast approximate solutions.</p>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin: 0 0 16px 0; font-size: 16px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">System Status</h3>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="width: 12px; height: 12px; background: #27ae60; border-radius: 50%;"></span>
                        <span style="font-size: 13px; font-weight: 600; color: #2c3e50;">System Running</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="width: 12px; height: 12px; background: #27ae60; border-radius: 50%;"></span>
                        <span style="font-size: 13px; font-weight: 600; color: #2c3e50;">Model Loaded</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="width: 12px; height: 12px; background: #f39c12; border-radius: 50%;"></span>
                        <span style="font-size: 13px; font-weight: 600; color: #2c3e50;">APIs Partial</span>
                    </div>
                </div>
            </div>
            
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); padding: 24px; margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin: 0 0 16px 0; font-size: 16px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">Quick Actions</h3>
                
                <button style="width: 100%; padding: 12px; background: #667eea; color: white; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; margin-bottom: 8px; transition: all 0.3s ease;" onmouseover="this.style.background='#5568d3'" onmouseout="this.style.background='#667eea'">
                    Test APIs
                </button>
                
                <button style="width: 100%; padding: 12px; background: #f0f2f5; color: #2c3e50; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; margin-bottom: 8px; transition: all 0.3s ease;" onmouseover="this.style.background='#ecf0f1'" onmouseout="this.style.background='#f0f2f5'">
                    Clear Cache
                </button>
                
                <button style="width: 100%; padding: 12px; background: #f0f2f5; color: #2c3e50; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; transition: all 0.3s ease;" onmouseover="this.style.background='#ecf0f1'" onmouseout="this.style.background='#f0f2f5'">
                    Export Logs
                </button>
            </div>
            
            <div style="background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12); padding: 24px;">
                <h3 style="color: #2c3e50; margin: 0 0 16px 0; font-size: 16px; font-weight: 700; padding-bottom: 12px; border-bottom: 2px solid #f0f2f5;">Version Info</h3>
                
                <div style="font-size: 12px; color: #7f8c8d; line-height: 1.8;">
                    <div><strong>App Version:</strong> 1.0.0</div>
                    <div><strong>Build:</strong> 2026.04</div>
                    <div><strong>Python:</strong> 3.9+</div>
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #ecf0f1;">
                        <a href="#" style="color: #667eea; text-decoration: none; font-weight: 600;">Check for updates</a>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)