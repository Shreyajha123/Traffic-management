"""
TrafficSense AI - Flask Web Application Backend
===============================================
Project: Fundamentals of Artificial Intelligence (FAI) Mini-Project
Description: Intelligent Traffic Detection, Prediction & Management System
Author: TrafficSense AI Team
"""

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    abort
)

# Load environment variables
load_dotenv()

# App Configuration
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "trafficsense-ai-fai-secret-2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "traffic_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "model_metrics.json")

# Congestion Thresholds (Demonstration Standard)
CONGESTION_THRESHOLDS = {
    "low_max": 299,
    "medium_min": 300,
    "medium_max": 700,
    "high_min": 701,
}

DAYS_MAP = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Lazy-loaded cached dataset and model
_cached_df = None
_cached_model = None


def get_dataset():
    """Returns pandas DataFrame of the dataset or None if missing."""
    global _cached_df
    if _cached_df is not None:
        return _cached_df
    if os.path.exists(DATA_PATH):
        try:
            _cached_df = pd.read_csv(DATA_PATH)
            return _cached_df
        except Exception as e:
            logging.error(f"Error reading dataset: {e}")
            return None
    return None


def get_model():
    """Loads and caches the ML pipeline."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if os.path.exists(MODEL_PATH):
        try:
            _cached_model = joblib.load(MODEL_PATH)
            return _cached_model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    return None


def classify_congestion(count):
    """Categorizes vehicle volume into Low, Medium, or High."""
    if count < 300:
        return "Low"
    elif count <= 700:
        return "Medium"
    else:
        return "High"


def is_gemini_configured():
    """Checks whether a valid Google Gemini API key is configured."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and key != "your_key_here")


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route("/")
def index():
    """Home / Landing page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Traffic Analytics Dashboard page."""
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    """ML Prediction & Traffic Management Agent page."""
    return render_template("predict.html")


@app.route("/map")
def map_page():
    """Interactive Bengaluru Traffic Map page."""
    return render_template("map.html")


@app.route("/assistant")
def assistant_page():
    """AI Traffic Assistant chat page."""
    return render_template("assistant.html")


@app.route("/data")
def data_page():
    """Dataset & Report exploration page."""
    return render_template("data.html")


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    """Checks the status of the dataset, ML model, and Gemini API."""
    data_exists = os.path.exists(DATA_PATH)
    model_exists = os.path.exists(MODEL_PATH)
    metrics = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    total_records = 0
    if data_exists:
        df = get_dataset()
        if df is not None:
            total_records = len(df)

    return jsonify({
        "status": "online",
        "dataset_exists": data_exists,
        "total_records": total_records,
        "model_trained": model_exists,
        "gemini_configured": is_gemini_configured(),
        "metrics": metrics.get("metrics") if metrics else None,
        "thresholds": CONGESTION_THRESHOLDS,
        "disclaimer": "All dataset metrics and predictions are educational simulations."
    })


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """Calculates summary statistics and aggregated chart data from the dataset."""
    df = get_dataset()
    if df is None or df.empty:
        return jsonify({
            "success": False,
            "error": "Dataset not found. Please run 'python generate_data.py' to generate data."
        }), 404

    # 1. Metric Cards
    total_records = int(len(df))
    avg_vehicle_count = round(float(df["vehicle_count"].mean()), 1)
    avg_speed = round(float(df["avg_speed"].mean()), 1)

    # Congestion breakdown
    low_count = int((df["vehicle_count"] < 300).sum())
    med_count = int(((df["vehicle_count"] >= 300) & (df["vehicle_count"] <= 700)).sum())
    high_count = int((df["vehicle_count"] > 700).sum())
    congestion_pct = round(((med_count + high_count) / total_records) * 100, 1)

    # Peak traffic hour
    hourly_mean = df.groupby("hour")["vehicle_count"].mean()
    peak_hour = int(hourly_mean.idxmax())

    # 2. Hourly Chart Data (0 - 23)
    hourly_df = df.groupby("hour").agg({
        "vehicle_count": "mean",
        "avg_speed": "mean"
    }).reset_index()

    hourly_chart = {
        "hours": [f"{int(h):02d}:00" for h in hourly_df["hour"]],
        "volumes": [round(float(v), 1) for v in hourly_df["vehicle_count"]],
        "speeds": [round(float(s), 1) for s in hourly_df["avg_speed"]]
    }

    # 3. Road Volume Distribution
    road_df = df.groupby("road_name").agg({
        "vehicle_count": "mean",
        "avg_speed": "mean"
    }).reset_index().sort_values(by="vehicle_count", ascending=False)

    road_chart = {
        "roads": road_df["road_name"].tolist(),
        "volumes": [round(float(v), 1) for v in road_df["vehicle_count"]],
        "speeds": [round(float(s), 1) for s in road_df["avg_speed"]]
    }

    # 4. Weather Impact Chart Data
    weather_df = df.groupby("weather").agg({
        "vehicle_count": "mean",
        "avg_speed": "mean"
    }).reset_index()

    weather_chart = {
        "conditions": weather_df["weather"].tolist(),
        "volumes": [round(float(v), 1) for v in weather_df["vehicle_count"]],
        "speeds": [round(float(s), 1) for s in weather_df["avg_speed"]]
    }

    # 5. Congestion Pie Breakdown
    congestion_chart = {
        "labels": ["Low (<300)", "Medium (300-700)", "High (>700)"],
        "counts": [low_count, med_count, high_count],
        "percentages": [
            round((low_count / total_records) * 100, 1),
            round((med_count / total_records) * 100, 1),
            round((high_count / total_records) * 100, 1),
        ]
    }

    return jsonify({
        "success": True,
        "metrics": {
            "total_records": total_records,
            "avg_vehicle_count": avg_vehicle_count,
            "avg_speed": avg_speed,
            "congestion_percentage": congestion_pct,
            "peak_traffic_hour": f"{peak_hour:02d}:00"
        },
        "charts": {
            "hourly": hourly_chart,
            "road": road_chart,
            "weather": weather_chart,
            "congestion": congestion_chart
        },
        "disclaimer": "Calculated dynamically from synthetic demonstration dataset."
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    ML Prediction Endpoint.
    Expects JSON: {date, hour, weather, road_name}
    Auto-computes day_of_week and is_weekend from date.
    """
    model = get_model()
    if model is None:
        return jsonify({
            "success": False,
            "error": "Trained Machine Learning model not found. Please execute 'python train_model.py' to generate model/traffic_model.pkl."
        }), 404

    payload = request.get_json() or {}

    # Extract & validate fields
    date_str = payload.get("date", "").strip()
    hour = payload.get("hour")
    weather = payload.get("weather", "").strip()
    road_name = payload.get("road_name", "").strip()

    if not date_str or hour is None or not weather or not road_name:
        return jsonify({
            "success": False,
            "error": "Missing required fields: date, hour, weather, and road_name are mandatory."
        }), 400

    try:
        hour = int(hour)
        if not (0 <= hour <= 23):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Hour must be an integer between 0 and 23."
        }), 400

    # Parse date and derive day_of_week and is_weekend
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = DAYS_MAP[parsed_date.weekday()]
        is_weekend = 1 if parsed_date.weekday() >= 5 else 0
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Invalid date format. Expected YYYY-MM-DD."
        }), 400

    # Build feature DataFrame matching exact training feature names
    input_data = pd.DataFrame([{
        "hour": hour,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "weather": weather,
        "road_name": road_name
    }])

    try:
        prediction = model.predict(input_data)[0]
        predicted_count = max(0, int(round(prediction)))
        congestion_level = classify_congestion(predicted_count)

        # Estimate realistic speed from regression relationship
        est_speed = max(8.0, min(60.0, round(52.0 - (predicted_count / 1100.0) * 40.0, 1)))
        if weather == "Rainy":
            est_speed = max(6.0, round(est_speed * 0.70, 1))

        return jsonify({
            "success": True,
            "prediction": {
                "predicted_count": predicted_count,
                "congestion_level": congestion_level,
                "estimated_speed": est_speed,
                "road_name": road_name,
                "date": date_str,
                "hour": hour,
                "time_display": f"{hour:02d}:00",
                "day_of_week": day_of_week,
                "is_weekend": bool(is_weekend),
                "weather": weather
            },
            "thresholds": CONGESTION_THRESHOLDS,
            "disclaimer": "Predicted vehicle counts are educational estimates produced by RandomForestRegressor."
        })
    except Exception as e:
        logging.error(f"Prediction inference error: {e}")
        return jsonify({
            "success": False,
            "error": f"Model inference error: {str(e)}"
        }), 500


@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    """
    AI Assistant Chatbot Endpoint.
    Uses Google Gemini API via official google-genai SDK if configured.
    Falls back gracefully to rich educational responses if API key is not configured.
    """
    payload = request.get_json() or {}
    user_query = payload.get("message", "").strip()

    if not user_query:
        return jsonify({
            "success": False,
            "error": "Query message cannot be empty."
        }), 400

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_gemini = bool(api_key and api_key != "your_key_here")

    if use_gemini:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            system_instruction = (
                "You are TrafficSense AI, an intelligent urban mobility and traffic management assistant "
                "for Bengaluru city, developed for an academic Fundamentals of Artificial Intelligence (FAI) project. "
                "Provide clear, professional, structured, and beginner-friendly explanations. "
                "Cover traffic flow theories, bottlenecks (Silk Board, KR Puram, Hebbal, etc.), weather impacts, "
                "and modern AI solutions (signal timing optimization, route diversion, transit integration). "
                "Always be constructive and concise."
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_instruction}\n\nUser Question: {user_query}"
            )

            return jsonify({
                "success": True,
                "reply": response.text,
                "source": "Google Gemini 2.5 Flash",
                "is_demo": False
            })

        except Exception as e:
            logging.warning(f"Gemini API request failed: {e}. Falling back to educational demo engine.")

    # Educational Demo Fallback (Rule-Based Knowledge Engine)
    fallback_reply = generate_demo_assistant_reply(user_query)
    return jsonify({
        "success": True,
        "reply": fallback_reply,
        "source": "TrafficSense Educational Knowledge Engine",
        "is_demo": True,
        "notice": "Gemini API key is not configured in .env. Showing high-quality educational demonstration response."
    })


def generate_demo_assistant_reply(query):
    """Generates structured educational responses when Gemini API is unconfigured."""
    q = query.lower()

    if "silk board" in q:
        return (
            "### Silk Board Junction Analysis\n\n"
            "**Why Silk Board gets congested:**\n"
            "- It serves as the primary gateway linking Electronic City, Outer Ring Road (ORR), and Koramangala.\n"
            "- Multiple grade separators converge heavy commercial, IT employee, and intercity bus traffic.\n\n"
            "**AI & Engineering Solutions:**\n"
            "1. **Adaptive Signal Control (ITMS):** Dynamic green wave allocation prioritizing ORR directional surges.\n"
            "2. **Grade Separation & Metro Connectivity:** The Yellow Line Metro interchange significantly absorbs private commuter vehicle load.\n"
            "3. **Micro-diversion:** Promoting Sarjapur Road and BTM layout bypass routes during peak hours (8:30-11:00 AM)."
        )
    elif "mg road" in q:
        return (
            "### MG Road Traffic Dynamics\n\n"
            "**Characteristics:**\n"
            "- High mix of commercial shopping, recreational traffic, and central business district commuters.\n"
            "- Constrained carriageway width with Metro pillars in the median.\n\n"
            "**Traffic Optimization Recommendations:**\n"
            "1. **Smart Parking Guidance:** Parking search causes up to 25% of circulation delays; IoT parking apps reduce wandering vehicles.\n"
            "2. **Pedestrian Crossing Phasing:** Synchronized pedestrian-scramble phases reduce mid-block pedestrian-vehicle friction.\n"
            "3. **Last-Mile Feeder Transit:** Enhanced frequency of electric shuttle buses from Trinity and Cubbon Park metro stations."
        )
    elif "rain" in q or "weather" in q:
        return (
            "### Impact of Weather on Urban Traffic Flow\n\n"
            "- **Speed Reduction:** Vehicle speeds drop by 30% to 50% during rainfall due to reduced tire traction and poor visibility.\n"
            "- **Waterlogging Bottlenecks:** Low-lying underpasses (such as Hebbal and Marathahalli) lose active lanes, creating immediate upstream shockwaves.\n"
            "- **Modal Shift:** Two-wheeler riders seek shelter under flyovers or transition to app-based cabs, surging four-wheeler volumes.\n\n"
            "*Key AI mitigation:* Pre-emptive traffic rerouting around sensor-flagged flood zones and automated variable message signs (VMS)."
        )
    elif "model" in q or "random forest" in q or "fai" in q or "predict" in q:
        return (
            "### Random Forest in TrafficSense AI\n\n"
            "- **What it is:** An ensemble machine learning technique combining multiple decision trees (bagging).\n"
            "- **Features Used:** `hour`, `day_of_week`, `weather`, `road_name`, and `is_weekend`.\n"
            "- **Why Random Forest:** It effectively captures non-linear interactions (e.g., Silk Board behaves differently on rainy Monday mornings vs. Sunday afternoons) without overfitting.\n"
            "- **Classification:** Continuous volume predictions are mapped to Low (<300), Medium (300-700), and High (>700) congestion categories."
        )
    else:
        return (
            "### TrafficSense AI Urban Mobility Overview\n\n"
            f"You asked: *\"{query}\"*\n\n"
            "**Key Insights for Urban Traffic Management:**\n"
            "1. **Demand Management:** Staggered work hours across tech corridors (Whitefield, Bellandur, E-City) flatten peak spikes by up to 22%.\n"
            "2. **Predictive Dispatch:** Using Machine Learning forecasts to adjust signal cycle lengths before queues build up prevents gridlock.\n"
            "3. **Multi-Modal Integration:** Seamless bus-metro ticketing encourages mass transit adoption over personal vehicles.\n\n"
            "*(Tip: To enable live Google Gemini AI answers, add your free GEMINI_API_KEY to the project .env file!)*"
        )


@app.route("/api/agent", methods=["POST"])
def api_agent():
    """
    AI Traffic Management Agent Endpoint.
    Analyzes prediction attributes and generates actionable advisory policies.
    """
    payload = request.get_json() or {}
    road = payload.get("road_name", "Selected Corridor")
    hour = payload.get("hour", 9)
    weather = payload.get("weather", "Clear")
    volume = payload.get("predicted_count", 500)
    congestion = payload.get("congestion_level", classify_congestion(volume))
    day = payload.get("day_of_week", "Weekday")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    use_gemini = bool(api_key and api_key != "your_key_here")

    if use_gemini:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            agent_prompt = (
                f"Act as the TrafficSense AI Autonomous Traffic Management Agent for Bengaluru. "
                f"Given the situation: Road: {road}, Hour: {hour:02d}:00, Day: {day}, Weather: {weather}, "
                f"Predicted Volume: {volume} vehicles/hr, Congestion Level: {congestion}. "
                f"Generate exactly 4 distinct, actionable traffic management measures. "
                f"For each measure, specify: 1) Policy Action, 2) Expected Operational Impact, 3) Why it works. "
                f"Format cleanly in markdown bullet points. Do not claim to physically move signals."
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=agent_prompt
            )

            return jsonify({
                "success": True,
                "road": road,
                "congestion_level": congestion,
                "volume": volume,
                "suggestions": response.text,
                "source": "Google Gemini 2.5 Flash Agent",
                "is_demo": False
            })
        except Exception as e:
            logging.warning(f"Gemini Agent call failed: {e}. Falling back to Rule-Based Agent Engine.")

    # Rule-Based Policy Recommendation Generator
    policy_output = generate_rule_based_agent_policy(road, hour, weather, volume, congestion)
    return jsonify({
        "success": True,
        "road": road,
        "congestion_level": congestion,
        "volume": volume,
        "suggestions": policy_output,
        "source": "TrafficSense Rule-Based Management Engine",
        "is_demo": True,
        "disclaimer": "Educational simulation recommendations. The system does not directly manipulate municipal traffic controllers."
    })


def generate_rule_based_agent_policy(road, hour, weather, volume, congestion):
    """Generates structured expert management policies based on deterministic rules."""
    measures = []

    # 1. Signal & Capacity Policy
    if congestion == "High":
        measures.append(
            f"**1. Adaptive Dynamic Green Extension (High Priority)**\n"
            f"- *Action:* Extend green phase by +35 seconds along primary {road} corridor; truncate minor side-street phases.\n"
            f"- *Why it works:* Prevents queue spillbacks from blocking adjacent box junctions during heavy loads ({volume} veh/hr)."
        )
    elif congestion == "Medium":
        measures.append(
            f"**1. Coordinated Green Wave Synchronization**\n"
            f"- *Action:* Maintain continuous 50 km/h green wave progression across sequential intersections along {road}.\n"
            f"- *Why it works:* Smooths stop-and-go platooning and keeps vehicle flow within sustainable corridor capacity."
        )
    else:
        measures.append(
            f"**1. Energy-Conserving Cycle Truncation**\n"
            f"- *Action:* Shorten overall cycle length from 120s to 60s with vehicle actuation sensors.\n"
            f"- *Why it works:* Avoids empty red light waiting times during low volume ({volume} veh/hr)."
        )

    # 2. Weather Specific Policy
    if weather == "Rainy":
        measures.append(
            f"**2. Monsoon Hazard & Drainage Response Protocol**\n"
            f"- *Action:* Dispatch BBMP quick-response drainage team to known waterlogging dips near {road}; activate Variable Message Signs (VMS) cautioning 30 km/h max speed.\n"
            f"- *Why it works:* Heavy wet roads double stopping distance; mitigating puddles prevents single-lane bottlenecks."
        )
    elif weather == "Foggy":
        measures.append(
            f"**2. Low Visibility Safety Beacon Activation**\n"
            f"- *Action:* Turn on amber fog beacons and synchronize street lighting illumination.\n"
            f"- *Why it works:* Prevents rear-end collisions during early morning visibility dips."
        )
    else:
        measures.append(
            f"**2. High-Flow Corridor Lane Discipline Monitoring**\n"
            f"- *Action:* Deploy automated CCTV lane tracking to penalize illegal shoulder parking along {road}.\n"
            f"- *Why it works:* Recovers 100% of physical lane width for moving traffic."
        )

    # 3. Diversion & Route Guidance Policy
    if congestion == "High":
        measures.append(
            f"**3. Strategic Perimeter Traffic Diversion**\n"
            f"- *Action:* Broadcast detour advisories via navigation apps (e.g., Google Maps / MapmyIndia) recommending alternate bypasses around {road}.\n"
            f"- *Why it works:* Shaves 15-20% of incoming transit volume before it enters the critical bottleneck bottleneck."
        )
    else:
        measures.append(
            f"**3. Variable Advisory Messaging**\n"
            f"- *Action:* Display estimated corridor traversal time on overhead LED signage.\n"
            f"- *Why it works:* Informs commuters of expected travel times, stabilizing driving behavior."
        )

    # 4. Public Transit Prioritization
    measures.append(
        f"**4. Transit Priority (BMTC Feeder Boost)**\n"
        f"- *Action:* Increase BMTC feeder bus frequency by 2 extra runs per hour along the {road} transit corridor.\n"
        f"- *Why it works:* One full bus takes 40+ private cars off the road, directly addressing root congestion."
    )

    return "\n\n".join(measures)


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    """
    Paginated, searchable, filterable dataset explorer.
    Query params: page, limit, road, weather, congestion, search
    """
    df = get_dataset()
    if df is None or df.empty:
        return jsonify({"success": False, "error": "Dataset not found"}), 404

    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 20, type=int)
    road = request.args.get("road", "").strip()
    weather = request.args.get("weather", "").strip()
    congestion = request.args.get("congestion", "").strip()
    search = request.args.get("search", "").strip()

    filtered_df = df.copy()

    # Filter by road
    if road and road.lower() != "all":
        filtered_df = filtered_df[filtered_df["road_name"].str.lower() == road.lower()]

    # Filter by weather
    if weather and weather.lower() != "all":
        filtered_df = filtered_df[filtered_df["weather"].str.lower() == weather.lower()]

    # Filter by congestion
    if congestion and congestion.lower() != "all":
        if congestion.lower() == "low":
            filtered_df = filtered_df[filtered_df["vehicle_count"] < 300]
        elif congestion.lower() == "medium":
            filtered_df = filtered_df[(filtered_df["vehicle_count"] >= 300) & (filtered_df["vehicle_count"] <= 700)]
        elif congestion.lower() == "high":
            filtered_df = filtered_df[filtered_df["vehicle_count"] > 700]

    # Search keyword
    if search:
        search_lower = search.lower()
        mask = (
            filtered_df["road_name"].astype(str).str.lower().str.contains(search_lower) |
            filtered_df["weather"].astype(str).str.lower().str.contains(search_lower) |
            filtered_df["day_of_week"].astype(str).str.lower().str.contains(search_lower) |
            filtered_df["date"].astype(str).str.contains(search_lower)
        )
        filtered_df = filtered_df[mask]

    total_records = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_records / limit)))
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_data = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")

    # Add congestion label to each record
    for record in page_data:
        record["congestion_level"] = classify_congestion(record["vehicle_count"])

    return jsonify({
        "success": True,
        "total": total_records,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "data": page_data
    })


@app.route("/api/download-csv", methods=["GET"])
def api_download_csv():
    """Exports and downloads the actual CSV dataset used by the application."""
    if not os.path.exists(DATA_PATH):
        return jsonify({
            "success": False,
            "error": "Dataset file not found. Please run 'python generate_data.py' first."
        }), 404

    return send_file(
        DATA_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="traffic_data.csv"
    )


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "An internal server error occurred."
    }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print("\n" + "=" * 65)
    print(f" TrafficSense AI Web Application running on http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
