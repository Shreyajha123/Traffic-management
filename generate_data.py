"""
TrafficSense AI - Synthetic Traffic Data Generator
==================================================
Project: Fundamentals of Artificial Intelligence (FAI) Mini-Project
Purpose: Generates 2,000 synthetic demonstration traffic records for Bengaluru roads.

IMPORTANT EDUCATIONAL DISCLAIMER:
---------------------------------
This dataset is purely SYNTHETIC DEMONSTRATION DATA generated for academic and
educational modeling purposes. It does NOT represent real-time surveillance or
official sensor measurements from the Bengaluru Traffic Police (BTP) or BBMP.

How real-world datasets could be integrated in future versions:
1. Connecting to municipal traffic management APIs or open data portals.
2. Integrating loop detector / camera counting CSV feeds (e.g., BTP SCATS or ITMS).
3. Utilizing aggregated mobility data providers (e.g., TomTom / Google Maps Traffic API).
"""

import os
import csv
import random
from datetime import datetime, timedelta

# Configuration
NUM_RECORDS = 2000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "traffic_data.csv")

# Bengaluru demonstration roads and their base traffic multiplier
# Higher multiplier = busier junction / known bottleneck
ROADS = {
    "Silk Board": {"weight": 1.45, "base_speed": 22},
    "KR Puram": {"weight": 1.40, "base_speed": 24},
    "Hebbal": {"weight": 1.35, "base_speed": 28},
    "Marathahalli": {"weight": 1.30, "base_speed": 25},
    "Electronic City": {"weight": 1.20, "base_speed": 35},
    "Whitefield": {"weight": 1.15, "base_speed": 30},
    "Indiranagar": {"weight": 1.10, "base_speed": 32},
    "MG Road": {"weight": 1.05, "base_speed": 28}
}

WEATHER_CONDITIONS = {
    "Clear": {"vol_mod": 1.0, "speed_mod": 1.0, "prob": 0.55},
    "Cloudy": {"vol_mod": 1.05, "speed_mod": 0.95, "prob": 0.20},
    "Rainy": {"vol_mod": 1.25, "speed_mod": 0.65, "prob": 0.18},
    "Foggy": {"vol_mod": 0.90, "speed_mod": 0.75, "prob": 0.07}
}

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_hourly_factor(hour, is_weekend):
    """
    Simulates realistic urban rush-hour traffic curves.
    """
    if is_weekend:
        # Weekend pattern: slow morning, peak in afternoon & evening leisure hours
        if 0 <= hour <= 6:
            return 0.25
        elif 7 <= hour <= 10:
            return 0.55
        elif 11 <= hour <= 15:
            return 0.95
        elif 16 <= hour <= 21:
            return 1.15
        else:
            return 0.50
    else:
        # Weekday pattern: steep morning peak (8-11 AM) and evening peak (17-20 PM)
        if 0 <= hour <= 5:
            return 0.18
        elif 6 <= hour <= 7:
            return 0.55
        elif 8 <= hour <= 11:
            return 1.55  # Morning peak
        elif 12 <= hour <= 16:
            return 0.85  # Afternoon lull
        elif 17 <= hour <= 20:
            return 1.65  # Evening peak
        elif 21 <= hour <= 22:
            return 0.70
        else:
            return 0.30


def generate_records(num_records=NUM_RECORDS):
    """
    Generates synthetic traffic records spanning a multi-month demonstration window.
    """
    records = []
    start_date = datetime(2026, 8, 1)
    road_keys = list(ROADS.keys())
    weather_keys = list(WEATHER_CONDITIONS.keys())
    weather_probs = [WEATHER_CONDITIONS[w]["prob"] for w in weather_keys]

    for i in range(num_records):
        # Sample random date within a 90-day window
        date_offset = random.randint(0, 90)
        record_date = start_date + timedelta(days=date_offset)
        hour = random.randint(0, 23)
        day_of_week = DAYS_OF_WEEK[record_date.weekday()]
        is_weekend = 1 if record_date.weekday() >= 5 else 0

        # Sample road and weather
        road_name = random.choice(road_keys)
        weather = random.choices(weather_keys, weights=weather_probs, k=1)[0]

        # Calculate vehicle volume
        road_info = ROADS[road_name]
        weather_info = WEATHER_CONDITIONS[weather]
        hourly_factor = get_hourly_factor(hour, is_weekend)

        # Baseline volume formula with stochastic Gaussian noise
        base_volume = 420  # average baseline vehicles per hour per corridor
        calculated_volume = (
            base_volume
            * road_info["weight"]
            * hourly_factor
            * weather_info["vol_mod"]
            + random.gauss(0, 45)
        )
        vehicle_count = max(40, int(round(calculated_volume)))

        # Average speed formula (inverse relationship with congestion, affected by weather)
        # Higher volume -> slower speed; Rain -> significantly slower speed
        max_speed = road_info["base_speed"] * 1.5
        min_speed = 8.0  # crawl speed in heavy traffic

        # Speed reduction factor based on congestion load (normalized around 800)
        congestion_load = min(1.0, vehicle_count / 1100.0)
        raw_speed = max_speed - (congestion_load * (max_speed - min_speed))
        raw_speed *= weather_info["speed_mod"]
        raw_speed += random.gauss(0, 2.0)
        avg_speed = round(max(min_speed, min(65.0, raw_speed)), 1)

        records.append({
            "date": record_date.strftime("%Y-%m-%d"),
            "hour": hour,
            "day_of_week": day_of_week,
            "weather": weather,
            "road_name": road_name,
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "is_weekend": is_weekend
        })

    # Sort deterministically by date and hour for clean viewing
    records.sort(key=lambda r: (r["date"], r["hour"], r["road_name"]))
    return records


def main():
    print("=" * 65)
    print(" TrafficSense AI: Generating Synthetic Demonstration Traffic Data")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = generate_records(NUM_RECORDS)

    fieldnames = [
        "date", "hour", "day_of_week", "weather",
        "road_name", "vehicle_count", "avg_speed", "is_weekend"
    ]

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Successfully generated {len(records)} records.")
    print(f"[FILE] Saved to: {OUTPUT_FILE}")
    print("\nDemonstration Summary:")
    print(f"- Total records: {len(records)}")
    print(f"- Target roads: {', '.join(ROADS.keys())}")
    print(f"- Features: {', '.join(fieldnames)}")
    print("-" * 65)
    print("NOTE: These records are synthetic for educational simulation purposes.")
    print("=" * 65)


if __name__ == "__main__":
    main()
