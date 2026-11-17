"""
TrafficSense AI - Comprehensive Automated Test Suite
====================================================
Tests all page routes, REST API endpoints, ML predictions,
dashboard aggregations, CSV downloads, and edge cases.
"""

import os
import json
import unittest
from app import app, DATA_PATH, MODEL_PATH, classify_congestion


class TrafficSenseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        app.config["TESTING"] = True

    def test_01_files_exist(self):
        """Verify that essential dataset and model files exist."""
        self.assertTrue(os.path.exists(DATA_PATH), f"Dataset missing at {DATA_PATH}")
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model missing at {MODEL_PATH}")

    def test_02_congestion_classification(self):
        """Verify Low, Medium, High congestion classification rules."""
        self.assertEqual(classify_congestion(150), "Low")
        self.assertEqual(classify_congestion(299), "Low")
        self.assertEqual(classify_congestion(300), "Medium")
        self.assertEqual(classify_congestion(500), "Medium")
        self.assertEqual(classify_congestion(700), "Medium")
        self.assertEqual(classify_congestion(701), "High")
        self.assertEqual(classify_congestion(1200), "High")

    def test_03_page_routes(self):
        """Verify all 6 HTML page routes return HTTP 200."""
        routes = ["/", "/dashboard", "/predict", "/map", "/assistant", "/data"]
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(
                response.status_code, 200, f"Route '{route}' returned {response.status_code}"
            )
            self.assertIn(b"TrafficSense", response.data)

    def test_04_api_model_status(self):
        """Verify GET /api/model-status returns correct metadata."""
        response = self.client.get("/api/model-status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("dataset_exists"))
        self.assertTrue(data.get("model_trained"))
        self.assertEqual(data.get("total_records"), 2000)
        self.assertIn("metrics", data)

    def test_05_api_dashboard(self):
        """Verify GET /api/dashboard calculates real statistics and chart data."""
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))

        metrics = data.get("metrics", {})
        self.assertEqual(metrics.get("total_records"), 2000)
        self.assertGreater(metrics.get("avg_vehicle_count"), 0)
        self.assertGreater(metrics.get("avg_speed"), 0)
        self.assertIn(":", metrics.get("peak_traffic_hour"))

        charts = data.get("charts", {})
        self.assertEqual(len(charts.get("hourly", {}).get("hours", [])), 24)
        self.assertEqual(len(charts.get("road", {}).get("roads", [])), 8)
        self.assertEqual(len(charts.get("congestion", {}).get("labels", [])), 3)

    def test_06_api_predict_valid(self):
        """Verify POST /api/predict produces valid ML forecasts."""
        test_payload = {
            "road_name": "Silk Board",
            "date": "2026-08-10",
            "hour": 9,
            "weather": "Rainy"
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(test_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))

        pred = data.get("prediction", {})
        self.assertGreater(pred.get("predicted_count"), 0)
        self.assertIn(pred.get("congestion_level"), ["Low", "Medium", "High"])
        self.assertEqual(pred.get("road_name"), "Silk Board")
        self.assertEqual(pred.get("day_of_week"), "Monday")
        self.assertFalse(pred.get("is_weekend"))

    def test_07_api_predict_invalid_input(self):
        """Verify POST /api/predict handles invalid and missing inputs."""
        # Missing fields
        res1 = self.client.post("/api/predict", data=json.dumps({}), content_type="application/json")
        self.assertEqual(res1.status_code, 400)

        # Invalid hour (> 23)
        res2 = self.client.post(
            "/api/predict",
            data=json.dumps({"road_name": "Hebbal", "date": "2026-08-10", "hour": 25, "weather": "Clear"}),
            content_type="application/json"
        )
        self.assertEqual(res2.status_code, 400)

        # Invalid date format
        res3 = self.client.post(
            "/api/predict",
            data=json.dumps({"road_name": "Hebbal", "date": "10-08-2026", "hour": 10, "weather": "Clear"}),
            content_type="application/json"
        )
        self.assertEqual(res3.status_code, 400)

    def test_08_api_assistant_and_fallback(self):
        """Verify POST /api/assistant delivers structured educational responses."""
        test_query = {"message": "How can traffic congestion be reduced at MG Road?"}
        response = self.client.post(
            "/api/assistant",
            data=json.dumps(test_query),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(len(data.get("reply")) > 20)
        self.assertIn("source", data)

    def test_09_api_agent(self):
        """Verify POST /api/agent synthesizes tactical traffic management policies."""
        agent_payload = {
            "road_name": "Silk Board",
            "hour": 9,
            "day_of_week": "Monday",
            "weather": "Rainy",
            "predicted_count": 950,
            "congestion_level": "High"
        }
        response = self.client.post(
            "/api/agent",
            data=json.dumps(agent_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("suggestions", data)
        self.assertTrue(len(data.get("suggestions")) > 50)

    def test_10_api_dataset_pagination_and_filter(self):
        """Verify GET /api/dataset supports pagination and road/weather filters."""
        # Standard page 1
        res = self.client.get("/api/dataset?page=1&limit=20")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(len(data.get("data", [])), 20)
        self.assertEqual(data.get("total"), 2000)

        # Filter by road
        res_road = self.client.get("/api/dataset?road=Silk%20Board&limit=10")
        self.assertEqual(res_road.status_code, 200)
        data_road = res_road.get_json()
        for r in data_road.get("data", []):
            self.assertEqual(r.get("road_name"), "Silk Board")

    def test_11_api_download_csv(self):
        """Verify GET /api/download-csv streams the actual CSV file."""
        response = self.client.get("/api/download-csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        self.assertIn(b"date,hour,day_of_week,weather,road_name,vehicle_count,avg_speed,is_weekend", response.data)


if __name__ == "__main__":
    unittest.main()
