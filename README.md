# TrafficSense AI – Intelligent Traffic Detection, Prediction & Management System

> **Academic Course Project**: Fundamentals of Artificial Intelligence (FAI)  
> **Platform**: Windows / Cross-Platform Python Web Application  
> **Author**: TrafficSense AI Academic Team  

---

## 1. Project Overview

**TrafficSense AI** is an educational, end-to-end urban mobility web platform developed for the *Fundamentals of Artificial Intelligence (FAI)* mini-project. It combines classical Machine Learning regression with Generative AI policy synthesis to forecast vehicle volume, categorize congestion levels, and recommend tactical traffic management interventions across key Bengaluru corridors.

### Core Capabilities:
- **Synthetic Data Generation**: Parametrically generates 2,000 demonstration traffic records modeling rush-hour curves, weather hazards, and corridor capacity.
- **Machine Learning Regression**: Employs a Scikit-Learn `RandomForestRegressor` with full `OneHotEncoder` and `StandardScaler` preprocessing to forecast hourly vehicle counts ($R^2 > 96\%$).
- **Demonstration Congestion Classification**: Categorizes traffic into **Low (<300)**, **Medium (300–700)**, and **High (>700)** density tiers.
- **Dynamic Analytics Dashboard**: Computes live KPIs, rush-hour volume curves, weather impact graphs, and congestion breakdowns using Chart.js.
- **Interactive Bengaluru Geospatial Map**: Renders 8 illustrative junction markers (Silk Board, KR Puram, Hebbal, Marathahalli, Whitefield, etc.) on Leaflet.js and OpenStreetMap with color-coded status pins.
- **AI Urban Mobility Assistant**: Conversational assistant powered by Google Gemini 2.5 Flash (via official `google-genai` SDK) with an educational rule-based fallback when offline.
- **Autonomous Traffic Management Agent**: Translates quantitative ML prediction counts into 4 actionable municipal engineering policies (adaptive signal green extensions, transit feeder dispatch, monsoon flood protocols, and perimeter diversions).
- **Data Explorer & CSV Export**: Searchable, filterable historical data table with a one-click button to download the actual `traffic_data.csv` used by the system.

---

## 2. Technology Stack

Strictly designed to meet FAI course requirements (**No React, No Angular, No Node.js backend**):

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3 (Custom Dark Navy & White Theme), Vanilla JavaScript (ES6+) |
| **Backend** | Python 3.10+, Flask 3.x |
| **Machine Learning** | Scikit-Learn (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`), Pandas, NumPy, Joblib |
| **AI Assistant / Agent** | Google Gemini API (Official `google-genai` SDK), `python-dotenv` |
| **Data Visualizations** | Chart.js 4.4 (Line, Grouped Bar, Doughnut, Horizontal Benchmark) |
| **Geospatial Mapping** | Leaflet.js 1.9, OpenStreetMap |

---

## 3. Project Directory Structure

```text
trafficsence_AI/
│
├── app.py                     # Flask web server & REST API (13 endpoints)
├── generate_data.py           # Synthetic dataset generator (2,000 records)
├── train_model.py             # Scikit-Learn Random Forest pipeline trainer
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules (prevents .env leakage)
├── README.md                  # Complete documentation and beginner guide
│
├── dataset/
│   └── traffic_data.csv       # Generated demonstration traffic records
│
├── model/
│   ├── traffic_model.pkl      # Serialized Scikit-Learn ML pipeline
│   └── model_metrics.json     # Performance scores (MAE, RMSE, R²) & metadata
│
├── templates/
│   ├── index.html             # Home page (Intro, Objective, Workflow, Tech Stack)
│   ├── dashboard.html         # Live Analytics Dashboard (KPIs & 4 Chart.js charts)
│   ├── predict.html           # ML Prediction form & Autonomous AI Agent
│   ├── map.html               # Leaflet.js Bengaluru corridor map with filters
│   ├── assistant.html         # Gemini AI Assistant chatbot with demo fallback
│   └── data.html              # Dataset explorer, filters, and CSV download
│
└── static/
    ├── css/
    │   └── style.css          # Master stylesheet (Dark navy blue & white theme)
    └── js/
        ├── main.js            # Frontend logic (AJAX, Agent, Chatbot, Table)
        └── chart.js           # Chart.js initialization and dynamic updates
```

---

## 4. Understanding What Each File Does

- `generate_data.py`: Creates 2,000 synthetic Bengaluru traffic records with date, hour, day of week, weather condition, road name, vehicle count, and speed. Models realistic morning/evening rush hours and rainy speed drops.
- `train_model.py`: Reads the CSV, validates data types, handles preprocessing using `ColumnTransformer` (OneHotEncoder for categorical features + StandardScaler for numeric features), splits data into 80% train / 20% test, fits a `RandomForestRegressor`, evaluates MAE, RMSE, and $R^2$, and exports the entire pipeline to `model/traffic_model.pkl`.
- `app.py`: The Flask backend serving 6 user-facing HTML pages and 7 JSON REST API endpoints (`/api/dashboard`, `/api/predict`, `/api/assistant`, `/api/agent`, `/api/dataset`, `/api/download-csv`, `/api/model-status`).
- `templates/*.html`: Clean, semantic HTML templates rendering each section of the application.
- `static/css/style.css`: Polished dark navy blue (`#0a0f1d`, `#0f172a`, `#1e293b`) and white theme with glowing metric cards, responsive mobile menu, custom Leaflet popups, and typography.
- `static/js/chart.js`: Manages Chart.js instances, updating charts dynamically based on backend JSON feeds.
- `static/js/main.js`: Powers form validation, asynchronous API communication via `fetch()`, animated counter effects, chat bubble generation, markdown parsing, and data table filtering.

---

## 5. Step-by-Step Beginner Guide to Running on Windows

Follow these simple steps in order using **PowerShell** or **Command Prompt (cmd)**.

### Step 1: Open the Project Directory
Open PowerShell or Command Prompt, then navigate into the project folder:
```powershell
cd c:\Users\shrey\OneDrive\Desktop\trafficsence_AI
```

### Step 2: Create a Python Virtual Environment
A virtual environment keeps your project packages isolated and clean:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
Activate the environment so that commands use your local packages:
```powershell
venv\Scripts\activate
```
*(You will see `(venv)` appear at the beginning of your command prompt line).*

> **Note for PowerShell Users**: If you see an `Execution_Policies` script error when activating, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> and then retry `venv\Scripts\activate`.

### Step 4: Install Dependencies
Install all required libraries specified in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 5: Generate the Synthetic Dataset
Run the data generator to create `dataset/traffic_data.csv`:
```powershell
python generate_data.py
```
*Output*: Successfully generated 2,000 demonstration records.

### Step 6: Train the Machine Learning Model
Train the Scikit-Learn Random Forest Regressor and serialize it to `model/traffic_model.pkl`:
```powershell
python train_model.py
```
*Expected Evaluation Output*:
- **Mean Absolute Error (MAE)**: ~41.8 vehicles
- **Root Mean Squared Error (RMSE)**: ~53.1 vehicles
- **Test $R^2$ Score**: ~0.965 (96.5% variance explained)

### Step 7: (Optional) Configure Google Gemini API Key
To enable live responses from Google Gemini 2.5 Flash:
1. Make a copy of `.env.example` named `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` in Notepad or VS Code:
   ```powershell
   notepad .env
   ```
3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
4. Replace `your_key_here` with your key:
   ```env
   GEMINI_API_KEY=AIzaSy...your_actual_key
   ```
5. Save and close the file.

> **Offline Demonstration Support**: If you do not configure an API key, the website **will not crash**. The AI Assistant and Autonomous Agent will automatically switch to the built-in **Educational Knowledge Engine**, which provides structured, realistic responses with a clear educational badge.

### Step 8: Start the Flask Web Application
Launch the Flask development server:
```powershell
python app.py
```

### Step 9: Open the Application in Your Web Browser
Open your browser (Chrome, Edge, Firefox) and navigate to:
```text
http://127.0.0.1:5000
```

---

## 6. How to Test Each Feature in Your Presentation

| Feature | Where to Click | What to Demonstrate to Your Teacher |
|---|---|---|
| **1. Overview & Objective** | Home (`/`) | Explain the FAI project scope, 4-step workflow, and strict Flask architecture. |
| **2. Dynamic Dashboard** | Dashboard (`/dashboard`) | Show that the KPI cards, Hourly curve (with 9 AM and 6 PM spikes), Congestion doughnut, and Weather charts are computed from the 2,000 records. |
| **3. ML Volume Prediction** | Predict & Agent (`/predict`) | Select **Silk Board**, set time to **09:00 AM**, select **Rainy**, and click **Predict**. Point out the predicted volume (~1,100+ veh), **High Congestion** badge, speed estimate, and benchmark chart. Then change to **MG Road** at **02:00 AM Clear** to demonstrate **Low Congestion** (<300 veh). |
| **4. Autonomous AI Agent** | Predict & Agent (`/predict`) | After running a prediction, scroll to the AI Agent panel and click **Generate Management Directives**. Show how the agent synthesizes 4 tactical engineering measures (Signal green extension, drainage alerts, BMTC feeder boosts). |
| **5. Bengaluru Map** | Bengaluru Map (`/map`) | Demonstrate the 8 markers on OpenStreetMap. Click the filter buttons (**High**, **Medium**, **Low**) and click on **Silk Board** or **Hebbal Flyover** to view the advisory popup. |
| **6. AI Assistant Chatbot** | AI Assistant (`/assistant`) | Click the preset question *"What causes the severe bottleneck at Silk Board Junction?"* and demonstrate the structured AI response with source attribution. |
| **7. Data Explorer & Export** | Dataset & Report (`/data`) | Search for "Rainy" or filter by "Silk Board", inspect paginated records, and click **Download Actual CSV** to show the browser downloading `traffic_data.csv`. |

---

## 7. Machine Learning & Congestion Architecture

### Features Selected for Regression:
1. `hour`: Continuous time-of-day integer ($0 - 23$) capturing diurnal commute patterns.
2. `day_of_week`: Categorical day (Monday through Sunday).
3. `is_weekend`: Binary flag ($0$ for Weekday, $1$ for Weekend) adjusting leisure vs corporate peaks.
4. `weather`: Categorical weather condition (`Clear`, `Cloudy`, `Rainy`, `Foggy`).
5. `road_name`: Categorical corridor identifier (`Silk Board`, `KR Puram`, `Hebbal`, etc.).

### Preprocessing Pipeline:
$$\text{Input Features} \longrightarrow \begin{cases} \text{Categorical} \longrightarrow \text{OneHotEncoder}(\text{handle\_unknown}=\text{'ignore'}) \\ \text{Numeric} \longrightarrow \text{StandardScaler}() \end{cases} \longrightarrow \text{RandomForestRegressor}(n=120)$$

### Demonstration Congestion Thresholds:
$$\text{Congestion Level} = \begin{cases} \text{Low} & \text{if } V < 300 \text{ veh/hr} \\ \text{Medium} & \text{if } 300 \le V \le 700 \text{ veh/hr} \\ \text{High} & \text{if } V > 700 \text{ veh/hr} \end{cases}$$

---

## 8. Important Academic Disclaimers & Limitations

1. **Synthetic Demonstration Data**: All 2,000 records were generated parametrically for academic simulation; they do not represent physical sensor measurements from the Bengaluru Traffic Police (BTP) or BBMP.
2. **Demonstration Map Pins**: Map markers and congestion colors are educational demonstrations of key Bengaluru bottlenecks, not real-time satellite surveillance.
3. **Machine Learning Estimates**: Predictions are mathematical outputs of a regression model trained on synthetic distributions, not official municipal traffic forecasts.
4. **Demonstration Thresholds**: The 300 and 700 vehicle cut-offs are pedagogical benchmarks chosen for this project and are not official Indian MoRTH or IRC road capacity standards.
5. **Advisory Nature of AI Suggestions**: Directives generated by the AI Agent and Gemini Assistant are academic suggestions; the system has no physical control over municipal traffic signals.
6. **External CDN Connectivity**: Displaying Leaflet map tiles and Chart.js requires an active internet connection to download CDN assets.

---

## 9. Future Scope: Connecting Real-World Smart City Data

In a production version, TrafficSense AI can be extended by:
1. Connecting directly to BTP's Intelligent Traffic Management System (ITMS) camera feeds via automated vehicle counting APIs.
2. Integrating loop detector data feeds and municipal Google Maps / TomTom mobility APIs.
3. Embedding physical microcontroller (Arduino / Raspberry Pi) traffic light controllers triggered via Flask webhooks for automated green-phase adjustments in a laboratory test bench.

---
&copy; 2026 TrafficSense AI &bull; Fundamentals of Artificial Intelligence (FAI) Mini-Project
