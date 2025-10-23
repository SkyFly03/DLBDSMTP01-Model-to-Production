# Model to Production: Anomaly Detection in an IoT Setting - DLBDSMTP01

## Project Overview

This project demonstrates a full end-to-end **Model-to-Production pipeline** for detecting turbine anomalies from simulated sensor data (temperature, humidity, sound).  
It includes model training, REST API prediction, streaming ingestion, visualization, and containerization for reproducible runs.

---

## 1. How to Clone and Run Locally

### Clone the Repository
```bash
git clone https://github.com/SkyFly03/DLBDSMTP01-Model-to-Production.git
cd DLBDSMTP01-Model-to-Production
```

### Create and Activate Virtual Environment (Python 3.10)
```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### Upgrade pip and Install Dependencies
```bash
python -q -m pip install --upgrade pip
pip install -q -r requirements.txt
```

### Run Project in Sequence
```bash
1) python -m app.train_eval           # simulate data, train model, create visuals
2) python -m app.api                  # start REST API (http://localhost:5000)
3) python -m app.sender               # send simulated data to API (HTTP POST)
4) python -m app.model                # load IsolationForest model
5) python -m app.visualize_from_csv   # visualize predictions from CSV log
```

---

## 2. Run with Docker and Docker Compose

### Build and Start Containers
- **Launch Docker Desktop** and wait until Docker is fully started, before executing the commands below!
```bash
docker compose up --build
```

### Stop Containers
```bash
docker compose down
```

**Services:**
- **API** – Flask REST service exposing prediction endpoints → `http://localhost:5000`
- **Sender** → continuously streams data to API for inference  
- **Healthcheck** → waits until REST API is live before starting sender  

---

## 3. Project Folder Structure

```
DLBDSMTP01 - M2P - Anomaly detection in an IoT setting/
│
├── app/                                   # main application package
│   ├── __init__.py                        # makes app a Python module
│   ├── api.py                             # REST API endpoints (/predict, /health, /model/info, /metrics)
│   ├── model.py                           # IsolationForest model loading and prediction logic
│   ├── sender.py                          # simulates live data streaming to API via HTTP POST
│   ├── train_eval.py                      # data simulation, model training, evaluation & visualization
│   ├── visualize_from_csv.py              # creates anomaly heatmaps & time-based PNG visuals from logs
│   └── outputs/                           # stores generated PNGs and CSV logs
│
├── archive/                               # stores early test files or deprecated scripts
│
├── models/                                # serialized ML models
│   └── turbine_iforest.pkl                # trained IsolationForest model
│
├── tests/                                 # project validation and integration tests
│   └── test.py                            # verifies API, metrics, model existence, and output visuals
│
├── metadata/                              # background or configuration references
│   └── metadata.txt                       # technical notes or version info
│
├── Dockerfile                             # defines image build (Flask + Python environment)
├── docker-compose.yml                     # orchestrates multi-container run (API + Sender)
├── requirements.txt                       # pinned Python dependencies
├── README.md                              # main project documentation
└── .gitignore / .dockerignore             # exclude unnecessary files from Git & Docker
```

---

## 4. Workflow Overview
<img width="2877" height="744" alt="Mermaid Chart 2025-10-18" src="https://github.com/user-attachments/assets/5ca155e8-c09a-4783-aa2b-daed8759d320" />

### **4.1 Concept & Architecture**
- Defines objective: Detect turbine anomalies using temperature, humidity, and sound.
- Architecture planning covers: module interaction, REST design, and data flow. 
- Output: `README.md` + system diagram (see `mermaid.md` or PNG).

### **4.2 Data Source & Simulation**
- Generates realistic fictional sample data.
- Features: `temperature (°C)`, `humidity (%)`, `sound_volume (arb)`.
- Split 70/10/20 for Train/Val/Test.
- Script: `app/train_eval.py`.

### **4.3 Train & Save Simple Model**
- Algorithm: **IsolationForest** (unsupervised anomaly detector) - detecting outliers.
- Model file saved: `models/turbine_iforest.pkl`.
- Visualizes key statistics (histograms, feature distributions, anomaly scores).
- Code: `app/model.py`.

### **4.4 Prediction API (REST)**
- Deploys a Flask-based REST interface for live inference.  
- Endpoints include: `/predict`, `/health`, `/model/info`, `/metrics`.
- Automatically loads the model on start → `turbine_iforest.pkl`.
- Logs every prediction to: `app/outputs/predictions_log.csv`.
- Code: `app/api.py`.

### **4.5 Data Ingestion & Streaming**
- `app/sender.py` sends HTTP POST requests to `/predict`, mimicing real-time turbine sensor streaming.
- Each record is POSTed to the API and Log data is stored as CSV.
- Clear anomaly visuals generated via `app/visualize_from_csv.py` → PNG outputs.

### **4.6 Reproducible Run (Container + Tests)**
- Containerized with `Dockerfile`→ full reproducibility
- `docker-compose.yml`connects both API and Sender containers.  
  
#### 4.6.1 Output: Build and launch: Docker Compose starts the API and sender containers.
<img width="1100" height="608" alt="docker-compose1" src="https://github.com/user-attachments/assets/893202cc-c033-4293-a5b6-0d759e37cd01" />

#### 4.6.2 Output: Live logs: Containers exchange sensor data and serve predictions in real time.
<img width="958" height="370" alt="docker-compose2" src="https://github.com/user-attachments/assets/fc1a8a97-20f9-4c17-b297-60d13f4c56d7" />

#### 4.6.3 Output: Health and prediction requests: Both services handle API traffic and log model outputs.
<img width="863" height="194" alt="docker-compose3" src="https://github.com/user-attachments/assets/5b7b3a71-07d0-4c59-92f5-3fef3e7b93c1" />

- Automated test: `tests/test.py`confirms that all artifacts (model, API, visuals) are correctly generated.
- Tests guarantee the project is production-ready and technically verifiable. 

<img width="719" height="204" alt="test py_run" src="https://github.com/user-attachments/assets/96f35212-cafa-4bd0-b6f5-c24cb362c06c" />

---

## 5. Results & Visuals

Output figures stored in:
```
app/outputs/
```
Examples include:
- `metrics_table.png` - Accuracy, Precision, Recall, F1, ROC_AUC (“Area Under the Receiver Operating Characteristic Curve”)
<img width="1485" height="433" alt="metrics_table" src="https://github.com/user-attachments/assets/87772000-2b03-4540-8c0c-64269aefe4e7" />

- `learning_dashboard_2x2.png` – training progress & model metrics 
<img width="1500" height="1200" alt="learning_dashboard_2x2" src="https://github.com/user-attachments/assets/7a1475fe-e02a-4002-906f-df23c30a6563" />
 
- `heatmaps_grid.png` – correlation + confusion matrix - anomaly correlation and feature relationships 
<img width="1645" height="466" alt="heatmaps_grid" src="https://github.com/user-attachments/assets/0b30b113-6678-4288-8560-33051eed8cba" />
  
- `anomalies_over_time.png` – anomalies per feature over time
<img width="1781" height="582" alt="anomalies_over_time" src="https://github.com/user-attachments/assets/fe1cbbfe-bdec-4863-ad07-97c914bb530d" />
  
- `histograms.png` - Normal distributions and anomaly values for each sensor feature, illustrating how abnormal data points differ from typical behavior.
<img width="1784" height="408" alt="histograms" src="https://github.com/user-attachments/assets/9dbb656d-07a8-4809-8947-c48fa0dd5cad" />
---

## 6. Conclusion

This project successfully simulates a complete MLOps-style workflow:
- From **data simulation to API deployment**
- With clear modular structure, testing, and containerized reproducibility  
- Demonstrating practical integration of data science and software engineering for anomaly detection in IoT settings.

---

