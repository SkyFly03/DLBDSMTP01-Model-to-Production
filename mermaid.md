# ------------------------------------------------------------
# Mermaid chart
# ------------------------------------------------------------
# ------------------------------------------------------------
---
config:
  theme: neutral
  look: classic
  layout: dagre
---
flowchart LR
    N1["**1.
    Concept &amp; Architecture**
    ---------------------------------- 
    Task: 
    detect turbine anomalies
    ---------------------------------- 
    *README* + system diagram"] --> N2[("**2.
    Data Source & Simulation**
    ----------------------------------
    Fictional sample data: 
    temp, humidity, sound
    ----------------------------------
    70/10/20 split 
    ----------------------------------
    *app/train_eval.py*")]
    N2 --> N3[("**3.<br>Train &amp; Save Simple Model**
    ----------------------------------
    IsolationForest
    ----------------------------------
    Save model file:
    *models/turbine_iforest.pkl*
    *app/model.py*")]
    N3 --> N4[("**4. 
    Prediction API (REST)**
    ----------------------------------
    /predict /health 
    /model/info /metrics
    ----------------------------------
    Loads: *models/turbine_iforest.pkl* 
    on start
    ----------------------------------
    Logs: *app/outputs/predictions_log.csv*
    ----------------------------------
    *app/api.py*")]
    N4 --> N5[("**5.
    Data Ingestion & Streaming**
    ----------------------------------
    *app/sender.py* → HTTP POST /predict (JSON)
    ----------------------------------
    *visualize_from_csv.py* → PNGs from CSV log")]
    N5 --> N6[("**6.
    Reproducible Run 
    (Container + Tests)**
    ---------------------------------- 
    *Dockerfile* + 
    *docker-compose.yml*
    ----------------------------------
    API check: *test.py*")]
    N1@{ shape: cyl}
     N1:::yellow
     N2:::orange
     N3:::red
     N4:::blue
     N5:::teal
     N6:::green
    classDef yellow fill:#FFF5B1,stroke:#E6C14A,stroke-width:2,color:#111
    classDef orange fill:#FFC98B,stroke:#F2A552,stroke-width:2,color:#111
    classDef red    fill:#FFB3B8,stroke:#F28A92,stroke-width:2,color:#111
    classDef blue   fill:#AECBFA,stroke:#7DA6F2,stroke-width:2,color:#111
    classDef teal   fill:#B3E5FC,stroke:#72B6F2,stroke-width:2,color:#111
    classDef green  fill:#C7F3D6,stroke:#7BC89A,stroke-width:2,color:#111
    style N1 color:#000000,fill:#FFF9C4