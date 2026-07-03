# 🛡️ Hybrid Log Classification & Real-Time Analytics Platform

A production-ready, custom-trainable log classification platform. It leverages a **hybrid cascade pipeline (Regex + BERT ML Model + LLM)** to classify log streams with speed and high precision. Additionally, it offers **active human-in-the-loop corrections**, **closed-loop model retraining on database logs**, an **advanced model history registry**, and a **developer API integration helper**.

---

## 📺 Demo Walkthrough

Watch the complete video demo showing the platform's key features, including log ingestion, human correction, live simulator streaming, and automated model retraining:

[![Log Classification Platform Demo](backend/resources/arch.png)](backend/resources/demo_recording.webm)

*(Click the image above or navigate directly to the [Demo Video file](backend/resources/demo_recording.webm) to watch the recording.)*

---

## 🚀 Key Features

### 1. Hybrid Cascade Classification Pipeline

The backend routes incoming logs through a 3-stage intelligence cascade:

- **Stage 1: Dynamic Regex Rules**: Evaluates high-frequency, predictable log patterns first. Rules are managed directly in the UI and compiled in-memory (with a 5-second TTL cache) to ensure maximum throughput.
- **Stage 2: BERT Embeddings + Logistic Regression**: Complex log patterns are encoded using `SentenceTransformer` (`all-MiniLM-L6-v2`) and classified via a custom-trained Logistic Regression model (equipped with `class_weight='balanced'` to prevent minority class bias).
- **Stage 3: LLM (DeepSeek-R1 via Groq)**: Invoked as an intelligent fallback when training data is absent (e.g., for LegacyCRM logs) or classification certainty is low.

### 2. Active Learning & Human-in-the-Loop Corrections

- **Interactive Label Override**: Users can correct model predictions directly from the logs dashboard table.
- **Log Expansion Panel**: Click any log row to slide open an accordion details view containing the full log message, raw confidence probability, ingestion timestamp, and an inline dropdown to assign a corrected label.
- **Immediate Feedback**: Correcting a label instantly sets `user_corrected = True`, assigns `100%` confidence, flags the method as `"Manual override"`, and updates the dashboard charts in real-time.

### 3. Closed-Loop Training on Stored Database Logs

- **No-CSV Retraining**: Train new classifiers directly from the logs stored in your database by clicking **"Retrain Model on Database Logs"**.
- **Data Filtering**: The Celery task automatically extracts high-confidence logs (`confidence >= 0.8`) and user-corrected logs to construct a clean, high-quality training set, preventing the model from learning from prediction errors.
- **Training Console Output**: Progress logs and metrics are piped in real-time into the scrolling console emulator in the frontend.

### 4. Advanced Model Versioning & Registry

- **Trained Model History**: Track all trained versions inside a history table showing accuracy, records count, and active status.
- **Performance Inspection**: Click **"Inspect"** on any version to open a modal detailing the per-class precision, recall, and F1-score classification report.
- **Model Purging**: Click **"Delete"** to permanently delete inactive model version weights from disk and remove their records from the DB.

### 5. Developer API Keys & Integration Explorer

- **SaaS API Key Registry**: Generates secure tokens starting with `nlp_live_` to authenticate external integration calls. Keys are saved securely using SHA-256 hashes, with prefixes (e.g. `nlp_live_abc12345`) exposed for registry visibility.
- **Request Telemetry**: Tracks live usage metrics (`total_calls`) individually across all active API keys.
- **Developer Snippets**: Includes copy-pasteable snippets for **cURL**, **Python (requests)**, and **Node.js (fetch)**. The code explorer allows developers to dynamically inject their active API key prefixes into code blocks for easy onboarding.

### 6. Live Log Simulator & Source Heuristic Inference

- **Log Stream Simulator**: Generate synthetic production logs at adjustable speeds (from 500ms to 3s) using a toggle switch in the **System Monitoring** dashboard.
- **Source Inference Heuristics**: Logs classified without a source are run through a heuristic pattern engine that assigns specific service names (like `DatabasePool`, `SecurityMonitor`, `AuthService`, `PaymentGateway`, etc.) based on log content keywords.

---

## 📂 Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── api/             # API routes (FastAPI)
│   │   ├── models/          # Database & SQLModel schema definitions
│   │   ├── services/        # Classification engines (Regex, BERT, LLM, Classify)
│   │   ├── database.py      # Database initializer and connection session pool
│   │   ├── worker.py        # Celery task queue (Model training & data prep)
│   │   └── main.py          # FastAPI application startup & lifecycle management
│   ├── training/            # Notebooks and dataset generation scripts
│   ├── uploads/             # Directory for temporary CSV uploads
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React Application & UI Layout
│   │   ├── index.css        # Tailwind & Custom styling definitions
│   │   └── main.tsx         # Vite React entry point
│   ├── package.json         # Node package configuration
│   └── vite.config.ts       # Vite configuration
└── docker-compose.yml       # Orchestrates PostgreSQL, Redis, API, and Worker
```

---

## 🛠️ Setup & Running Instructions

### Option 1: Running with Docker Compose (Recommended)

Docker Compose spins up the entire backend stack (PostgreSQL, Redis, FastAPI, Celery worker) automatically.

1. **Create the Environment File**:
   In the root directory, create a `.env` file to configure your Groq API key (used for LLM fallback):

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
2. **Start Backend Services**:
   Run the following in the root directory:

   ```bash
   docker compose up --build
   ```

   This initializes:

   - **PostgreSQL Database** on port `5432`
   - **Redis Cache & Celery Broker** on port `6379`
   - **FastAPI API Server** on port `8000` (Swagger UI: `http://localhost:8000/docs`)
   - **Celery Worker** (handling embeddings and training)
3. **Start the Frontend**:
   Open a new terminal, navigate to the `frontend/` folder, install the packages, and start the development server:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open `http://localhost:5173` in your browser.

---

### Option 2: Running Locally (Manual Setup)

To run services individually without Docker:

1. **Setup Backend**:

   - Ensure you have **PostgreSQL** and **Redis** installed and running on default ports.
   - Navigate to the `backend/` folder:
     ```bash
     cd backend
     ```
   - Create and activate a virtual environment:
     ```bash
     # Windows (PowerShell)
     python -m venv venv
     .\venv\Scripts\Activate.ps1

     # macOS/Linux
     python -m venv venv
     source venv/bin/activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Create a `.env` file in `backend/`:
     ```env
     DATABASE_URL=postgresql://user:password@localhost:5432/logsdb
     CELERY_BROKER_URL=redis://localhost:6379/0
     CELERY_RESULT_BACKEND=redis://localhost:6379/0
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Start the FastAPI application:
     ```bash
     uvicorn app.main:app --reload --port 8000
     ```
   - In a separate terminal with the virtual environment activated, start the Celery worker:
     ```bash
     celery -A app.worker.celery_app worker --loglevel=info
     ```
2. **Setup Frontend**:

   - Navigate to the `frontend/` folder:
     ```bash
     cd ../frontend
     ```
   - Install dependencies and start the Vite dev server:
     ```bash
     npm install
     npm run dev
     ```
   - The frontend will be running at `http://localhost:5173`.

---

## 📖 How to Use

### 1. Ingestion & Analysis (Logs Analytics Tab)

- **Log Ingestion**: Drag and drop any log CSV file. The CSV only requires `source` and `log_message` columns (order does not matter, and common aliases like `message`, `msg`, `system`, or `app` are resolved dynamically).
- **Expandable Detail Accordion**: Click on any log row to slide open a detail pane showing the full, un-truncated message, creation timestamp, raw confidence probability, and human correction label select box.
- **Search & Filters**: Instantly query logs by message or source, or filter them by label or classification method.

### 2. Custom Overrides (Model Training Tab)

- **Add Regex Override**: Enter a regex pattern (e.g. `Inference completed.*`) and a target label (e.g. `Success`).
- **Pipeline Priority**: If a log matches the regex rule, it will instantly bypass ML/LLM models, classifying with `100.0%` confidence under the `Regex` method.

### 3. Model Training & Pseudo-Labeling

- **Supervised Training**: Upload a labeled CSV (containing `log_message` and `target_label`) to train the model.
- **Pseudo-Labeled (Raw) Training**: Upload a raw log CSV (containing only `log_message` and `source`). The system automatically uses the current active pipeline to label the logs and fits the model.
- **Database Retraining**: Click the **"Retrain Model on Database Logs"** button under the Model Training tab to fit a new Logistic Regression classifier directly on logs collected in the database.
- **Version Switcher**: View accuracy, precision, and recall metrics in the training report and click **Activate** on any history entry to instantly roll back or promote a model version.

### 4. API Integration & Key Authentication

For direct program integrations, query the production classification endpoint:

- **Endpoint**: `POST http://localhost:8000/api/logs/classify`
- **Headers**:
  - `Content-Type: application/json`
  - `X-API-Key: <YOUR_API_KEY>` (Optional. If provided, validates the key and increments key invocation metrics. If missing, allows unauthenticated local requests to succeed).
- **Request Format (Single Log)**:
  ```json
  {
    "log_message": "IP 192.168.133.114 blocked due to potential attack",
    "source": "ModernCRM"
  }
  ```
- **Request Format (Batch logs)**:
  ```json
  [
    { "log_message": "User 12345 logged in.", "source": "BillingSystem" },
    { "log_message": "Backup completed successfully.", "source": "AnalyticsEngine" }
  ]
  ```
- **Response Format**:
  ```json
  {
    "log_message": "IP 192.168.133.114 blocked due to potential attack",
    "source": "ModernCRM",
    "target_label": "Security Alert",
    "classification_method": "ML",
    "confidence": 0.985
  }
  ```

---

## 🎯 What You Can Do with This Project (Use Cases)

1. **Real-Time Security & Intrusion Detection (SIEM)**:

   - Run active traffic log streams through the `/api/logs/classify` production API.
   - Instantly catch attacks (like unauthorized data access, brute-forcing, or admin privilege escalations) and generate security alerts.
2. **Automated Incident Response & Alerting**:

   - Classify logs as `Critical Error` or `Workflow Error` in real-time.
   - Hook the classification outputs to Slack, PagerDuty, or Webhooks to notify engineering teams the second an anomaly is detected.
3. **Log Noise Filtering & Cost Optimization**:

   - Apply dynamic custom regex rules to identify high-volume, low-severity messages (such as standard HTTP status codes, backup completes).
   - Filter them out or route them to cold storage to optimize database storage costs.
4. **Automated Knowledge Distillation**:

   - Feed raw, unlabeled log streams into the training console.
   - The platform will auto-label the dataset using the pipeline (Regex + ML + LLM) and fit a compact, high-speed ML classifier for local deployments.
5. **Operational Monitoring & Analytics**:

   - Visualize system health, pipeline classification ratios, and label distribution curves to quickly detect anomalies or service degradation.
