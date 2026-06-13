# 🛡️ Hybrid Log Classification & Real-Time Analytics Platform

A production-ready, custom-trainable log classification platform. It leverages a **hybrid cascade pipeline (Regex + BERT ML Model + LLM)** to classify log streams with speed and high precision. Additionally, it offers **real-time model training with dynamic pseudo-labeling**, a **custom regex overrides manager**, and a **Recharts-powered system monitoring dashboard**.

---

## 🚀 Key Features

### 1. Hybrid Cascade Classification Pipeline
The backend routes incoming logs through a 3-stage intelligence cascade:
- **Stage 1: Dynamic Regex Rules**: Evaluates high-frequency, predictable log patterns first. Rules are managed directly in the UI and compiled in-memory (with a 5-second TTL cache) to ensure maximum throughput.
- **Stage 2: BERT Embeddings + Logistic Regression**: Complex log patterns are encoded using `SentenceTransformer` (`all-MiniLM-L6-v2`) and classified via a custom-trained Logistic Regression model (equipped with `class_weight='balanced'` to prevent minority class bias).
- **Stage 3: LLM (DeepSeek-R1 via Groq)**: Invoked as an intelligent fallback when training data is absent (e.g., for LegacyCRM logs) or classification certainty is low.

### 2. Real-Time Model Training & Auto-Labeling
- **Dynamic Pseudo-Labeling (Unlabeled Training Fallback)**: If you upload a training CSV missing a `target_label` column, the platform automatically activates the pipeline (Regex + ML + LLM) to label the raw logs on-the-fly and trains the model on these generated predictions.
- **Interactive Training Console**: Watch live training logs stream directly from the Celery worker into the React frontend console emulator.
- **Version Registry & Activation**: Maintain a history of model versions with validation metrics, and swap active classifiers instantly without server restarts.

### 3. System Monitoring & Interactive Analytics
- **Live Activity Feed**: Stream the 5 most recent log classifications in real-time.
- **Visual Recharts Graphs**:
  - **Classification Methods Ratio** (Doughnut chart displaying percentages of Regex vs ML vs LLM processing).
  - **Log Categories Distribution** (Interactive Bar chart displaying frequency count of the top 8 labels).
  - **Pipeline Performance Indicators** (Live progression bars tracking matching efficiency).
- **System Health Checks**: Live status indicators for PostgreSQL, Redis, Celery Workers, and Groq API keys.

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
- **Search & Filters**: Instantly query logs by message or source, or filter them by label or classification method.

### 2. Custom Overrides (Model Training Tab)
- **Add Regex Override**: Enter a regex pattern (e.g. `Inference completed.*`) and a target label (e.g. `Success`).
- **Pipeline Priority**: If a log matches the regex rule, it will instantly bypass ML/LLM models, classifying with `100.0%` confidence under the `Regex` method.

### 3. Model Training & Pseudo-Labeling
- **Supervised Training**: Upload a labeled CSV (containing `log_message` and `target_label`) to train the model.
- **Pseudo-Labeled (Raw) Training**: Upload a raw log CSV (containing only `log_message` and `source`). The system automatically uses the current active pipeline to label the logs and fits the model.
- **Version Switcher**: View accuracy, precision, and recall metrics in the training report and click **Activate** on any history entry to instantly roll back or promote a model version.

### 4. API Integration
For direct program integrations, query the production classification endpoint:
- **Endpoint**: `POST http://localhost:8000/api/logs/classify`
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
