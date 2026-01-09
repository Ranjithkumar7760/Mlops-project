# SMS Spam Detection - End-to-End AI/ML + DevOps Project

## 📋 Project Overview
This is a production-ready AI-based Text Classification system for SMS Spam Detection, deployable on Kubernetes using Docker and CI/CD.

## 🚀 Running Locally (Without Docker)

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step-by-Step Instructions

#### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Train the Machine Learning Model
```bash
python training/train.py
```

This will:
- Load the dataset from `data/spam.csv`
- Train a Multinomial Naive Bayes model
- Save the trained model to `model/model.pkl`
- Save the vectorizer to `model/vectorizer.pkl`
- Display accuracy and classification report

**Expected Output:**
```
Loading dataset...
Encoding labels...
Splitting data into train and test sets...
Applying TF-IDF vectorization...
Training Multinomial Naive Bayes model...
Evaluating model...

==================================================
Model Accuracy: 0.XXXX
==================================================

Classification Report:
...
Model saved to: model/model.pkl
Vectorizer saved to: model/vectorizer.pkl

Training completed successfully!
```

#### Step 3: Start the FastAPI Application
```bash
python app/main.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
Loading model and vectorizer...
Model and vectorizer loaded successfully!
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 4: Test the API

**Option A: Using Browser**
- Open: http://localhost:8000/health
- Open: http://localhost:8000/docs (Interactive API documentation)

**Option B: Using curl (Command Line)**

1. **Health Check:**
```bash
curl http://localhost:8000/health
```

2. **Test Prediction (Spam):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Congratulations! You've won a $1000 gift card. Click here to claim.\"}"
```

3. **Test Prediction (Not Spam):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"I'm going to be late, sorry\"}"
```

**Option C: Using Python requests**
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Congratulations! You've won a $1000 gift card."}
)
print(response.json())
```

**Expected Response:**
```json
{
  "prediction": "Spam",
  "confidence": 0.95
}
```

## 📁 Project Structure
```
text-classifier/
├── app/
│   └── main.py              # FastAPI application
├── training/
│   └── train.py             # ML model training script
├── model/
│   ├── model.pkl            # Trained model (generated)
│   └── vectorizer.pkl       # TF-IDF vectorizer (generated)
├── data/
│   └── spam.csv             # SMS spam dataset
├── k8s/
│   ├── deployment.yaml      # Kubernetes deployment
│   └── service.yaml         # Kubernetes service
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── .github/workflows/
    └── deploy.yml           # CI/CD pipeline
```

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError for model files
**Solution:** Run the training script first:
```bash
python training/train.py
```

### Issue: Port 8000 already in use
**Solution:** Use a different port:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Issue: Dataset not found
**Solution:** Ensure `data/spam.csv` exists with columns: `label` and `message`

## 🐳 Docker Deployment (Optional)
See Dockerfile for containerization instructions.

## ☸️ Kubernetes Deployment (Optional)
See `k8s/` directory for Kubernetes manifests.

## 🔄 CI/CD Pipeline (Optional)
See `.github/workflows/deploy.yml` for GitHub Actions workflow.

## ☁️ Cloud Deployment (Kubernetes + CI/CD)

### Quick Start
See **[QUICK_START_DEPLOY.md](QUICK_START_DEPLOY.md)** for a 5-minute deployment guide.

### Detailed Guide
See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for comprehensive cloud deployment instructions.

### Prerequisites
- Docker Hub account
- Kubernetes cluster (AWS EKS, Google GKE, or Azure AKS)
- GitHub repository with secrets configured

### Deployment Steps

1. **Push Docker Image:**
   ```bash
   docker build -t YOUR_USERNAME/text-classifier:latest .
   docker push YOUR_USERNAME/text-classifier:latest
   ```

2. **Configure GitHub Secrets:**
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password
   - `KUBE_CONFIG`: Content of `~/.kube/config`

3. **Deploy:**
   - Push to `main` branch (triggers CI/CD)
   - Or use manual script: `./scripts/deploy.sh spam-detection YOUR_USERNAME`

4. **Access API:**
   ```bash
   kubectl get service text-classifier-service -n spam-detection
   # Use EXTERNAL-IP to access: http://EXTERNAL-IP/docs
   ```

### Kubernetes Manifests
- `k8s/deployment.yaml` - Application deployment
- `k8s/service.yaml` - LoadBalancer service
- `k8s/namespace.yaml` - Namespace definition
- `k8s/ingress.yaml` - Ingress for custom domain (optional)

