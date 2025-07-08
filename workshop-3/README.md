## Workshop 3: Deploying a Custom Model With FastAPI

### 1. Environment Setup

```sh
# Create a virtual environment
python3 -m venv env

# Activate the environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Files

Before running the API, make sure you have the following files in the `workshop-3/` directory:

- `checkpoints/best_model.pth` — The trained model weights.
- `class_mapping.json` — A JSON file mapping class names to indices (e.g., `{"pizza": 0, "burger": 1, ...}`).

If you do not have these files, the API will not start. You can train your model and export these files, or obtain them from [class_mapping.json]() and [best_model.pth]().

### 3. Running the API

```sh
# From the workshop-3 directory (with env activated)
uvicorn main:app --reload --workers 4 --timeout-keep-alive 30
```

The API will be available at `http://localhost:8000`.

### 4. Testing the API

#### Predict Endpoint

```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample/pexels-valeriya-1639557.jpg" \
  -F "text=Image of pizza"
```

#### Health Check

```sh
curl http://localhost:8000/health
```

#### List Classes

```sh
curl http://localhost:8000/classes
```

#### System Status

```sh
curl http://localhost:8000/system_status
```

#### Reload Model (after updating weights or class mapping)

```sh
curl -X POST http://localhost:8000/reload_model
```

### 5. Troubleshooting

- **Missing `best_model.pth` or `class_mapping.json`:** Ensure these files are present in the correct locations. The API will fail to start if they are missing.
- **CUDA Errors:** If you do not have a GPU, the code will automatically use CPU.
- **Dependency Issues:** Double-check you are using the provided `requirements.txt` and a clean virtual environment.

---

For more details, watch [Workshop 2 Recording]()
