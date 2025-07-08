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

If you do not have these files, the API will not start. You can train your model and export these files, or obtain them from [class_mapping.json](https://cdn.bridgelabs.tech/ghana-ai-hackathon/food-classifier/) and [best_model.pth](https://cdn.bridgelabs.tech/ghana-ai-hackathon/food-classifier/best_model.pth). You can also download by running the `update_model.sh` to get the latest files into the right directory.

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

### 6. Caching System

This API uses an in-memory caching system to speed up repeated predictions and reduce computation:

- **PredictionCache**: Stores up to 100 recent predictions. Each cache entry is keyed by a hash of the image bytes and the input text. If a request with the same image and text is received, the cached result is returned instantly, reducing latency and server load.
- **Text Preprocessing LRU Cache**: Uses Python's `functools.lru_cache` to cache up to 1000 recent text preprocessing results (tokenization). This speeds up repeated predictions with the same text input.

**How it works:**

- When a prediction request is received, the API computes a hash key from the image and text.
- If the key exists in the cache, the cached prediction is returned (with `cache_hit: true`).
- If not, the model runs inference, the result is cached, and then returned (with `cache_hit: false`).
- The cache automatically evicts the oldest entries when full.

**Note:** This cache is in-memory and will be cleared if the server restarts. For production, consider using a distributed cache (e.g., Redis) for scalability and persistence.

---

### 7. Production Setup Notes

For deploying this API in production, consider the following best practices:

- **Use a Production WSGI/ASGI Server:**
  - Instead of running with `uvicorn --reload`, use a process manager like [Gunicorn](https://gunicorn.org/) with Uvicorn workers:
    ```sh
    gunicorn -k uvicorn.workers.UvicornWorker main:app --workers 4 --bind 0.0.0.0:8000
    ```
- **Environment Variables:**
  - Store secrets and configuration (e.g., model paths, device selection) in environment variables or a `.env` file, not hardcoded in code.
- **Security:**
  - Use HTTPS in production (behind a reverse proxy like Nginx or Caddy).
  - Set up authentication and rate limiting if exposing the API publicly.
  - Validate and sanitize all inputs.
- **Scaling:**
  - For high traffic, run multiple worker processes and/or deploy behind a load balancer.
  - Use a distributed cache (e.g., Redis) for prediction caching if running multiple instances.
- **Monitoring:**
  - Integrate logging, error tracking, and health checks.
  - Use the `/system_status` and `/health` endpoints for basic monitoring.
- **Model Reloading:**
  - Use the `/reload_model` endpoint to reload model weights and class mappings without restarting the server after updates.

For more details, see the [FastAPI deployment documentation](https://fastapi.tiangolo.com/deployment/).

---
