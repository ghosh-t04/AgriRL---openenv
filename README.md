# 🎽 Agriculture Environment Server

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

---

## 🚀 Quick Start

Use the `AgricultureEnv` class:

```python
from agriculture import AgricultureAction, AgricultureEnv

try:
    # Create environment from Docker image
    agricultureenv = AgricultureEnv.from_docker_image("agriculture-env:latest")

    # Reset
    result = agricultureenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = agricultureenv.step(AgricultureAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    agricultureenv.close()
```

The `AgricultureEnv.from_docker_image()` method handles:
- Starting the Docker container  
- Waiting for the server to be ready  
- Connecting to the environment  
- Cleaning up when you call `close()`  

---

## 🐳 Building the Docker Image

```bash
# From project root
docker build -t agriculture-env:latest -f server/Dockerfile .
```

---

## 🌐 Deploying to Hugging Face Spaces

Deploy with:

```bash
openenv push
```

Or specify options:

```bash
openenv push --namespace my-org --private
```

### Options
- `--directory`, `-d`: Directory containing the environment (default: current directory)  
- `--repo-id`, `-r`: Repository ID in format `username/repo-name`  
- `--base-image`, `-b`: Override Dockerfile base image  
- `--private`: Deploy as private (default: public)  

### Examples
```bash
openenv push
openenv push --repo-id my-org/my-env
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest
openenv push --private
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:  
`https://huggingface.co/spaces/<repo-id>`

Includes:
- **Web Interface** at `/web`  
- **API Docs** at `/docs`  
- **Health Check** at `/health`  
- **WebSocket** at `/ws`  

---

## 📦 Environment Details

### Action
**AgricultureAction**  
- `message` (str): The message to echo back  

### Observation
**AgricultureObservation**  
- `echoed_message` (str)  
- `message_length` (int)  
- `reward` (float)  
- `done` (bool, always False)  
- `metadata` (dict)  

### Reward
`reward = message_length × 0.1`  
- `"Hi"` → 0.2  
- `"Hello, World!"` → 1.3  
- `""` → 0.0  

---

## ⚙️ Advanced Usage

### Connect to Existing Server
```python
from agriculture import AgricultureEnv, AgricultureAction

agricultureenv = AgricultureEnv(base_url="http://localhost:8000")
result = agricultureenv.reset()
result = agricultureenv.step(AgricultureAction(message="Hello!"))
```

### Context Manager
```python
with AgricultureEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    for msg in ["Hello", "World", "!"]:
        result = env.step(AgricultureAction(message=msg))
        print(result.observation.echoed_message)
```

### Concurrent WebSocket Sessions
```python
from agriculture import AgricultureAction, AgricultureEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with AgricultureEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(AgricultureAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

---

## 🧪 Development & Testing

### Direct Environment Testing
```bash
python3 server/agriculture_environment.py
```

Verifies:
- Environment resets correctly  
- Step executes actions properly  
- State tracking works  
- Rewards are calculated correctly  

### Run Locally
```bash
uvicorn server.app:app --reload
```

---

## 📂 Project Structure

```
agriculture/
├── .dockerignore
├── __init__.py
├── README.md
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── client.py
├── models.py
└── server/
    ├── __init__.py
    ├── agriculture_environment.py
    ├── app.py
    └── Dockerfile
```

---

