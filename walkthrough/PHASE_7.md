# Phase 7: Containerization

## 📋 Overview

In Phase 7, we'll containerize the CTR Prediction System using Docker. This enables easy deployment, consistent environments, and scalability.

**Goals:**
- Create Dockerfiles for API and Streamlit services
- Set up docker-compose for orchestration
- Build and test Docker images
- Understand containerization best practices

**Deliverable:** Containerized application ready for deployment

---

## 🎯 Learning Objectives

By the end of this phase, you'll understand:
- How to create Dockerfiles for Python applications
- Multi-stage builds for optimization
- Docker Compose for multi-container applications
- Volume mounting for data persistence
- Health checks and container orchestration

---

## 📦 Prerequisites

Before starting Phase 7, ensure you have:
- ✅ Completed Phase 6 (API is working)
- ✅ Docker installed and running
- ✅ Docker Compose installed (usually comes with Docker Desktop)
- ✅ Trained models in `models/` directory

**Check Docker Installation:**
```bash
docker --version
docker-compose --version
```

---

## 🚀 Step-by-Step Guide

### Step 1: Understand the Docker Setup

We'll create:
1. **Dockerfile.api** - For the FastAPI service
2. **Dockerfile.streamlit** - For the Streamlit UI
3. **docker-compose.yml** - Orchestrates both services
4. **.dockerignore** - Excludes unnecessary files

### Step 2: Build Docker Images

**Option A: Using Build Scripts**

```bash
# Linux/Mac
chmod +x scripts/docker_build.sh
./scripts/docker_build.sh

# Windows PowerShell
.\scripts\docker_build.ps1
```

**Option B: Manual Build**

```bash
# Build API image
docker build -f Dockerfile.api -t ctr-prediction-api:latest .

# Build Streamlit image
docker build -f Dockerfile.streamlit -t ctr-prediction-streamlit:latest .
```

**Option C: Using Docker Compose**

```bash
docker-compose build
```

### Step 3: Run Containers

**Option A: Using Docker Compose (Recommended)**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Option B: Manual Run**

```bash
# Run API container
docker run -d \
  --name ctr-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  ctr-prediction-api:latest

# Run Streamlit container
docker run -d \
  --name ctr-streamlit \
  -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  ctr-prediction-streamlit:latest
```

### Step 4: Verify Services

**Check API:**
```bash
# Health check
curl http://localhost:8000/health

# Or open in browser
open http://localhost:8000/docs
```

**Check Streamlit:**
```bash
# Open in browser
open http://localhost:8501
```

### Step 5: View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit

# Last 100 lines
docker-compose logs --tail=100
```

### Step 6: Stop and Clean Up

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove images
docker rmi ctr-prediction-api:latest
docker rmi ctr-prediction-streamlit:latest
```

---

## 🔍 Understanding the Dockerfiles

### Dockerfile.api

**Multi-stage Build:**
- **Stage 1 (builder)**: Installs dependencies
- **Stage 2 (final)**: Copies only necessary files

**Key Features:**
- Python 3.11 slim base image
- Installs system dependencies (gcc, g++)
- Copies requirements and installs Python packages
- Copies application code
- Exposes port 8000
- Health check included
- Runs uvicorn server

**Optimizations:**
- Multi-stage build reduces final image size
- Requirements copied first for better caching
- No cache for pip installs

### Dockerfile.streamlit

**Similar Structure:**
- Multi-stage build
- Python 3.11 slim
- Installs Streamlit dependencies
- Exposes port 8501
- Runs Streamlit server

### docker-compose.yml

**Services:**
- **api**: FastAPI service
- **streamlit**: Streamlit UI

**Features:**
- Port mapping
- Volume mounting (models, data)
- Environment variables
- Health checks
- Network configuration
- Dependency management

---

## 📊 Docker Commands Reference

### Building

```bash
# Build specific image
docker build -f Dockerfile.api -t ctr-api:latest .

# Build with no cache
docker build --no-cache -f Dockerfile.api -t ctr-api:latest .

# Build with build args
docker build --build-arg PYTHON_VERSION=3.11 -f Dockerfile.api -t ctr-api:latest .
```

### Running

```bash
# Run container
docker run -d -p 8000:8000 ctr-prediction-api:latest

# Run with volume
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models ctr-prediction-api:latest

# Run with environment variables
docker run -d -p 8000:8000 -e API_KEY=secret ctr-prediction-api:latest

# Run interactively
docker run -it ctr-prediction-api:latest /bin/bash
```

### Managing Containers

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Start container
docker start <container_id>

# Remove container
docker rm <container_id>

# View logs
docker logs <container_id>
docker logs -f <container_id>  # Follow logs
```

### Docker Compose

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs -f

# Execute command in service
docker-compose exec api python -c "print('Hello')"

# Scale services
docker-compose up --scale api=3
```

---

## 🔧 Configuration Options

### Environment Variables

Create `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501
API_URL=http://api:8000

# Model Configuration
MODELS_DIR=/app/models
DEFAULT_MODEL=xgboost
```

Use in docker-compose.yml:

```yaml
environment:
  - API_URL=${API_URL}
  - DEFAULT_MODEL=${DEFAULT_MODEL}
```

### Volume Mounting

**Models Directory:**
```yaml
volumes:
  - ./models:/app/models:ro  # Read-only
```

**Data Directory:**
```yaml
volumes:
  - ./data:/app/data:ro
```

**Logs Directory:**
```yaml
volumes:
  - ./logs:/app/logs
```

### Port Configuration

Change ports in docker-compose.yml:

```yaml
ports:
  - "8080:8000"  # Host:Container
```

---

## 🧪 Testing the Containers

### Test API Container

```bash
# Build and run
docker build -f Dockerfile.api -t test-api .
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models test-api

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict_ctr \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","ad_id":"test","device":"mobile","placement":"header"}'
```

### Test Streamlit Container

```bash
# Build and run
docker build -f Dockerfile.streamlit -t test-streamlit .
docker run -d -p 8501:8501 -e API_URL=http://host.docker.internal:8000 test-streamlit

# Open browser
open http://localhost:8501
```

### Test Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Test API
curl http://localhost:8000/health

# Test Streamlit
curl http://localhost:8501/_stcore/health

# View logs
docker-compose logs -f
```

---

## 🔍 Common Issues & Solutions

### Issue 1: Models Not Found

**Problem**: Container can't find model files

**Solution:**
```bash
# Ensure models directory exists and is mounted
docker run -v $(pwd)/models:/app/models ctr-prediction-api:latest

# Check volume mount
docker inspect <container_id> | grep Mounts
```

### Issue 2: Port Already in Use

**Problem**: Port 8000 or 8501 already in use

**Solution:**
```bash
# Change ports in docker-compose.yml
ports:
  - "8080:8000"  # Use different host port

# Or stop existing service
docker stop $(docker ps -q --filter "publish=8000")
```

### Issue 3: Permission Denied

**Problem**: Can't write to mounted volumes

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER models/

# Or run with user flag
docker run -u $(id -u):$(id -g) ...
```

### Issue 4: Build Fails

**Problem**: Docker build fails with dependency errors

**Solution:**
```bash
# Clear Docker cache
docker builder prune

# Rebuild without cache
docker build --no-cache -f Dockerfile.api -t ctr-api:latest .
```

### Issue 5: Containers Can't Communicate

**Problem**: Streamlit can't reach API

**Solution:**
```yaml
# Use service name in docker-compose
environment:
  - API_URL=http://api:8000  # Not localhost!
```

---

## 📊 Expected Results

After completing Phase 7, you should have:

1. **Docker Images:**
   - `ctr-prediction-api:latest`
   - `ctr-prediction-streamlit:latest`

2. **Running Containers:**
   - API accessible at `http://localhost:8000`
   - Streamlit accessible at `http://localhost:8501`

3. **Docker Compose Setup:**
   - Both services running
   - Services can communicate
   - Volumes mounted correctly

---

## ✅ Phase 7 Checklist

- [ ] Docker installed and running
- [ ] Dockerfiles created (API and Streamlit)
- [ ] docker-compose.yml created
- [ ] .dockerignore created
- [ ] Images build successfully
- [ ] Containers run successfully
- [ ] API accessible in container
- [ ] Streamlit accessible in container
- [ ] Services can communicate
- [ ] Health checks working
- [ ] Ready for Phase 8 (Cloud Deployment)

---

## 🚀 Next Steps

Once you've completed Phase 7, you're ready for:

**Phase 8: Cloud Deployment**
- Deploy to GCP Cloud Run
- Deploy to AWS ECS/Fargate
- Configure auto-scaling
- Set up monitoring

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Dockerfile Reference](https://docs.docker.com/reference/dockerfile/)

---

## 💡 Tips for Success

1. **Test Locally First**: Build and test containers locally before deploying
2. **Use Multi-stage Builds**: Reduces image size significantly
3. **Leverage Caching**: Order Dockerfile commands by change frequency
4. **Use .dockerignore**: Exclude unnecessary files
5. **Health Checks**: Always include health checks for production
6. **Volume Mounting**: Mount models/data as volumes, not copy
7. **Environment Variables**: Use env vars for configuration
8. **Logs**: Check logs regularly for debugging

---

## 🎓 Key Concepts Learned

### 1. Containerization Benefits

- **Consistency**: Same environment everywhere
- **Isolation**: No conflicts between applications
- **Portability**: Run anywhere Docker runs
- **Scalability**: Easy to scale horizontally

### 2. Multi-stage Builds

- **Smaller Images**: Only include runtime dependencies
- **Faster Builds**: Better layer caching
- **Security**: Fewer attack surfaces

### 3. Docker Compose

- **Orchestration**: Manage multiple containers
- **Networking**: Automatic service discovery
- **Volumes**: Shared data between containers
- **Dependencies**: Start services in order

---

**Congratulations on completing Phase 7!** 🎉

You now have a containerized application ready for cloud deployment!


