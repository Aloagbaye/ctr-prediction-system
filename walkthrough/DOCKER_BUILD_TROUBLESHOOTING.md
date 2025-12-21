# Docker Build Troubleshooting Guide

## 🐌 Why Docker Builds Take Time

Docker builds can take several minutes, especially the first time. Here's why:

### 1. **Base Image Download**
- First build downloads the base image (`python:3.11-slim`) - can be 50-100MB
- Subsequent builds use cached layers

### 2. **System Dependencies**
- Installing `gcc`, `g++` and other build tools
- Running `apt-get update` (downloads package lists)

### 3. **Python Dependencies**
- Installing packages from `requirements.txt`
- Some packages compile from source (XGBoost, LightGBM, NumPy, etc.)
- Large packages like TensorFlow, PyTorch can be 100-500MB each

### 4. **Network Speed**
- Download speed affects build time
- Slower connections = longer builds

## ⏱️ Expected Build Times

**First Build:**
- Base image download: 1-3 minutes
- System dependencies: 1-2 minutes
- Python dependencies: 5-15 minutes (depending on packages)
- **Total: 7-20 minutes**

**Subsequent Builds (with cache):**
- If only code changed: 30 seconds - 2 minutes
- If requirements changed: 3-10 minutes

## 🔍 How to Check Build Progress

### Option 1: Use Verbose Output

```powershell
# Build with progress output
docker build -f Dockerfile.api -t my-image:latest . --progress=plain

# Or with no cache to see all steps
docker build -f Dockerfile.api -t my-image:latest . --no-cache --progress=plain
```

### Option 2: Check Docker Logs

```powershell
# In another terminal, check Docker Desktop logs
# Or use:
docker system events
```

### Option 3: Build Step by Step

Modify the Dockerfile temporarily to see where it's stuck:

```dockerfile
# Add this after each RUN command to see progress
RUN echo "Step completed: Installing system dependencies"
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
RUN echo "Step completed: System dependencies installed"

RUN echo "Step completed: Installing Python dependencies"
RUN pip install --no-cache-dir --user -r requirements.txt
RUN echo "Step completed: Python dependencies installed"
```

## 🚨 Common Issues

### Issue 1: Build Appears Stuck

**Symptoms:**
- No output for several minutes
- Build seems frozen

**Causes:**
- Installing large packages (XGBoost, LightGBM, TensorFlow)
- Compiling packages from source
- Network issues

**Solutions:**
1. **Wait longer** - Some packages take 5-10 minutes to compile
2. **Check network** - Ensure stable internet connection
3. **Use verbose mode** - See what's happening:
   ```powershell
   docker build -f Dockerfile.api -t my-image:latest . --progress=plain
   ```
4. **Check Docker Desktop** - Look at resource usage (CPU/Memory)

### Issue 2: Build Fails During Package Installation

**Symptoms:**
- Error messages about missing dependencies
- Compilation errors

**Solutions:**
1. **Ensure system dependencies are installed:**
   ```dockerfile
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       make \
       && rm -rf /var/lib/apt/lists/*
   ```

2. **Install packages one at a time** to identify problematic package:
   ```dockerfile
   RUN pip install --no-cache-dir --user pandas numpy
   RUN pip install --no-cache-dir --user scikit-learn
   RUN pip install --no-cache-dir --user xgboost
   # etc.
   ```

### Issue 3: Out of Memory

**Symptoms:**
- Build fails with memory errors
- Docker Desktop shows high memory usage

**Solutions:**
1. **Increase Docker memory limit:**
   - Docker Desktop → Settings → Resources → Memory
   - Increase to at least 4GB (8GB recommended)

2. **Build with less parallelism:**
   ```powershell
   docker build -f Dockerfile.api -t my-image:latest . --build-arg BUILDKIT_INLINE_CACHE=1
   ```

### Issue 4: Slow Network Downloads

**Symptoms:**
- Very slow package downloads
- Timeouts

**Solutions:**
1. **Use pip mirrors:**
   ```dockerfile
   RUN pip install --no-cache-dir --user -r requirements.txt \
       -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **Use Docker buildkit cache:**
   ```powershell
   $env:DOCKER_BUILDKIT=1
   docker build -f Dockerfile.api -t my-image:latest .
   ```

## 🛠️ Optimization Tips

### 1. Use Multi-Stage Builds (Already Implemented)

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
# Install dependencies

# Runtime stage
FROM python:3.11-slim
# Copy only what's needed
```

### 2. Leverage Docker Cache

```dockerfile
# Copy requirements first (changes less frequently)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy code last (changes more frequently)
COPY src/ ./src/
```

### 3. Use .dockerignore

Ensure `.dockerignore` excludes unnecessary files:
```
__pycache__
*.pyc
.git
.env
*.md
notebooks/
data/
```

### 4. Build Specific Stages

```powershell
# Build only the builder stage
docker build -f Dockerfile.api -t my-image:builder --target builder .
```

## 📊 Monitoring Build Progress

### Check What Docker is Doing

```powershell
# List running containers (builds run in containers)
docker ps

# Check Docker stats
docker stats

# View build cache
docker system df
```

### Use BuildKit for Better Output

```powershell
# Enable BuildKit
$env:DOCKER_BUILDKIT=1
$env:BUILDKIT_PROGRESS=plain

# Build with progress
docker build -f Dockerfile.api -t my-image:latest .
```

## ⚡ Quick Fixes

### If Build is Taking Too Long

1. **Cancel and retry:**
   ```powershell
   # Press Ctrl+C to cancel
   # Then rebuild
   docker build -f Dockerfile.api -t my-image:latest .
   ```

2. **Clear cache and rebuild:**
   ```powershell
   docker builder prune
   docker build -f Dockerfile.api -t my-image:latest . --no-cache
   ```

3. **Check if it's actually working:**
   ```powershell
   # In another terminal
   docker ps
   docker stats
   ```

## 🎯 Expected Output During Build

Normal build output should show:
```
[+] Building X.Xs (Y/Y) FINISHED
 => [internal] load build definition from Dockerfile.api
 => [internal] load .dockerignore
 => [internal] load metadata for docker.io/library/python:3.11-slim
 => [builder 1/6] FROM docker.io/library/python:3.11-slim
 => [internal] load build context
 => [builder 2/6] RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
 => [builder 3/6] COPY requirements.txt .
 => [builder 4/6] RUN pip install --no-cache-dir --user -r requirements.txt
 => [stage-1 5/6] COPY src/ ./src/
 => [stage-1 6/6] CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

If you see this, the build is progressing normally!

## 💡 Pro Tips

1. **First build always takes longest** - Subsequent builds are faster due to caching

2. **Large packages compile from source** - XGBoost, LightGBM can take 5-10 minutes each

3. **Check Docker Desktop** - Resource usage shows if Docker is working

4. **Use `--progress=plain`** - See detailed output of what's happening

5. **Be patient** - First build can take 10-20 minutes depending on your system

---

**Remember:** Docker builds aren't stuck, they're just working! 🐳

