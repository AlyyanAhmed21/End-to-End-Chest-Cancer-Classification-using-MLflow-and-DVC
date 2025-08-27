FROM python:3.8-slim
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the essential application source code
COPY app.py .
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY config ./config
COPY params.yaml .
# ... copy any other essential source files

# --- CRITICAL CHANGE ---
# Create a 'model' directory inside the container and
# copy ONLY our LFS-tracked model file into it.
RUN mkdir model
COPY artifacts/training/best_model.h5 ./model/

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]