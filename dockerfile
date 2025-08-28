# 1. Use a standard Python base image
FROM python:3.8-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy ONLY the requirements file to leverage caching
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Selectively copy ONLY the files needed for the APP to RUN.
# We do NOT copy the whole project.
COPY app.py .
COPY static ./static
COPY templates ./templates
COPY src/ . 
# We need the model, which should be tracked by Git LFS.
# Create a model directory in the container and copy the model into it.
RUN mkdir model
COPY artifacts/training/best_model.h5 ./model/

# 6. Expose the port the app runs on
EXPOSE 7860

# 7. The command to start the production server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]