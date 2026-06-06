FROM python:3.11-slim

WORKDIR /app

# Install the shopping list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Put your code and your picture into the box
COPY week2_practice.py .
COPY real_drone_map.jpg .

# Tell the box to run your code when it turns on
CMD ["python", "week2_practice.py"]