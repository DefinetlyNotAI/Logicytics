FROM python:3.11-slim
WORKDIR /app/CODE
COPY ../requirements.txt .
RUN pip install -r requirements.txt
COPY .. .
CMD ["python", "Logicytics.py", "-h"]
