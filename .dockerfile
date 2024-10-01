FROM python:3.11-slim
WORKDIR /app/CODE
COPY ../requirements.txt .
RUN pip install -r requirements.txt
COPY .. .
CMD ["python", "Logicytics.py", "-h"]
# Need someone to run `docker build -t logicytics .` as my wsl is corrupted and microsoft ain't helping :( Please
