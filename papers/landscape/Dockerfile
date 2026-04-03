FROM python:3.12-slim

WORKDIR /experiments

COPY experiments/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY experiments/ .

# Re-run the full experiment suite (~4h on 8 cores, downloads from OpenML)
CMD ["python", "run_v3_experiments.py"]
