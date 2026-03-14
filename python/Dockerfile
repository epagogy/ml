FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pytest pytest-timeout numpy pandas scikit-learn scipy skops matplotlib
COPY ml/ ml/
COPY tests/ tests/
COPY pyproject.toml .
CMD ["python3", "-m", "pytest", "tests/", "--timeout=120", "-q", "--ignore=tests/test_harvest.py"]
