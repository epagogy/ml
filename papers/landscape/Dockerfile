FROM python:3.12-slim

WORKDIR /paper

RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY compile_claims.py figures.py claims.py ./
COPY data/ data/

CMD ["sh", "-c", "\
  echo '=== Compile claims ===' && \
  python compile_claims.py && \
  echo '' && \
  echo '=== Generate figures ===' && \
  mkdir -p figures && \
  python figures.py && \
  echo '' && \
  echo 'All steps passed.' \
"]
