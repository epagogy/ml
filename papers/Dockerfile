FROM python:3.12-slim

WORKDIR /papers

RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY landscape/requirements.txt /tmp/landscape-req.txt
COPY grammar/requirements.txt /tmp/grammar-req.txt
RUN pip install --no-cache-dir -r /tmp/landscape-req.txt -r /tmp/grammar-req.txt

COPY . .

CMD ["sh", "-c", "\
  echo '=== 1. Landscape: compile claims ===' && \
  cd /papers/landscape && python compile_claims.py && \
  echo '' && \
  echo '=== 2. Landscape: generate figures ===' && \
  cd /papers/landscape && python figures.py && \
  echo '' && \
  echo '=== 3. Grammar: compile claims ===' && \
  cd /papers/grammar && python compile_claims.py && \
  echo '' && \
  echo 'All verification steps passed.' \
"]
