FROM python:3.12-slim

WORKDIR /paper

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY compile_claims.py claims.py ./

# Landscape claims.json must be mounted: docker run -v ../landscape/claims.json:/paper/landscape_claims.json
CMD ["python", "compile_claims.py"]
