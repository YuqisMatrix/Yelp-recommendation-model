# ---- base ----
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    build-essential \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=${JAVA_HOME}" >> /etc/environment

ENV PATH="${JAVA_HOME}/bin:${PATH}"


WORKDIR /app

# ---- dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy project files ----
COPY app.py /app/
COPY templates/ /app/templates/
COPY recommendation/ /app/recommendation/
COPY model/ /app/model/
COPY data/ /app/data/

# ---- environment variables ----
ENV MODEL_PATH=/app/model/yelp_model.pkl \
    CF_ARTIFACTS_DIR=/app/model/recommender_business \
    FLASK_APP=app.py \
    PYSPARK_PYTHON=python3

EXPOSE 5000


RUN mkdir -p /app/logs

# ---- healthcheck ----
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]

