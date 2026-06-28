# Self-contained sandbox / Stage-3 reproduction image.
#   docker build -t redrob-ranker .
#   docker run --rm -v "$PWD:/data" redrob-ranker \
#       python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the ranker on a mounted candidates file. Override the command to
# launch the Streamlit sandbox instead:  streamlit run app.py --server.port 7860
CMD ["python", "rank.py", "--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
