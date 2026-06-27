# build deps into a throwaway venv so the final image doesn't carry pip/build cruft
FROM python:3.11-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

FROM python:3.11-slim
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/serving_model \
    PYTHONUNBUFFERED=1
COPY --from=builder /opt/venv /opt/venv
# serving_model is produced by `python -m seoul_bike_mlops.registry` before building
COPY serving_model ./serving_model
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "seoul_bike_mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
