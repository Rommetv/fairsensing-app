# FastAPI + geospatial stack
FROM python:3.11-slim

# System deps for geopandas / rasterio / shapely / PROJ / GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin libgdal-dev \
    proj-bin libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# install Python deps first (better layer caching)
COPY pipfair-requirements.txt /app/
RUN pip install --no-cache-dir -r pipfair-requirements.txt

# copy the rest of your code
COPY . /app

# run as non-root
RUN useradd -m appuser
USER appuser

# Sliplane will hit this port
EXPOSE 8000
ENV PORT=8000

# Adjust `app.main:app` to your FastAPI location if different
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
