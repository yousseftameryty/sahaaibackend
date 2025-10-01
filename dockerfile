FROM python:3.12-slim

# Install system deps for Ollama (as root)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama as root
RUN curl -fsSL https://ollama.com/install.sh | sh

# User setup (HF req)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy & install
COPY --chown=user . $HOME/app
RUN pip install --no-cache-dir -r requirements.txt

# Preload (build time)
RUN python preload_models.py

# Expose & run
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]