# Start with a minimal Ubuntu base image
FROM ubuntu:22.04

# Add the UV binary to the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables to avoid interaction during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install essential utilities
RUN apt-get update && \
apt-get install -y --no-install-recommends \
ca-certificates \
curl \
build-essential \
tini && \
rm -rf /var/lib/apt/lists/*

# Add a non-root user with a home directory
RUN useradd -m -s /bin/bash user

# Switch to the non-root user
USER user

# Set tini as the init system to handle zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]

WORKDIR /app

COPY pyproject.toml pyproject.toml
RUN uv sync

# activate .venv automatically
RUN echo "source .venv/bin/activate" >> ~/.bashrc

# Run bash by default
CMD ["bash"]