# Base: Ubuntu 22.04 — no ROS dependency
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Core tools ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget git tmux nano \
    python3-pip python3-dev python3-venv \
    cmake build-essential pkg-config \
    lsb-release gnupg \
    && rm -rf /var/lib/apt/lists/*

# ── Gazebo Garden + dev headers ──────────────────────────────
RUN wget https://packages.osrfoundation.org/gazebo.gpg \
        -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
          signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
          http://packages.osrfoundation.org/gazebo/ubuntu-stable \
          $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y \
    gz-garden \
    libgz-sim7-dev \
    libgz-transport12-dev \
    libgz-msgs9-dev \
    libgz-sensors7-dev \
    libgz-math7-dev \
    && rm -rf /var/lib/apt/lists/*

# ── pybind11 (for gz_transport_py build) ─────────────────────
RUN apt-get update && apt-get install -y \
    python3-pybind11 pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ───────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r /tmp/requirements.txt

# ── GPU / display ────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV GZ_VERSION=garden

# ── Shell aliases ────────────────────────────────────────────
RUN { \
      echo 'alias train="bash /workspace/scripts/run.sh"'; \
      echo "alias unfollow=\"gz service -s /gui/follow --reqtype gz.msgs.StringMsg --reptype gz.msgs.Boolean --timeout 2000 --req 'data: \\\"\\\"'\""; \
    } >> ~/.bash_aliases

WORKDIR /workspace
