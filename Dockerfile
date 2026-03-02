# Use the official ROS 2 Humble desktop image
FROM osrf/ros:humble-desktop

# Set non-interactive to avoid timezone prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic utility tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    tmux \
    nano \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for GUI and Gazebo
ENV DISPLAY=$DISPLAY
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV GZ_VERSION=garden

# Source ROS 2 automatically in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install Gazebo Garden and Dev Headers
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && apt-get install -y \
    gz-garden \
    libgz-sim7-dev \
    libgz-transport12-dev \
    libgz-sensors7-dev \
    libgz-math7-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone PX4 (v1.14 stable)
WORKDIR /
RUN git clone https://github.com/PX4/PX4-Autopilot.git -b release/1.14 --recursive

# Install PX4 Dependencies
WORKDIR /PX4-Autopilot
RUN bash ./Tools/setup/ubuntu.sh --no-sim-tools
RUN pip3 install --no-cache-dir "setuptools<59.0.0" "empy<4" kconfiglib jsonschema jinja2 pyros-genmsg

# XRCE-DDS Bridge & ROS 2 Tools
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /Micro-XRCE-DDS-Agent && \
    cd /Micro-XRCE-DDS-Agent && mkdir build && cd build && \
    cmake .. && make && make install && ldconfig /usr/local/lib

RUN apt-get update && apt-get install -y python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# 1. Pull the compiled uv binary directly from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Copy your requirements file into the container
COPY requirements.txt /tmp/requirements.txt

# 3. Create a virtual environment that allows ROS 2 imports
RUN uv venv /opt/venv --system-site-packages

# 4. Make the virtual environment the default Python path for all future commands
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install -r /tmp/requirements.txt

WORKDIR /workspace