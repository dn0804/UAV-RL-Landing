#!/bin/bash

# 1. Check for NVIDIA GPU hardware
if lspci | grep -i nvidia &> /dev/null; then
    echo "NVIDIA hardware detected."
    
    # 2. Check if NVIDIA Container Toolkit is installed
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        echo "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi

    # 3. Create Docker Compose Override for GPU
    cat <<EOF > docker-compose.override.yml
services:
  uav_dev:
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
else
    echo "No NVIDIA hardware detected. Falling back to CPU rendering."
    rm -f docker-compose.override.yml
fi

# 4. Set display permissions for X11
xhost +local:docker &> /dev/null

# 5. Build and Start
docker compose up -d --build
echo "Container started! Run 'docker exec -it uav_rl_container bash' to enter."