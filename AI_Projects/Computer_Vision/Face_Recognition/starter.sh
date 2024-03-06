#!/bin/bash

# Update package lists and install Docker
sudo apt update
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker

# Add the current user to the docker group
sudo usermod -aG docker $USER

# Install Python 3 pip
sudo apt install -y python3-pip

# Install required Python packages
pip3 install -r requirements.txt

