# Set the base image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Adding the user "docker"
RUN useradd -ms /bin/bash docker

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && python --version \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base Python packages
RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install numpy cython

# Copy and install Python requirements
RUN mkdir /donegal-bus-routes
COPY requirements.txt /donegal-bus-routes
RUN cd /donegal-bus-routes && python3.8 -m pip install -r requirements.txt

# Add project to sys.path so modules can be found for notebook imports
ENV PYTHONPATH="${PYTHONPATH}:/donegal-bus-routes"

# Copy project files
COPY . /donegal-bus-routes

# Configure Jupyter
RUN python3.8 -m ipykernel install --user \
    && jupyter notebook --generate-config --allow-root

# Run Jupyter server
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/donegal-bus-routes", "--ip=0.0.0.0", "--port=8888", "--no-browser"]