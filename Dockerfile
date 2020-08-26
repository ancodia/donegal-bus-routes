# Set the base image
FROM ubuntu:18.04

# Adding the user "docker" and switching to "docker"
RUN useradd -ms /bin/bash docker && su docker

# Copy Python requirements to image
RUN mkdir /donegal-bus-routes
COPY requirements.txt /donegal-bus-routes

# Installing Python 3.8, dependencies and pip packages
RUN apt-get update && apt-get install -y software-properties-common \
  && apt-get update && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update  \
  && apt-get install -y python3.8 python3-pip python3-dev libproj-dev \
                        proj-data proj-bin libgeos-dev libspatialindex-dev curl autoconf libtool \
  #&& ln -s /usr/bin/python3 /usr/local/bin/python \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && python --version \
  && pip3 install --upgrade pip \
  && curl -L https://github.com/libspatialindex/libspatialindex/archive/1.8.5.tar.gz | tar -xz \
  && cd libspatialindex-1.8.5 && ./autogen.sh && ./configure && make && make install \
  && pip3 install numpy cython \
  && cd /donegal-bus-routes && pip3 install -r requirements.txt

# add project to sys.path so modules can be found for notebook imports
ENV PYTHONPATH "${PYTHONPATH}:/donegal-bus-routes"

# Setting Jupyter notebook configurations
RUN jupyter notebook --generate-config --allow-root

# Run the command to start the Jupyter server
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/donegal-bus-routes", "--ip=0.0.0.0", "--port=8888", "--no-browser"]