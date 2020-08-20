#Set the base image
FROM ubuntu:18.04

# Adding the user "docker" and switching to "docker"
RUN useradd -ms /bin/bash docker
RUN su docker

# Setting the installation logfile and direcotry locations
ENV LOG_DIR_DOCKER="/root/dockerLogs"
ENV LOG_INSTALL_DOCKER="/root/dockerLogs/install-logs.log"

RUN mkdir -p ${LOG_DIR_DOCKER} \
 && touch ${LOG_INSTALL_DOCKER}  \
 && echo "Logs directory and file created"  | sed -e "s/^/$(date +%Y%m%d-%H%M%S) :  /" 2>&1 | tee -a ${LOG_INSTALL_DOCKER}

# Installing the python dependencies
RUN apt-get update | sed -e "s/^/$(date +%Y%m%d-%H%M%S) :  /" 2>&1 | tee -a ${LOG_INSTALL_DOCKER} \
  && apt-get install -y python3-pip python3-dev libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev | sed -e "s/^/$(date +%Y%m%d-%H%M%S) :  /" 2>&1 | tee -a ${LOG_INSTALL_DOCKER} \
  && ln -s /usr/bin/python3 /usr/local/bin/python | sed -e "s/^/$(date +%Y%m%d-%H%M%S) :  /" 2>&1 | tee -a ${LOG_INSTALL_DOCKER} \
  && pip3 install --upgrade pip | sed -e "s/^/$(date +%Y%m%d-%H%M%S) :  /" 2>&1 | tee -a ${LOG_INSTALL_DOCKER}

# libspatialindex required by rtree Python package
RUN apt-get install -y --fix-missing curl autoconf libtool
RUN curl -L https://github.com/libspatialindex/libspatialindex/archive/1.8.5.tar.gz | tar -xz
RUN cd libspatialindex-1.8.5 && ./autogen.sh && ./configure && make && make install

# copy project to docker container
COPY . /donegal-bus-routes

WORKDIR /donegal-bus-routes

# Installing python packages
RUN pip3 install numpy
RUN pip3 install cython
RUN pip3 install -r requirements.txt

# add project to sys.path so modules can be found for notebook imports
ENV PYTHONPATH "${PYTHONPATH}:/donegal-bus-routes"

# Setting Jupyter notebook configurations
RUN jupyter notebook --generate-config --allow-root
#RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# Run the command to start the Jupyter server
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/donegal-bus-routes", "--ip=0.0.0.0", "--port=8888", "--no-browser"]