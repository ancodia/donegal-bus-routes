# Graph Analytics for Rural Bus Route Planning

This project can be built as a Docker image so that the included Jupyter notebooks can be inspected and run as follows:

Prerequisites: Install Docker (https://www.docker.com/get-started)

- Clone the repository and build the Docker image:
  ```
  git clone https://github.com/ancodia/donegal-bus-routes.git
  cd donegal-bus-routes
  docker build -t donegal-bus-routes .
  ```

- Run a container, mounting the project directory:
  
  Linux:
  ```
  docker run -v $(pwd):/donegal-bus-routes -p 8888:8888 donegal-bus-routes
  ```
  Windows cmd (untested):
  ```
  docker run -v %cd%:/donegal-bus-routes -p 8888:8888 donegal-bus-routes
  ```
  
- Open the link found in the Docker terminal window like: http://127.0.0.1:8888/?token=...
