# Graph-based Rural Bus Route Planning

This project can be built as a Docker image so that the included Jupyter notebooks can be inspected and run as follows:

- Clone the repository:
  ```
  git clone https://github.com/ancodia/donegal-bus-routes.git
  ```

- Download and install Docker: 

  https://www.docker.com/get-started

- Change to the donegal-bus-routes project directory:
  ```
  cd project_dir
  ```

- Build the Docker image:
  ```
  docker build -t donegal-bus-routes .
  ```

- Run a container:
  ```
  docker run -p 8888:8888 donegal-bus-routes
  ```

- Open the link found in the Docker terminal window like: http://127.0.0.1:8888/?token=...
