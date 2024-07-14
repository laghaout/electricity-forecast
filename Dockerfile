FROM python:3.11-slim

# Declare the working directory.
WORKDIR forecast

# Flag that we're in a Docker container.
ENV DOCKERIZED Yes

# Install Linux utilities.
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install less
RUN apt -y install tree

# Install Python packages.
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install tensorflow==2.15.0
RUN pip3 install seaborn==0.12.2
RUN pip3 install matplotlib==3.8.0
RUN pip3 install numpy==1.26.4
RUN pip3 install pandas==2.1.4
RUN pip3 install scikit-learn==1.2.2
RUN pip3 install pytest==7.4.0
RUN pip3 install python-dotenv==1.0.1
RUN pip3 install pydantic==1.10.12

# Install the local package.
COPY Dockerfile *.py *.yml *.txt *.sh *.toml README.* .pre-commit-config.yaml .env ./
COPY forecast/ forecast
COPY tests/ tests

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "main.py"]
