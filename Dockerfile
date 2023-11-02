# Dockerfile for a Streamlit web application
# ----------------------------------------------------------
# This Dockerfile is used to build a Docker image for running
# a Streamlit-based web application. It starts with an Alpine-
# based Python 3.10 image, sets up the working directory,
# installs the Python dependencies listed in requirements.txt,
# copies the application source code into the container,
# exposes port 8501 to allow external access, and finally,
# runs the Streamlit app using the specified command.
# ----------------------------------------------------------

FROM python:3.10.0-alpine

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/app.py"]
