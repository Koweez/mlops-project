FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# copy files into the container
COPY front.py /app/front.py
COPY requirements.txt /app/requirements.txt

# set the working directory
WORKDIR /app

# install dependencies
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"]
