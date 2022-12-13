FROM python:3.9.6

# Maintainer info
LABEL maintainer="i60996395@gmail.com"

EXPOSE 8080
# Make working directories
RUN  mkdir -p  /airmusicstramlit_demo
WORKDIR  /airmusicstramlit_demo

# Upgrade pip with no cache
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
# Run the python application
CMD ["python", "main.py"]
ENTRYPOINT ["streamlit", "run", "./main.py", "--server.port=8080", "--server.address=0.0.0.0"]