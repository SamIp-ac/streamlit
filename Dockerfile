# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SamIp-ac/streamlit.git .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "main_page.py", "--server.port=8501", "--server.address=0.0.0.0"]

CMD streamlit run main.py --server.port 8501