FROM python:3.9

# required for PDF2Image package
# RUN apt-get update
# RUN apt-get -y install poppler-utils

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY models ./models
COPY app.py ./app.py
COPY config.py ./config.py
COPY handler.py ./handler.py
COPY __init__.py ./__init__.py

EXPOSE 8080
ENTRYPOINT ["python3"]
CMD ["app.py"]