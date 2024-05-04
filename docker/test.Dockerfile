FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "-m", "unittest", "discover", "-s", "tests/unit"]
