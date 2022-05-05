FROM python:3.8 as builder
COPY requirements.txt .

RUN pip3 install --user -r requirements.txt
COPY . .
CMD [ "python3", "test.py"]