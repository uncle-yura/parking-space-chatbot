FROM python:3.13-slim-trixie AS python

RUN pip install --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY ./app ./app
COPY ./data ./data
COPY ./langgraph.json .

EXPOSE 8123

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8123", "--no-browser"]
