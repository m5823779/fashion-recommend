FROM tiangolo/uwsgi-nginx-flask:python3.6
COPY docker/requirements.txt ./
RUN pip install -r requirements.txt
COPY /app /app