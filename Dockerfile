FROM python:3

ADD requirements.txt /
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/yaml/pyyaml.git

ADD main.py /
COPY configs /configs
COPY src /src
RUN mkdir Exp

ENTRYPOINT [ "python", "./main.py" ]