FROM python:3.4

RUN pip install neat-python
RUN pip install graphviz matplotlib gizeh moviepy Pillow

RUN apt-get update
RUN apt-get install -y graphviz

RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl
RUN pip install python-mnist
RUN pip install git+git://github.com/pybrain/pybrain.git

RUN mkdir /src
WORKDIR /src

CMD ["bash"]
