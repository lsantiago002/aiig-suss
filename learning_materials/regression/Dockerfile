FROM python:3.7-slim

RUN apt-get update

WORKDIR /learning_materials/regression
COPY . /learning_materials/regression

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# CMD ["bash"]
