FROM continuumio/miniconda3:22.11.1

RUN mkdir /home/dev
WORKDIR /home/dev

# install pip, pip-tools, and setuptools
RUN pip install --no-cache-dir --upgrade pip pip-tools setuptools

# Install PyTorch with CUDA support and openCV
RUN pip install --pre torch==2.0.1+cu117 torchvision --index-url https://download.pytorch.org/whl/cu117
RUN pip install torchmetrics==1.0.1

# install pip packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get --allow-releaseinfo-change update
RUN apt-get --allow-releaseinfo-change-suite update
RUN apt-get update
RUN apt-get install -y vim

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--allow-root"]
