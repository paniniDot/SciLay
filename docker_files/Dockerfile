FROM nvcr.io/nvidia/pytorch:22.12-py3

 

# Set the working directory inside the container

WORKDIR /workspace/

RUN apt-get update -y && apt-get autoremove -y
RUN pip install --upgrade pip

COPY requirements_llm.txt /workspace/

# Install the required packages

RUN pip install --no-cache-dir -r requirements_llm.txt

RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

RUN pip uninstall transformer_engine -y