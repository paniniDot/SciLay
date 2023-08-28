FROM huggingface/transformers-pytorch-latest-gpu

 

# Set the working directory inside the container

WORKDIR /app

 

# Copy the requirements file into the container at /app

COPY requirements.txt /app/

 

# Install the required packages

RUN pip install --no-cache-dir -r requirements.txt
