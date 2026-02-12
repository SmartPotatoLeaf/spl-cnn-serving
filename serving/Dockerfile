FROM tensorflow/serving:2.18.0

# Sets model name and version as build arguments
ARG MODEL_NAME=SmartPotatoLeaf
ARG MODEL_VERSION=1

# Sets environment variable for model name
ENV MODEL_NAME=${MODEL_NAME}

# Copy the serving dir to expected dir: /models/<MODEL_NAME>/<version>
# Its assumed that the model files are in the local path: serving/v1
COPY v1 /models/${MODEL_NAME}/${MODEL_VERSION}

# Expose ports for gRPC and RESTful API
# gRPC port 8500
# RESTful API port 8501
EXPOSE 8501
EXPOSE 8500