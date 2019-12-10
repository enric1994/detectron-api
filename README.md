## Human Mask Generator API, made using Facebook's Detectron2 model

Powered by Docker. GPU support

![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fenric1994%2Fdetectron-api)

### Input
<img src="demo_img.jpg" width="40%">

### Output
<img src="response_file.png" width="40%">

## Dependencies
* Any Linux distribution and a GPU
* [Docker](https://gist.github.com/enric1994/3b5c20ddb2b4033c4498b92a71d909da)
* [Docker-Compose](https://gist.github.com/enric1994/3b5c20ddb2b4033c4498b92a71d909da)
* [nvidia-Docker](https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)

## Usage
1. From the `docker` folder run: `docker-compose up -d` 

After building the Docker container (it can take a while), the API will start.

You can check the status of the container with: `docker ps -a`

You can also see what's going on inside the container with `docker logs detectron-api -f`

2. On the main folder run: `python send_file.py` 

This will send `demo_img.jpg` to the API using the Python requests package.

Feel free to modify the script and add it to your pipeline

Tested on Ubuntu 18.04 with a RTX 2080 Ti GPU (drivers 440.36)

## Advantages
* You don't have to install CUDA
* Language agnostic
* Very easy to integrate in any pipeline, only a HTTP request needed.
* The API will be isolated from your code. No dependency errors
