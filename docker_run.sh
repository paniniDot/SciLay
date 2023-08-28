#! /bin/bash

docker run --name nlp-container --gpus all -v /home/panni/workspace:/workspace -v /llms:/llms -v /datasets:/datasets -it nlp-image test.sh
