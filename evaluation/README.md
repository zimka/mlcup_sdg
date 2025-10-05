## How to build
0) cd mlcup_sdg/evaluation
1) sudo docker build . -t worldscore:latest
2) sudo docker run --rm -it  --gpus all worldscore:latest /bin/bash