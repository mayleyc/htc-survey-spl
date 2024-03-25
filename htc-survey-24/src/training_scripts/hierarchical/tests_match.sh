#! /bin/bash

# You might need to setup your environment
# export PYTHONPATH=$PYTHONPATH:path/to/repository
# e.g.: export PYTHONPATH=$PYTHONPATH:/home/alessandroz/repo/htc-survey-22
# cd path/to/repository
# e.g.: cd /home/alessandroz/repo/htc-survey-22 || exit

# Comment/ Uncomment as needed
python3 src/models/Match/main.py -d config/Match/BUGS.yaml -m config/Match/model.yaml --reg 1 --mode train
python3 src/models/Match/main.py -d config/Match/BUGS.yaml -m config/Match/model1.yaml --reg 1 --mode train
python3 src/models/Match/main.py -d config/Match/BUGS.yaml -m config/Match/model2.yaml --reg 1 --mode train
python3 src/models/Match/main.py -d config/Match/WOS.yaml -m config/Match/model2.yaml --reg 1 --mode train
python3 src/models/Match/main.py -d config/Match/AMZ.yaml -m config/Match/model2.yaml --reg 1 --mode train
python3 src/models/Match/main_split.py -d config/Match/BGC.yaml -m config/Match/model2.yaml --reg 1 --mode train
python3 src/models/Match/main_split.py -d config/Match/RCV1.yaml -m config/Match/model2.yaml --reg 1 --mode train

