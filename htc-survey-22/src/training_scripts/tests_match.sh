#! /bin/bash

#export PYTHONPATH=$PYTHONPATH:/home/alessandro/work/repo/htc-survey
export PYTHONPATH=$PYTHONPATH:/home/alessandroz/repo/htc-survey

#cd /home/alessandro/work/repo/htc-survey || exit
cd /home/alessandroz/repo/htc-survey || exit

#python3 src/models/Match/main.py -d config/Match/old/BUGS.yaml -m config/Match/old/model.yaml --reg 1 --mode train
#python3 src/models/Match/main.py -d config/Match/old/BUGS.yaml -m config/Match/old/model1.yaml --reg 1 --mode train
#python3 src/models/Match/main.py -d config/Match/old/BUGS.yaml -m config/Match/old/model2.yaml --reg 1 --mode train
#python3 src/models/Match/main.py -d config/Match/old/WOS.yaml -m config/Match/old/model2.yaml --reg 1 --mode train
#python3 src/models/Match/main.py -d config/Match/old/AMZ.yaml -m config/Match/old/model2.yaml --reg 1 --mode train
python3 src/models/Match/main_split.py -d config/Match/old/BGC.yaml -m config/Match/old/model2.yaml --reg 1 --mode train
python3 src/models/Match/main_split.py -d config/Match/old/RCV1.yaml -m config/Match/old/model2.yaml --reg 1 --mode train
#python3 src/training_scripts/flat/xmlcnn.py
#python3 src/training_scripts/flat/fasttext_ml.py
#python3 src/training_scripts/flat/bert_match.py
#python3 src/training_scripts/flat/bert_champ.py
