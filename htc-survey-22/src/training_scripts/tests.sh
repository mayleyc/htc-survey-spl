#! /bin/bash

#export PYTHONPATH=$PYTHONPATH:/home/alessandro/work/repo/htc-survey
export PYTHONPATH=$PYTHONPATH:/home/alessandroz/repo/htc-survey

#cd /home/alessandro/work/repo/htc-survey || exit
cd /home/alessandroz/repo/htc-survey || exit

python3 src/training_scripts/flat/bert.py
#python3 src/training_scripts/flat/xmlcnn.py
#python3 src/training_scripts/flat/fasttext_ml.py
python3 src/training_scripts/flat/bert_match.py
python3 src/training_scripts/flat/bert_champ.py
