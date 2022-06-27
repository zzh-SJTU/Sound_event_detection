# Sound_event_detection
Sound event detection using RCNN with week labels  
put data into this directory  
models.py  -- RCNN model   
dataset.py -- dataset process with data augamentation  

run the following command to conduct the experiments with the best performance without data augamentation   
python run.py train_evaluate configs/baseline.yaml data/eval/feature.csv data/eval/label.csv 

run the following command to conduct experiments with data augamentation  
python run_2.py --augment noi/pitch/mix

noi/pitch/mix indicate different data augamentation methods
