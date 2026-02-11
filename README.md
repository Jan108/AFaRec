# AFaRec
Link Collection:

Approaches 2014: https://www.kaggle.com/c/dogs-vs-cats/overview
Wildlife Datasets: https://github.com/WildlifeDatasets/wildlife-datasets -> only for maybe birds?
OpenImage Dataset: https://arxiv.org/pdf/1811.00982 currently v7 ? this would be v4
Coco - Common Objects in Context: https://cocodataset.org/ has image segmention for cat, dog, bird, sheep, cow, horse
PetFace (research only): https://dahlian00.github.io/PetFacePage/
LifeCLEF has Snake data with at least 3 images per individual -> metadata doesn't contain individual info :/

https://github.com/MarQuisCheshire/Pets-Face-Recognition -> only cats / dogs

Human Face Recognition/Detection
https://github.com/deepinsight/insightface
https://github.com/serengil/deepface
Benchmarks: https://paperswithcode.com/task/face-recognition/codeless

Most Popular pets:
https://americanpetproducts.org/industry-trends-and-stats
Dogs, Cats, Fish, Birds (Parrots, ...), Small animal (Hamsters, Rabbits, Guinea Pig, ...), Reptiles (Snake, Lizards, Turtles), Horses

Fish are dumb -> no naming
horses -> use selling site and crawl my own dataset ex. https://www.ehorses.de/ \~17.000 individuals
Birds -> have java and parakeet, parrot: also own dataset, ex. https://www.pets4homes.co.uk/sale/birds/parrots/ \~150 individuals
Still missing Reptiles...


Use CLIP/YOLO as detection?

Use more than one detection model:
\cite{zhang.2016_Joint} mtcnn https://github.com/kpzhang93/MTCNN\_face\_detection\_alignment https://github.com/DuinoDu/mtcnn/blob/master/demo.py
Google Facenet


Scope:

What is the current state of pet face detection and recognition?
How well perform state-of-the-art face detection and recognition approaches on pets?
How does my own approach perform?
What can be done to increase the accuracy?

General Goal: Build an ML Model that takes in an image and detects different types of animals and their faces, embed those faces and run them through a face detection algorithm.


Notes from 30.07. Talk with Bogdan:
spickende neuronale Netze
    → können das vielleicht besser

rückgekoppelte Netze

1 Seite
Ich will das machen, mit welchen Ansätzen, detaillierter als jetzt


# ToDo
-[ ] AnimalCLEF 2025
        https://ceur-ws.org/Vol-4038/paper_231.pdf -> Overview Paper
        https://ceur-ws.org/Vol-4038/paper_240.pdf
        https://ceur-ws.org/Vol-4038/paper_245.pdf
        https://ceur-ws.org/Vol-4038/paper_250.pdf
        https://ceur-ws.org/Vol-4038/paper_251.pdf
        https://ceur-ws.org/Vol-4038/paper_253.pdf
        https://ceur-ws.org/Vol-4038/paper_258.pdf
-[ ] my own dataset
  - find out which annotations I need -> pet position for animal detection (coco, openimagesv7); pet face pos for pet face detection (self annotate openimagesv7?); multiple pet faces for an individual for pet recognition (pet face)
  - plan for annotation of images based of openimagesv7 to annotate where the head is
- Next steps:
  - send mail to profs
  - start with animal detection
      - plan what to compare: yolo v11s on oai, yolo v11s default, different model?
        - see table below
        - run OpenAnimalImages test split on all models
        - For each model train on OAI, so that there are then 2 models, fine tuned/standart
        - results in 8 models
      - how to eval this? -> build custom evaluation based on wanted task (image in, class with pos out or not found)
        - can fiftyone do this? or do I need to build my own pipeline (inference with image path in; predictions out)
        - ultralytics can do it for yolo and rt-detr: https://docs.ultralytics.com/modes/val/
        - needs to eval on OAI test (is this used elsewhere?) and if it says there is non present
      - train and eval those models
      - write it down

Done:
    - Motivation schreiben
    - RQ schreiben
    - Overview unified System schreiben
    - Mail Profs Anmeldung / Termin
    - Mail Maletti Zugang Compute Server
    - Mail Bogdan aktueller Stand und Termin zwischen uns
    + create base images for oai_anno
    + RF-Detr rechnen
    - pip install label-studio
    - Add multi user support
    - add skip button
    - Dataset Annotation Plan / Umsetzen
    - Labeling Guidelines
    - ask friends and family
        - Lutz, This, Mama, Josi?, Vero?, Pepy?
    - write 4.2 datasets
    - write 4.2.1 OpenAnimalImages


Heute:
    + train on mlserv yolo26 / rtdetr (wait for current run to end)
    
Morgen:
    - write 4.2.2 OpenAnimalFaceImages
    - correct counts in 4.2.1
    - update data on mlserv2
    - write script to autostart mlserv2

Next Tasks:
    - Find out which face detectors I wanna use:
        - List Sota approaches
        - try some with a lower confidence
        - find easiest to train
        - use active learning
        - complete one, then decide how many I need
    - write 2
    - write 2.1 Object Detection
    - write 2.1.1 YOLO
    - write 2.1.2 RF-DETR
    - write 2.1.3 RT-DETR
    - write 4.3 Object detection

| name          | APval 50:95 COCO | Link                                                                           |
|---------------|------------------|--------------------------------------------------------------------------------|
| yolo v11s     | 47,0             | https://docs.ultralytics.com/models/yolo11/#performance-metrics                |
| yolo v12s     | 48,0             | https://docs.ultralytics.com/models/yolo12/#detection-performance-coco-val2017 |
| RT-DETRv2-S   | 48,1             | https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch https://colab.research.google.com/github/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb                |
| RT-DETRv3-R18 | 48,1             | https://github.com/clxia12/RT-DETRv3                                           |
| RF-DETR-S     | 53,0             | https://github.com/roboflow/rf-detr https://rfdetr.roboflow.com/learn/train/   |