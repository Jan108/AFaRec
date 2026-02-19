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
AnimalCLEF 2025
https://ceur-ws.org/Vol-4038/paper_231.pdf -> Overview Paper
https://ceur-ws.org/Vol-4038/paper_240.pdf
https://ceur-ws.org/Vol-4038/paper_245.pdf
https://ceur-ws.org/Vol-4038/paper_250.pdf
https://ceur-ws.org/Vol-4038/paper_251.pdf
https://ceur-ws.org/Vol-4038/paper_253.pdf
https://ceur-ws.org/Vol-4038/paper_258.pdf

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
    - write script to autostart mlserv2
    - correct counts in 4.2.1
    - write 2
    + update data on mlserv2
    - write 2.1 Object Detection
    - write 2.1.1 YOLO
    - write 2.1.2 RF-DETR
    + train on mlserv yolo26 / rtdetr (wait for current run to end)
    - write 4.3 Object detection
    - dedupe oai t=0.5
    - rerun oafi
    - update 4.2.1 with deduplication

Heute:
    - update data on mlserv2
    - write eval functions -> reproducability baseline+finetune, yolo26s/m, rfdetr-s/m
        - with confusion matrix like yolo train
        - test with different confidence scores

Morgen:

Next Tasks:
    - Find out which face detectors I wanna use:
        - List Sota approaches
        - try some with a lower confidence
        - find easiest to train
        - use active learning
        - complete one, then decide how many I need
        - write 4.2.2 OpenAnimalFaceImages
    - add example data (image with annotation) to 4.2.1
    - train again on mlserv2 yolo / rfdetr (less epochs)

Talk with Tobi:
    - How do I do baseline with YOLO26 because of missing classes in pretrained model


Paused:
    - write 2.1.3 RT-DETR