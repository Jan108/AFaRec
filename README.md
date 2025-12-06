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
-[ ] Code Boilerplate for project
-[ ] make data available
-[ ] my own dataset
  - find out which annotations I need -> pet position for animal detection (coco, openimagesv7); pet face pos for pet face detection (self annotate openimagesv7?); multiple pet faces for an individual for pet recognition (pet face)
  - plan for annotation of images based of openimagesv7 to annotate where the head is
- Next steps:
  - send mail to profs
  - start with animal detection