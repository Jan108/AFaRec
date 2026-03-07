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

# Reproduce FaceDetection
## YuNet
Clone [YuNet](https://github.com/Jan108/YuNet) into the [face_detection/YuNet](face_detection/YuNet) directory. And follow the [README.md](face_detection/YuNet/README.md)

## SCRFD
Clone [SCRFD](https://github.com/Jan108/SCRFD) into the [face_detection/SCRFD](face_detection/SCRFD) directory. And follow the [README.md](face_detection/SCRFD/README.md)

## RetinaFace
Clone [RetinaFace](https://github.com/Jan108/RetinaFace) into the [face_detection/RetinaFace](face_detection/RetinaFace) directory. And follow the [README.md](face_detection/RetinaFace/README.md)

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
    + update data on mlserv2
    + write eval functions -> reproducibility baseline+finetune, yolo26s/m, rfdetr-s/m
    - Analyze face detectors how to train?
    + train again on me yolo / rfdetr (less epochs)
    - adapt oafi_annotation with correct dataset and tags
    - fix current oafi needed_anno
    - train yunet
        - try their train method on wide faces -> wait for training done -> works
        - try on my data -> wait for training done
        - fine tune on pre weight? -> not possible currently
            - maybe use torch2onnx.py there model is loaded, adapt for train.py, could work \-_-/
    - write eval yunet
        - train all classes; train one per class
        - predict into fiftyone?
    - do I need landmarks? -> CenterFace and retinaFace use them, but I'm not sure if they NEED them
        - find out -> I guess it works with out them, yunet should, but better with maybe?
        - alter label process?
        - animalWeb
        -> No dont need them!
    - Find out which face detectors I wanna use:
        - List Sota approaches -> good framing why i use those models 3 is good
    - train rfdetr on mlserv2
    - Tobis anmerkungen einarbeiten
    - write 2.2 Face detection
    - write 2.2.1 RetinaFace
    - write 2.2.3 Yunet
    - write 2.2.2 CenterFace
    - write 2.2.4 SCRFD
    - write 4.2.2 OpenAnimalFaceImages
    - Mail Prof Vortrag?
    - add example data (image with annotation) to 4.2.1
        - dog: /mnt/data/afarec/data/OAFI_full/images/test/000f24d363c4b66c_0.jpg
        - bird: /mnt/data/afarec/data/OAFI_full/images/test/139a32269caab028_0.jpg
        - small animals: /mnt/data/afarec/data/OAFI_full/images/test/1c30a83373b711ce_0.jpg / /mnt/data/afarec/data/OAFI_full/images/test/1c30a83373b711ce_1.jpg
        - horse_like: /mnt/data/afarec/data/OAFI_full/images/test/1e69455253b84280_3.jpg
        - cat_like: /mnt/data/afarec/data/OAFI_full/images/test/115e3db0d6aa77e7_3.jpg
        - cat: /mnt/data/afarec/data/OAFI_full/images/test/6bdfb462395b22b6_0.jpg
        - dog_like: /mnt/data/afarec/data/OAFI_full/images/train/026748cbec73bd58_0.jpg
        - Annotator image: 699ca2f849c05958b946aee6
    - write 4.4 Animal Face Detection
    - write 4.4.1 Generalisation vs specification
    - RetinaFace with different Backbone MobinetV2, Resnet18
    - Yunet-s
    - Early stopping? -> I need to change some of the training params, but I don't know how yet... -> reduce LR, leave rest as is
    - Train/Predict CenterFace, SCRFD? -> CenterFace does not have available weights for training (Repo is also ass)
    - #Params Centerface via sum(p.numel() for p in model.parameters())
    - Clear status for Tobi: Anno done OAFI, fix errors
    - added images to annotate 1600 to bird, 750 to horse_like, 250 dog, 150 cat_like, 350 small_animals for annotation
    

Heute:
    - Mail Maletti Termin
    - Stammdaten LSW
    - Start with Face Recognition

Morgen:
    - write 3 Related Work

Plan Fragen:
    - Frist Studienbüro Abgabe Mail / Briefkasten
    - bis ~10.03. Klärung wann Vortrag, damit ich weiß Teilzeitstudium und mit LSW reden

Next Tasks:
    - Eval:
        - eval yolo26 s/m finetuned
        - eval yolo26 s/m standard
        - write eval animal detection
        - mAP: eval all classes/per class (cur only area size)
        - Confusion matrix plot for bbox
        - interannotator agreement -> ich schaue nochmal über 250 Bilder von Mama rüber und berechne wie die übereinstimmen
    - reframe intro: Problem isn't solved, thats what I do -> problem not trivial

Talk with Tobi:
    - How do I do baseline with YOLO26 because of missing classes in pretrained model?
    - Eval depth?
    - Opinion on current text? Detailed enough?


| Name               | Link                                                                                                                                                                                                                                                      | F1                                              | Training Prozess                                                                                                             | Order | Why use it?                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|-------|-----------------------------|
| RetinaFace         | [official repo not meant for training](https://github.com/serengil/retinaface/blob/master/retinaface/model/retinaface_model.py#L46)                                                                                                                       |                                                 | Repo doesnt give training stuff, but tfmodel, can be called with fit, but needs: loss, metrics and dataset                   | 2     | highest acc / gold standard |
| MTCNN / fast mtcnn | [implementation from someone](https://github.com/etosworld/etos-mtcnn) / [pip packet](https://mtcnn.readthedocs.io/en/latest/training/) / [better tutorial with code](https://deepwiki.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/6.4-mtcnn-training) | Outperformed SOTA at the time with large margin | 3 Networks, but maybe only need 2, wants landmark data, documented, found other repo https://github.com/etosworld/etos-mtcnn | -     | uses 3 networks             |
| Yunet              | [Official Train Repo](https://github.com/ShiqiYu/libfacedetection.train) / [OpenCV Readme](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)                                                                          | 0.8656 on medium AP                             | Dateset structure, one Network, has a train method                                                                           | 1     | Really small: 75k parm      |
| CenterFace         | [Unoffical repo](https://github.com/chenjun2hao/CenterFace.pytorch) / [Training Guide](https://deepwiki.com/chenjun2hao/CenterFace.pytorch/5.1-training-guide)                                                                                            | 0.9089 medium AP                                | DAtaset structure, one Network, guide with code implementation                                                               | -     | anchorless, no weights :(   |
| SCRFD              | [Offical Repo Train](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)                                                                                                                                                              | 94.92 AP Medium                                 | WiderFace with MMDet like Yunet                                                                                              | 4     | training optimization       |
