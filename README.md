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

Talk Maletti:
    - Themen:
        - Mein Ziel Abgabe 31.03., und Arbeitsbeginn 01.04.
        - aktueller Stand
            - Animal Detection: Implementiert und berechnet, eval fehlt
            - Face Detection: Implementiert und 2/3 berechnet, eval fehlt
            - Face Recognition: 1/3 Implementiert und 1/3 rechnet, eval fehlt
            - Annotation: Fertig 15,000 Bilder
        - Was kommt noch?
            - Animal Detection: Evaluation, compare with baseline; compare with each other; show problems
            - Face Detection: Evaluation, compare with baseline; compare speciliced with each other; show problems
            - Annotation: Interannotator agreement, highlight Problems with dataset, show difficulties
            - Face Recognition: implement rest and calculate; Evaluation: Verification Task: gg 2 Faces selbe Identität, inter-cls vs intra-cls
            - Nice to have: Cross age analysis on some dogs/cats (2cats, 4-5dogs)
        - Feedback
            - Meinung zu aktuellem Ansatz:
                - Baseline kurz fine tunen auf den oafi, ohne meine schritte, also ohne spezies 
                - Beiden schmeckt die Baseline nicht, da ich einen Vergleich mache, ohne das beide Ansätze die selben Daten gesehen haben
            - Allgemein:
                - sieht aus wie Master Level
        - Vortrag
            - Vorgaben?: Länge, Uni-Vorlage, Wo, Wann, vor wem?, Sprache
            - 27.03.: 60min
            - Falls Sie Bilder von ihren Tieren sehen wollen, können Sie mir die gerne schicken -> Bodgan hat Bock
        - Fragen allgemein:
            - Was ist ihnen bei der Arbeit/Code wichtig?
                - Bogdan, Bilder beschriftet, lesbar, Kurven erkennbar und identifizierbar
                - tausender trennung tabellen
            - Was passiert, wenn sie sagen durchgefallen? / Was muss gegeben sein damit das nicht passiert?
                - nur wenn ich schummel, sonst bei aktueller Leistung kein Problem
            - Teilzeitstudium anmelden ja/nein?
                - nein, lohnt nicht
            - Haben Sie noch Ideen/Wünsche/Vorgaben?
                - Persönlich abgeben am Di und sicher stellen, dass Stempel gesetzt als abgegeben
                - USB-Stick mit Code


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
    - 2.2 Datasets brigde to pretraining weights and why its important to me
    - 2.2 short desc categorys
    - Bounding box explained?
    - Label guidelines details
    - Mail Maletti Termin
    - define experiments FR
    - Start with Face Recognition
    - implement ArcFace and train it
    + implement GhostFaceNet
    - Frist Studienbüro Abgabe Mail / Briefkasten -> Abgabe am 31.03. persönlich im Büro, mit USB für Daten
    - bis ~10.03. Klärung wann Vortrag, damit ich weiß Teilzeitstudium und mit LSW reden
    - write 2.3 Face Recognition
    - write 2.3.1 SphereFace
    - write 2.3.2 ArcFace
    - write 2.3.3 GhostFaceNet
    

Heute:
    + SCRFD on mlserv2
    + GhostFaceNet on mlserv2

Morgen:
    - implement SphereFace

Next Tasks:
    - write 4.2.3 PetFace
    - write 4.5 Face Recognition
    - write 3 Related Work
    - Eval:
        - eval yolo26 s/m finetuned
        - eval yolo26 s/m standard
        - write eval animal detection
        - mAP: eval all classes/per class (cur only area size)
        - Confusion matrix plot for bbox
        - interannotator agreement -> ich schaue nochmal über 250 Bilder von Mama rüber und berechne wie die übereinstimmen
    - reframe intro: Problem isn't solved, thats what I do -> problem not trivial
    - Tobi Points
        - maybe make names cursive
        - maybe add small headlines in long texts like 2.2/2.3
    (- broken train-val-test split deterministic for OAFI)


Talk with Tobi:
    - How do I do baseline with YOLO26 because of missing classes in pretrained model?
    - Eval depth?
    - Opinion on current text? Detailed enough?

Plan Fragen:

Backbones: Ghostfacenet, ResNet50, ResNet100, VGG-Face?, FaceNet
Loss: ArcFace, CentreLoss, CosFace, Softmax, A-Softmax (SphereFace), TripletLoss

Data: Open-set vs closed-set -> learn classification; learn embeddings -> good fig wang.2021_Deep Fig. 20

Amount Times 5

ResNet 34/50 X ArcFace/CosFace https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch 4Modelle
SphereFaceNet 20/64 X SphereFace https://github.com/ydwen/opensphere/tree/OpenSphere_v0 2Models
GhostFaceNetV1-1.3-1 X ArcFace/CosFace https://github.com/HamadYA/GhostFaceNets 2 Models
