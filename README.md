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


# Reproduce Yunet
1. Run ```pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111```
2. Run ```pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html```
3. Clone Git Repo
4. Run ```cd face_detection/yunet/libfacedetection.train/```
5. Run ```python setup.py develop```
6. Run ```pip install Cython==0.29.33```
7. Run ```pip install mmpycocotools```
8. Run ```pip install yapf==0.30.0```
9. Run ```pip install -r requirements.txt ```
10. Alter ```libfacedetection.train/mmdet/datasets/pipelines/transforms.py:1082``` ```np.int``` -> ```np.int64```
11. Prepare data: Follow https://github.com/ShiqiYu/libfacedetection.train Readme.md
12. Train: ```CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh ./configs/yunet_n.py 1 12345```
13. Alter ```libfacedetection.train/tools/train.py:212``` and add following lines
```python
if cfg.load_from:
    from mmcv.runner import load_checkpoint
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cuda')
```

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

Heute:
    - train rfdetr on mlserv2

Morgen:
    - Tobis anmerkungen einarbeiten
    - train yunet
        - try their train method on wide faces -> wait for training done -> works
        - try on my data -> wait for training done
        - fine tune on pre weight? -> not possible currently
            - maybe use torch2onnx.py there model is loaded, adapt for train.py, could work \-_-/
    - do I need landmarks? -> CenterFace and retinaFace use them, but I'm not sure if they NEED them
        - find out -> I guess it works with out them, yunet should, but better with maybe?
        - alter label process?
        - animalWeb

Plan Fragen:
    - Frist Studienbüro Abgabe Mail / Briefkasten
    - Mail Prof Vortrag?
    - bis ~10.03. Klärung wann Vortrag, damit ich weiß Teilzeitstudium und mit LSW reden

Next Tasks:
    - Find out which face detectors I wanna use:
        - List Sota approaches -> good framing why i use those models 3 is good 
        - try some with a lower confidence
        - find easiest to train
        - use active learning
        - complete one, then decide how many I need
        - write 4.2.2 OpenAnimalFaceImages
    - add example data (image with annotation) to 4.2.1
    - write eval animal detection
        - with confusion matrix like yolo train
        - It needs sepparation of different classes
    - eval yolo26 s/m finetuned
    - eval yolo26 s/m standard

Talk with Tobi:
    - How do I do baseline with YOLO26 because of missing classes in pretrained model?
    - Eval depth?
    - Opinion on current text? Detailed enough?


Paused:
    - write 2.1.3 RT-DETR

| Name               | Link                                                                                                                                                                                                                                                      | F1                                              | Training Prozess                                                                                                             | Order | Why use it?                 |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|-------|-----------------------------|
| RetinaFace         | [official repo not meant for training](https://github.com/serengil/retinaface/blob/master/retinaface/model/retinaface_model.py#L46)                                                                                                                       |                                                 | Repo doesnt give training stuff, but tfmodel, can be called with fit, but needs: loss, metrics and dataset                   | 2     | highest acc / gold standard |
| MTCNN / fast mtcnn | [implementation from someone](https://github.com/etosworld/etos-mtcnn) / [pip packet](https://mtcnn.readthedocs.io/en/latest/training/) / [better tutorial with code](https://deepwiki.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch/6.4-mtcnn-training) | Outperformed SOTA at the time with large margin | 3 Networks, but maybe only need 2, wants landmark data, documented, found other repo https://github.com/etosworld/etos-mtcnn | 4?    | uses 3 networks             |
| Yunet              | [Official Train Repo](https://github.com/ShiqiYu/libfacedetection.train) / [OpenCV Readme](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)                                                                          | 0.8656 on medium AP                             | Dateset structure, one Network, has a train method                                                                           | 1     | Really small: 75k parm      |
| CenterFace         | [Unoffical repo](https://github.com/chenjun2hao/CenterFace.pytorch) / [Training Guide](https://deepwiki.com/chenjun2hao/CenterFace.pytorch/5.1-training-guide)                                                                                            | 0.9089 medium AP                                | DAtaset structure, one Network, guide with code implementation                                                               | 3     |                             |
