from pathlib import Path

from supervision import DetectionDataset
from supervision.metrics import MeanAveragePrecision
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import fiftyone as fo

from data.creation import load_yolo_dataset_from_disk


def train_yolo26s(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26s.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-s_oai_2026-02-07")


def train_yolo26m(dataset_dir: str = '/mnt/data/afarec/data/') -> None:
    YOLO("yolo26m.pt").train(data=dataset_dir+"OpenAnimalImages/dataset.yaml",
                             epochs=100, imgsz=640, device="cuda", batch=16, save=True, save_period=1,
                             name="yolo-m_oai_2026-02-07")


def resume_training() -> None:
    model = YOLO("/mnt/data/afarec/code/runs/detect/rtdetrv1-l_oai_2026-02-02/weights/last.pt")
    results = model.train(resume=True)
    # print(results)
    model.predict()


def eval_model(model: YOLO, dataset_path: str = '/mnt/data/afarec/data/OpenAnimalImages_RF-DETR/') -> None:
    dataset = DetectionDataset.from_yolo(
        images_directory_path=dataset_path + 'test/images',
        annotations_directory_path=dataset_path + 'test/labels',
        data_yaml_path=dataset_path + 'data.yaml',
    )
    map_metric = MeanAveragePrecision()
    for img_path, img, anno in tqdm(dataset, desc='Inference'):
        prediction = model.predict(source=img_path, device='cuda', verbose=False)
        det = sv.Detections(xyxy=prediction[0].boxes.xyxy.cpu().numpy(),
                            confidence=prediction[0].boxes.conf.cpu().numpy(),
                            class_id=prediction[0].boxes.cls.cpu().numpy())
        map_metric.update(det, anno)
        pass

    map_results = map_metric.compute()
    print('YOLO results:')
    print(map_results)


def eval_fiftyone(model: YOLO, dataset: fo.Dataset):
    for sample in tqdm(dataset):
        results = model(sample.filepath, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalized xyxy
                conf = box.conf[0].item()
                cls_idx = int(box.cls[0].item())
                label = results.names[cls_idx] if cls_idx in results.names else f"class_{cls_idx}"

                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=[x1, y1, x2 - x1, y2 - y1],  # [nx, ny, nw, nh]
                        confidence=conf,
                    )
                )

        sample["prediction_yolo26s_2026-02-13_best"] = fo.Detections(detections=detections)
        sample.save()



if __name__ == '__main__':
    # train_yolo26s()
    # resume_training()
    # eval_model(model=YOLO("/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-s_oai_2026-02-13_v3/weights/best.pt"))

    oai_Dataset = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OpenAnimalImages/')).match_tags('test')

    eval_fiftyone(model=YOLO("/mnt/data/afarec/code/runs/mlserv2/runs/detect/yolo-s_oai_2026-02-13_v3/weights/best.pt"),
                  dataset=oai_Dataset)

    session = fo.launch_app(address='0.0.0.0')
    session.wait(-1)