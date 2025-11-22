from ultralytics import YOLO

def yolov11() -> None:
    model = YOLO("yolo11s.pt")
    model.train()
    pass


if __name__ == '__main__':
    yolov11()
