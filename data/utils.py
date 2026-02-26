from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def test_mongo_connection():
    try:
        client = MongoClient(fo.config.database_uri, serverSelectionTimeoutMS=5000)
        # Force a connection to check if the server is available
        client.admin.command('ping')
        return
    except ConnectionFailure as e:
        msg = ("MongoDB connection failed, but FiftyOne needs one. Start the service with: \n"
               "sudo systemctl start mongod")
        raise ConnectionError(msg)


def convert_xyhwn_to_xyxy(bbox: tuple[int, int, int, int], img_height: int, img_width: int) -> tuple[
    int, int, int, int]:
    """
    Converts BBOX from FiftyOne format to xmin, ymin, xmax, ymax with absolut values based on image height and width

    :param bbox: BBox in FiftyOne format xmin, ymin, width, height relative
    :param img_height: Image height
    :param img_width: Image width
    :return: absolute xmin, ymin, xmax, ymax
    """

    def minmax_abs(v: float, a: int) -> int:
        return round(max(min(v, 1), 0) * a)

    xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xmax = minmax_abs(xmin + w, img_width)
    ymax = minmax_abs(ymin + h, img_height)
    xmin = minmax_abs(xmin, img_width)
    ymin = minmax_abs(ymin, img_height)
    return xmin, ymin, xmax, ymax


def convert_xyxy_to_xyhwn(xyxy: tuple[int, int, int, int], img_height: int,
                          img_width: int) -> tuple[float, float, float, float]:
    """
    Convert bounding box from xyxy format to xywhn format
    :param xyxy: BBox in xyxy format xmin, ymin, xmax, ymax
    :param img_height: Image height
    :param img_width: Image width
    :return: normalized xmin, ymin, width, height
    """
    x_min, y_min, x_max, y_max = xyxy

    width = x_max - x_min
    height = y_max - y_min

    x_min_norm = x_min / img_width
    y_min_norm = y_min / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_min_norm, y_min_norm, width_norm, height_norm


def load_yolo_dataset_from_disk(save_dir: Path, max_samples: int = None, persistence: bool = False,
                                name: str = None, unified_label_distribution: bool = False) -> fo.Dataset:
    """
    Loads a YOLO dataset from disk.

    :param save_dir: Directory where the dataset is saved in YOLO format
    :param max_samples: [Optional] Maximum number of samples per split to load
    :param persistence: [Optional] Should the dataset be persisted in the FiftyOne DB?
    :param name: [Optional] Name of the dataset
    :param unified_label_distribution: [Optional] Should the amount of samples per class be the same, when using max_samples?
    :return: Loaded fo.Dataset
    """
    save_dir = Path(save_dir)
    if not save_dir.exists() or not save_dir.is_dir():
        raise FileNotFoundError(f'The directory {save_dir} does not exist.')
    if not any(save_dir.iterdir()):
        raise ValueError(f'The directory {save_dir} is empty.')
    if not (save_dir / 'dataset.yaml').exists():
        raise ValueError(f'The directory {save_dir} is not a valid YOLO dataset: dataset.yaml not found')

    if name is None:
        name = f'{save_dir.name}'
        if max_samples is not None:
            name += f'-{max_samples}'

    if name in fo.list_datasets():
        print(f'Dataset {name} is already loaded into the FiftyOne DB.')
        dataset = fo.load_dataset(name)
        if persistence and not dataset.persistent:
            dataset.persistent = True
            print(f"The dataset {name} is wasn't persisted in the FiftyOne DB, but is now.")
        return dataset

    if unified_label_distribution and max_samples is not None:
        full_dataset = load_yolo_dataset_from_disk(save_dir=save_dir)
        dataset = fo.Dataset(name=name)
        labels = list(full_dataset.count_values('ground_truth.detections.label').keys())
        for label in labels:
            for split in ['train', 'validation', 'test']:
                label_view = full_dataset.match_tags(split).filter_labels("ground_truth", F("label").is_in([label]))
                dataset.add_samples(label_view.take(int(max_samples / len(labels)), seed=42))
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=save_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            max_samples=max_samples,
            seed=42,
            name=name,
            tags='test',
            split='test',
        )
        dataset.add_dir(
            dataset_dir=save_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            max_samples=max_samples,
            seed=42,
            tags='train',
            split='train',
        )
        dataset.add_dir(
            dataset_dir=save_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            max_samples=max_samples,
            seed=42,
            tags='validation',
            split='val',
        )
    dataset.persistent = persistence
    return dataset
