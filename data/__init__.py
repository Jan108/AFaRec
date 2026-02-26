from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

import oafi
import oai
import utils

# Setup FiftyOne
utils.test_mongo_connection()
fo.config.dataset_zoo_dir = Path('/mnt/data/afarec/data')
fo.config.database_uri = 'mongodb://127.0.0.1:27017'
fo.config.database_validation = False


def load_coco() -> None:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        label_types=["detections"],
        classes=['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    )
    dataset = foz.load_zoo_dataset(
        "sama-coco",
        label_types=["detections"],
        classes=['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    )
    session = fo.launch_app(dataset)
    session.wait()


def create_all_datasets(export_dir: Path) -> None:
    fo.delete_non_persistent_datasets()
    export_dir = Path(export_dir)
    oai.create_open_animal_images_dataset(export_dir, persistence=True)
    oai.convert_open_animal_images_to_rf_detr(export_dir / 'OpenAnimalImages', export_dir / 'OpenAnimalImages_RF-DETR')
    oafi.create_open_animal_face_images_dataset(
        export_dir / 'OpenAnimalImages', export_dir, name='OAFI_full', persistence=True)


if __name__ == '__main__':
    # create_all_datasets(Path('/mnt/data/afarec/data'))
    # create_open_animal_images_dataset(Path('/mnt/data/afarec/data'), max_samples=100)
    # create_open_animal_face_images_dataset(Path('/mnt/data/afarec/data') / 'OpenAnimalImages', Path('/mnt/data/afarec/data'), name='OAFI_full')

    session = fo.launch_app(address='0.0.0.0')
    session.wait(-1)
