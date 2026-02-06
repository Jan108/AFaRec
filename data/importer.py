import shutil
import tempfile
from pathlib import Path
from typing import Literal

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image

import deepface
from tqdm import tqdm

from utils import test_mongo_connection, convert_fo_bbox_to_absolute


def _get_open_image_mappings() -> dict[str, list[str]]:
    """
    Definition of the mapping from OpenImagesV7 to OpenAnimalImages dataset.

    :return:
    """
    # Create mapping of OpenImage classes to my own classes and reduce the number
    # removed Bird from mapping, because of amount of images (50k) and quality of label
    bird = ['Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl',
            'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey']
    # carnivore = ['Bear', 'Cat', 'Fox', 'Jaguar (Animal)', 'Lynx', 'Red panda', 'Tiger', 'Lion', 'Dog', 'Leopard',
    #              'Cheetah', 'Otter', 'Raccoon']
    cat_like = ['Jaguar (Animal)', 'Lynx', 'Tiger', 'Lion', 'Leopard', 'Cheetah']
    dog_like = ['Fox']
    cat = ['Cat']
    dog = ['Dog']
    # mammals = ['Bat (Animal)', 'Camel', 'Cattle', 'Giraffe', 'Rhinoceros', 'Goat', 'Horse', 'Hamster', 'Kangaroo',
    #            'Koala', 'Mouse', 'Pig', 'Rabbit', 'Squirrel', 'Sheep', 'Zebra', 'Monkey', 'Hippopotamus', 'Deer',
    #            'Elephant', 'Porcupine', 'Hedgehog', 'Bull', 'Antelope', 'Mule', 'Skunk', 'Alpaca', 'Armadillo']
    horse_like = ['Goat', 'Horse', 'Mule']
    small_animals = ['Hamster', 'Mouse', 'Rabbit']

    return {'bird': bird, 'cat': cat, 'cat_like': cat_like, 'dog': dog, 'dog_like': dog_like, 'horse_like':
        horse_like, 'small_animals': small_animals}


def create_open_animal_images_dataset(output_dir: Path, export_yolo: bool = True,
                                      export_coco: bool = False, max_samples: int = None) -> None:
    """
    Creates the OpenAnimalImages dataset based on the OpenImagesV7 dataset.
    It filters for wanted classes and relabels the samples with the new class.

    :param output_dir: Directory to save the OpenAnimalImages dataset.
    :param export_yolo: Should the dataset be exported in YOLO format?
    :param export_coco: Should the dataset be exported in COCO format?
    :param max_samples: [Optional] Maximum number of samples to load per split
    :return: None
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError('Output is not a directory.')

    save_path_yolo = Path(output_dir) / 'OpenAnimalImages'
    if export_yolo:
        if save_path_yolo.exists() and save_path_yolo.is_dir() and any(save_path_yolo.iterdir()):
            raise FileExistsError(f'The directory {save_path_yolo} already exists and is not empty.')
        save_path_yolo.mkdir(parents=True, exist_ok=True)


    save_path_coco = Path(output_dir) / 'OpenAnimalImagesCOCO'
    if export_coco:
        if save_path_coco.exists() and save_path_coco.is_dir() and any(save_path_coco.iterdir()):
            raise FileExistsError(f'The directory {save_path_coco} already exists and is not empty.')
        save_path_coco.mkdir(parents=True, exist_ok=True)


    for split in ['train', 'validation', 'test']:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            label_types=["detections"],
            classes=['Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle',
                     'Owl',
                     'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey', 'Cat',
                     'Jaguar (Animal)',
                     'Lynx', 'Tiger', 'Lion', 'Leopard', 'Cheetah', 'Dog', 'Fox', 'Goat', 'Horse', 'Mule', 'Hamster',
                     'Mouse', 'Rabbit'],
            split=split,
            max_samples=max_samples,
        )

        new_mappings = _get_open_image_mappings()

        reversed_mapping = {}
        for key, value in new_mappings.items():
            for item in value:
                reversed_mapping[item] = key

        view = dataset.map_labels('ground_truth', reversed_mapping)

        # Filter after only the classes I want to work with
        view = view.filter_labels("ground_truth", F("label").is_in(list(new_mappings.keys())))

        # Export dataset into the yolo format
        if export_yolo:
            view.export(
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                export_dir=str(save_path_yolo),
                classes=list(_get_open_image_mappings().keys()),
                split="val" if split == "validation" else split,
                overwrite=False,
            )

        # Export dataset into the coco format
        if export_coco:
            view.export(
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_dir=str(save_path_coco / split),
                classes=list(_get_open_image_mappings().keys()),
                overwrite=False,
            )


def load_yolo_dataset_from_disk(save_dir: Path, split: Literal['train', 'validation', 'test'], max_samples: int = None) -> fo.Dataset:
    """
    Loads a YOLO dataset from disk.

    :param save_dir: Directory where the dataset is saved in YOLO format
    :param split: One of 'train', 'validation', 'test' which defines the split to load.
    :param max_samples: [Optional] Maximum number of samples to load
    :return: Loaded fo.Dataset
    """
    save_dir = Path(save_dir)
    if not save_dir.exists() or not save_dir.is_dir():
        raise FileNotFoundError(f'The directory {save_dir} does not exist.')
    if not any(save_dir.iterdir()):
        raise ValueError(f'The directory {save_dir} is empty.')
    if not (save_dir / 'dataset.yaml').exists():
        raise ValueError(f'The directory {save_dir} is not a valid YOLO dataset: dataset.yaml not found')

    dataset = fo.Dataset.from_dir(
        dataset_dir=save_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        max_samples=max_samples,
        seed=42,
        name=f'{save_dir.name}_{split}',
        split="val" if split == "validation" else split,
    )
    return dataset


def convert_open_animal_images_to_rt_detr(input_dir: Path, output_dir: Path) -> None:
    """
    Converts the exported OpenAnimalImages YOLO dataset into a format so that RF-DETR can work with it.

    :param input_dir: Directory of the OpenAnimalImage dataset in YOLO format.
    :param output_dir: Directory to export the converted dataset to.
    :return: None
    """

    splits = [('train', 'train'), ('val', 'valid'), ('test', 'test')]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy dataset YML
    dataset_yml = input_dir / 'dataset.yaml'
    shutil.copyfile(dataset_yml, output_dir / 'data.yaml')

    # Copy and convert images
    for split in splits:
        print(f'Converting {split[0]} images to RT DETR')
        Path(output_dir / split[1] / 'images').mkdir(parents=True, exist_ok=True)
        for file_name in tqdm(Path(input_dir / 'images' / split[0]).glob('*.jpg')):
            Image.open(file_name).convert('RGB').save(
                output_dir / split[1] / 'images' / file_name.name
            )

    # Copy labels
    for split in splits:
        print(f'Converting {split[0]} labels to RT DETR')
        Path(output_dir / split[1] / 'labels').mkdir(parents=True, exist_ok=True)
        for file_name in tqdm(Path(input_dir / 'labels' / split[0]).glob('*.txt')):
            shutil.copyfile(
                file_name,
                output_dir / split[1] / 'labels' / file_name.name
            )


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


def create_open_animal_face_images_dataset(oai_dir: Path, export_dir: Path, max_samples: int = None) -> None:
    """
    Creates the OpenAnimalFaceImages dataset based on the OpenAnimalImages dataset.
    Crops each sample to the detected animal and exports it afterwards.

    It creates a tmp directory in the export_dir to export the dataset.

    :param oai_dir: OpenAnimalImages dataset directory to load from.
    :param export_dir: Directory to save the OpenAnimalImages dataset.
    :param max_samples: [Optional] Maximum number of samples to load.
    :return: None
    """
    if export_dir.exists() and not export_dir.is_dir():
        raise ValueError(f'Output {export_dir} is not a directory.')

    save_path = Path(export_dir) / 'OpenAnimalFaceImages'
    if save_path.exists() and save_path.is_dir() and any(save_path.iterdir()):
        raise FileExistsError(f'The directory {save_path} already exists and is not empty.')
    save_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=export_dir, prefix='Temp_OpenAnimalFaceImages') as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for split in ['train', 'validation', 'test']:
            oai_dataset = load_yolo_dataset_from_disk(oai_dir, split, max_samples=max_samples)
            new_dataset = fo.Dataset(f"OpenAnimalFaceImages_{split}")

            # Iterate through all samples, adjust them and add to the new dataset
            for sample in tqdm(oai_dataset, desc=f'Crop images for split {split} to new sizes', total=len(oai_dataset)):
                img_path = Path(sample.filepath)

                for i, det in enumerate(sample.ground_truth.detections):
                    new_img_path = tmp_dir / f'{img_path.stem}_{i}{img_path.suffix}'
                    img = Image.open(img_path)

                    bbox = convert_fo_bbox_to_absolute(det.bounding_box, img.height, img.width)
                    img.crop(bbox).save(new_img_path)

                    new_sample = fo.Sample(filepath=new_img_path)
                    new_sample["ground_truth"] = fo.Classification(label=det.label)
                    new_dataset.add_sample(new_sample)

            # Export dataset into the yolo format
            new_dataset.export(
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                export_dir=str(save_path),
                classes=list(_get_open_image_mappings().keys()),
                split="val" if split == "validation" else split,
                overwrite=False,
            )

            print(f'Split {split} converted.')


if __name__ == '__main__':
    # convert_open_animal_images_to_rt_detr(Path('/mnt/data/afarec/data/OpenAnimalImages_woBird'), Path('/mnt/data/afarec/data/OpenAnimalImages_RF-DETR_woBird'))
    test_mongo_connection()
    fo.config.dataset_zoo_dir = Path('/mnt/data/afarec/data')
    fo.config.database_uri = 'mongodb://127.0.0.1:27017'
    fo.config.database_validation = False
    # create_open_animal_images_dataset()
    # data = load_yolo_dataset_from_disk(Path('/mnt/data/afarec/data/OpenAnimalImages_woBird'), max_samples=30)
    create_open_animal_face_images_dataset(
        Path('/mnt/data/afarec/data/OpenAnimalImages_woBird'),
        Path('/mnt/data/afarec/data'), max_samples=10)
    # load_coco()
    # print(fo.list_datasets())
    # session = fo.launch_app()
    # session.wait(-1)
