import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.iou as foui
import fiftyone.utils.random as four
import fiftyone.zoo as foz
from PIL import Image
from fiftyone import ViewField as F
from tqdm import tqdm


def get_open_image_mappings() -> dict[str, list[str]]:
    """
    Definition of the mapping from OpenImagesV7 to OpenAnimalImages dataset.

    :return: Dictionary with key as the class in OpenAnimalImages and the value is a list of Classes from OpenImageV7
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


def get_coco_mapping() -> dict[str, list[str]]:
    """
    Definition of the mapping from COCO to OpenAnimalImages dataset.

    :return: Dictionary with key as the class in OpenAnimalImages and the value is a list of Classes from COCO
    """
    bird = ['bird']
    cat_like = []
    dog_like = []
    cat = ['cat']
    dog = ['dog']
    horse_like = ['horse']
    small_animals = []

    return {'bird': bird, 'cat': cat, 'cat_like': cat_like, 'dog': dog, 'dog_like': dog_like, 'horse_like':
        horse_like, 'small_animals': small_animals}


def create_open_animal_images_dataset(output_dir: Path, persistence: bool = False, name: str = None,
                                      train_split: float = 0.7, val_split: float = 0.15, max_samples: int = None,
                                      export_yolo: bool = True, export_coco: bool = False) -> fo.Dataset:
    """
    Creates the OpenAnimalImages dataset based on the OpenImagesV7 dataset.
    It filters for wanted classes and relabels the samples with the new class.

    :param output_dir: Directory to save the OpenAnimalImages dataset.
    :param persistence: [Optional] Should the dataset be persisted in the FiftyOne DB?
    :param name: Name of the dataset. If no name is given, the name is generated based on OpenAnimalImages-{max_samples}
    :param train_split: Proportion of the dataset to include in the training set.
    :param val_split: Proportion of the dataset to include in the validation set.
    :param max_samples: [Optional] Maximum number of samples to load per split
    :param export_yolo: Should the dataset be exported in YOLO format?
    :param export_coco: Should the dataset be exported in COCO format?
    :return: created dataset
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError('Output is not a directory.')

    if train_split + val_split >= 1.0:
        raise ValueError('Train and val split cannot be greater or equal than 1.0.')

    if name is None:
        name = f'OpenAnimalImages'
        if max_samples is not None:
            name += f"-{max_samples}"

    if name in fo.list_datasets():
        raise ValueError(f'The dataset {name} is already persisted in the FiftyOne DB.')

    save_path_yolo = Path(output_dir) / name
    if export_yolo:
        if save_path_yolo.exists() and save_path_yolo.is_dir() and any(save_path_yolo.iterdir()):
            raise FileExistsError(f'The directory {save_path_yolo} already exists and is not empty.')
        save_path_yolo.mkdir(parents=True, exist_ok=True)

    save_path_coco = Path(output_dir) / f'{name}_COCO'
    if export_coco:
        if save_path_coco.exists() and save_path_coco.is_dir() and any(save_path_coco.iterdir()):
            raise FileExistsError(f'The directory {save_path_coco} already exists and is not empty.')
        save_path_coco.mkdir(parents=True, exist_ok=True)

    dataset = fo.Dataset(name=name, persistent=persistence)

    dataset_oi = foz.load_zoo_dataset(
        "open-images-v7",
        label_types=["detections"],
        classes=['Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl',
                 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey', 'Cat', 'Jaguar (Animal)',
                 'Lynx', 'Tiger', 'Lion', 'Leopard', 'Cheetah', 'Dog', 'Fox', 'Goat', 'Horse', 'Mule', 'Hamster',
                 'Mouse', 'Rabbit'],
        max_samples=max_samples,
    )

    new_mappings = get_open_image_mappings()

    reversed_mapping = {}
    for key, value in new_mappings.items():
        for item in value:
            reversed_mapping[item] = key

    view = dataset_oi.map_labels('ground_truth', reversed_mapping)

    # Filter after only the classes I want to work with
    view = view.filter_labels("ground_truth", F("label").is_in(list(new_mappings.keys())))

    # Remove duplicates
    dup_ids = foui.find_duplicates(
        view, "ground_truth", iou_thresh=0.5, classwise=True
    )
    view = view.exclude_labels(ids=dup_ids)

    dataset.add_samples(view)

    # Create a splits
    dataset.untag_samples(["train", "test", 'validation'])
    for class_name in dataset.count_values('ground_truth.detections.label'):
        nonlabeled_data = dataset.match_tags(["train", "test", 'validation'], bool=False)
        four.random_split(
            nonlabeled_data.filter_labels("ground_truth", F("label").is_in([class_name])),
            {"train": train_split, "test": 1 - train_split - val_split, "validation": val_split},
            seed=42,
        )

    for split in ['train', 'validation', 'test']:
        split_view = dataset.match_tags(split, bool=True)
        # Export dataset into the yolo format
        if export_yolo:
            split_view.export(
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                export_dir=str(save_path_yolo),
                classes=list(get_open_image_mappings().keys()),
                split="val" if split == "validation" else split,
                overwrite=False,
            )

        # Export dataset into the coco format
        if export_coco:
            split_view.export(
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                export_dir=str(save_path_coco / split),
                classes=list(get_open_image_mappings().keys()),
                overwrite=False,
            )

    return dataset


def convert_open_animal_images_to_rf_detr(input_dir: Path, output_dir: Path) -> None:
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
