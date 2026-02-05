import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image

import deepface
from tqdm import tqdm

from utils import test_mongo_connection


def _get_open_image_mappings() -> dict[str, list[str]]:
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


def create_open_animal_images() -> None:
    for split in ['train', 'validation', 'test']:
        view = load_open_animal_images(split)
        # Export dataset into the yolo format
        view.export(
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            export_dir='/mnt/data/afarec/data/OpenAnimalImages_woBird',
            classes=list(_get_open_image_mappings().keys()),
            split="val" if split == "validation" else split,
            overwrite=False,
        )
        # view.export(
        #     dataset_type=fo.types.COCODetectionDataset,
        #     label_field="ground_truth",
        #     export_dir=f'/mnt/data/afarec/data/OpenAnimalImages_COCO/{split}',
        #     classes=list(_get_open_image_mappings().keys()),
        #     # split="val" if split == "validation" else split,
        #     overwrite=False,
        # )

def convert_open_animal_images_to_rt_detr(input_dir: Path, output_dir: Path) -> None:
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


def load_open_animal_images(split: str) -> fo.DatasetView:
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        label_types=["detections"],
        # classes=['Animal', 'Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken',
        #          'Eagle', 'Owl', 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey',
        #          'Invertebrate', 'Tick', 'Centipede', 'Marine invertebrates', 'Starfish', 'Isopod', 'Squid', 'Lobster',
        #          'Jellyfish', 'Shrimp', 'Crab', 'Insect', 'Bee', 'Beetle', 'Ladybug', 'Ant', 'Moths and butterflies',
        #          'Caterpillar', 'Butterfly', 'Dragonfly', 'Scorpion', 'Worm', 'Spider', 'Oyster', 'Snail', 'Mammal',
        #          'Bat (Animal)', 'Carnivore', 'Bear', 'Brown bear', 'Panda', 'Polar bear', 'Teddy bear', 'Cat', 'Fox',
        #          'Jaguar (Animal)', 'Lynx', 'Red panda', 'Tiger', 'Lion', 'Dog', 'Leopard', 'Cheetah', 'Otter',
        #          'Raccoon', 'Camel', 'Cattle', 'Giraffe', 'Rhinoceros', 'Goat', 'Horse', 'Hamster', 'Kangaroo', 'Koala',
        #          'Mouse', 'Pig', 'Rabbit', 'Squirrel', 'Sheep', 'Zebra', 'Monkey', 'Hippopotamus', 'Deer', 'Elephant',
        #          'Porcupine', 'Hedgehog', 'Bull', 'Antelope', 'Mule', 'Marine mammal', 'Dolphin', 'Whale', 'Sea lion',
        #          'Harbor seal', 'Skunk', 'Alpaca', 'Armadillo', 'Reptile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle',
        #          'Tortoise', 'Sea turtle', 'Crocodile', 'Frog', 'Fish', 'Goldfish', 'Shark', 'Rays and skates',
        #          'Seahorse', 'Shellfish', 'Oyster', 'Lobster', 'Shrimp', 'Crab']
        classes=['Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl',
                 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey', 'Cat', 'Jaguar (Animal)',
                 'Lynx', 'Tiger', 'Lion', 'Leopard', 'Cheetah', 'Dog', 'Fox', 'Goat', 'Horse', 'Mule', 'Hamster',
                 'Mouse', 'Rabbit'],
        split=split,
    )

    new_mappings = _get_open_image_mappings()

    reversed_mapping = {}
    for key, value in new_mappings.items():
        for item in value:
            reversed_mapping[item] = key

    view = dataset.map_labels('ground_truth', reversed_mapping)

    # Filter after only the classes I want to work with
    view = view.filter_labels("ground_truth", F("label").is_in(list(new_mappings.keys())))

    return view


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


def create_oai_anno_files(oai_dateset: fo.DatasetView) -> None:
    new_dataset = fo.Dataset("OpenAnimalImages_anno")

    for sample in oai_dateset:
        img = sample.filepath
        detections = sample.ground_truth.detections

        for det in detections:
            bbox = det.bounding_box
            x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            # Crop the image
            new_img = Image.open(img).crop((x, y, x + width, y + height))
            # cropped_img = fou.crop_image(img, x, y, width, height)

            new_sample = fo.Sample(filepath=cropped_img)
            new_sample["ground_truth"] = fo.Classification(label=det.label)
            new_dataset.add_sample(new_sample)

    # Save the new dataset
    new_dataset.persistent = True

    for split in ['train', 'validation', 'test']:
        # Export dataset into the yolo format
        new_dataset.export(
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            export_dir='/mnt/data/afarec/data/OpenAnimalImages_anno',
            classes=list(_get_open_image_mappings().keys()),
            split="val" if split == "validation" else split,
            overwrite=False,
        )

    print("New dataset created with cropped images!")


if __name__ == '__main__':
    convert_open_animal_images_to_rt_detr(Path('/mnt/data/afarec/data/OpenAnimalImages_woBird'), Path('/mnt/data/afarec/data/OpenAnimalImages_RF-DETR_woBird'))
    # test_mongo_connection()
    # fo.config.dataset_zoo_dir = Path('/mnt/data/afarec/data')
    # fo.config.database_uri = 'mongodb://127.0.0.1:27017'
    # fo.config.database_validation = False
    # create_open_animal_images()
    # data = load_open_animal_images(None)
    # load_coco()
    # print(fo.list_datasets())
    # session = fo.launch_app()
    # session.wait(-1)
