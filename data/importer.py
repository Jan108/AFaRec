from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import deepface


def create_open_animal_images() -> None:
    for split in ['train', 'validation', 'test']:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
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
                     'Mouse', 'Rabbit']
        )

        # Create mapping of OpenImage classes to my own classes and reduce the number
        bird = ['Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl', 'Duck',
                 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey']
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

        new_mappings = {'bird': bird, 'cat': cat, 'cat_like': cat_like, 'dog': dog, 'dog_like': dog_like, 'horse_like':
            horse_like, 'small_animals': small_animals}

        reversed_mapping = {}
        for key, value in new_mappings.items():
            for item in value:
                reversed_mapping[item] = key

        view = dataset.map_labels('ground_truth', reversed_mapping)

        # Filter after only the classes I want to work with
        view = view.filter_labels("ground_truth", F("label").is_in(list(new_mappings.keys())))
        view.save()

        # Export dataset into the yolo format
        view.export(
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            export_dir='/mnt/data/afarec/data/OpenAnimalImages',
            classes=list(new_mappings.keys()),
            split="val" if split == "validation" else split,
            overwrite=False,
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


if __name__ == '__main__':
    fo.config.dataset_zoo_dir = Path('/mnt/data/afarec/data')
    fo.config.database_uri = 'mongodb://127.0.0.1:27017'
    create_open_animal_images()
    # load_coco()
    # fo.launch_app().wait()
