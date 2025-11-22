from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

import deepface


def load_open_image() -> None:
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        label_types=["detections", "classifications"],
        classes=['Animal', 'Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken',
                 'Eagle', 'Owl', 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey',
                 'Invertebrate', 'Tick', 'Centipede', 'Marine invertebrates', 'Starfish', 'Isopod', 'Squid', 'Lobster',
                 'Jellyfish', 'Shrimp', 'Crab', 'Insect', 'Bee', 'Beetle', 'Ladybug', 'Ant', 'Moths and butterflies',
                 'Caterpillar', 'Butterfly', 'Dragonfly', 'Scorpion', 'Worm', 'Spider', 'Oyster', 'Snail', 'Mammal',
                 'Bat (Animal)', 'Carnivore', 'Bear', 'Brown bear', 'Panda', 'Polar bear', 'Teddy bear', 'Cat', 'Fox',
                 'Jaguar (Animal)', 'Lynx', 'Red panda', 'Tiger', 'Lion', 'Dog', 'Leopard', 'Cheetah', 'Otter',
                 'Raccoon', 'Camel', 'Cattle', 'Giraffe', 'Rhinoceros', 'Goat', 'Horse', 'Hamster', 'Kangaroo', 'Koala',
                 'Mouse', 'Pig', 'Rabbit', 'Squirrel', 'Sheep', 'Zebra', 'Monkey', 'Hippopotamus', 'Deer', 'Elephant',
                 'Porcupine', 'Hedgehog', 'Bull', 'Antelope', 'Mule', 'Marine mammal', 'Dolphin', 'Whale', 'Sea lion',
                 'Harbor seal', 'Skunk', 'Alpaca', 'Armadillo', 'Reptile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle',
                 'Tortoise', 'Sea turtle', 'Crocodile', 'Frog', 'Fish', 'Goldfish', 'Shark', 'Rays and skates',
                 'Seahorse', 'Shellfish', 'Oyster', 'Lobster', 'Shrimp', 'Crab']
    )
    session = fo.launch_app(dataset)
    session.wait()
    dataset.


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
    # load_open_image()
    load_coco()
    # fo.launch_app().wait()
