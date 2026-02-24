import shutil
import tempfile
from pathlib import Path
from typing import Literal

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four
from fiftyone import ViewField as F
import fiftyone.utils.iou as foui
from PIL import Image

from tqdm import tqdm

try:
    from data import utils
except ModuleNotFoundError:
    import utils


# Setup FiftyOne
utils.test_mongo_connection()
fo.config.dataset_zoo_dir = Path('/mnt/data/afarec/data')
fo.config.database_uri = 'mongodb://127.0.0.1:27017'
fo.config.database_validation = False

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
            {"train": train_split, "test": 1-train_split-val_split, "validation": val_split},
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
                dataset.add_samples(label_view.take(int(max_samples/len(labels)), seed=42))
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


def create_open_animal_face_images_dataset(oai_dir: Path, export_dir: Path, max_samples: int = None,
                                           name: str = None, persistence: bool = False,) -> None:
    """
    Creates the OpenAnimalFaceImages dataset based on the OpenAnimalImages dataset.
    Crops each sample to the detected animal and exports it afterwards.

    It creates a tmp directory in the export_dir to export the dataset.

    :param oai_dir: OpenAnimalImages dataset directory to load from.
    :param export_dir: Directory to save the OpenAnimalImages dataset.
    :param max_samples: [Optional] Maximum number of samples to load.
    :param persistence: [Optional] Should the dataset be persisted in the FiftyOne DB?
    :param name: [Optional] Name of the dataset
    :return: None
    """
    if export_dir.exists() and not export_dir.is_dir():
        raise ValueError(f'Output {export_dir} is not a directory.')

    if name is None:
        name = 'OpenAnimalFaceImages'
        if max_samples is not None:
            name += f'-{max_samples}'

    if name in fo.list_datasets():
        raise ValueError(f'The dataset {name} is already persisted in the FiftyOne DB.')

    save_path = Path(export_dir) / name
    if save_path.exists() and save_path.is_dir() and any(save_path.iterdir()):
        raise FileExistsError(f'The directory {save_path} already exists and is not empty.')
    save_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=export_dir, prefix='Temp_OpenAnimalFaceImages') as tmp_dir:
        tmp_dir = Path(tmp_dir)

        oai_dataset = load_yolo_dataset_from_disk(oai_dir, max_samples=max_samples)
        new_dataset = fo.Dataset(name, persistent=persistence)

        # Iterate through all samples, adjust them and add to the new dataset
        for sample in tqdm(oai_dataset, desc=f'Crop images to new sizes', total=len(oai_dataset)):
            img_path = Path(sample.filepath)

            for i, det in enumerate(sample.ground_truth.detections):
                new_img_path = tmp_dir / f'{img_path.stem}_{i}{img_path.suffix}'
                img = Image.open(img_path)

                bbox = utils.convert_fo_bbox_to_absolute(det.bounding_box, img.height, img.width)
                img.crop(bbox).save(new_img_path)

                new_sample = fo.Sample(filepath=new_img_path)
                new_sample.tags.extend(sample.tags)
                new_sample["ground_truth"] = fo.Classification(label=det.label)
                new_dataset.add_sample(new_sample)

        print('Images converted, now exporting ...')
        for split in ['train', 'validation', 'test']:
            # Export dataset into the yolo format
            new_dataset.match_tags(split).export(
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                export_dir=str(save_path),
                classes=list(get_open_image_mappings().keys()),
                split="val" if split == "validation" else split,
                overwrite=False,
            )

            print(f'Split {split} exported.')


def export_labels_to_yunet(dataset: fo.Dataset) -> None:
    """
    Export labels into txt file to use with Yunet.
    :param dataset: OpenAnimalFaceImages dataset
    :param filter_tags: List of tags to filter. Default ['annotated']
    :return: None
    """
    export_dir = Path(dataset.first().filepath).parent.parent.parent / 'labels_yunet'
    export_dir.mkdir(parents=True, exist_ok=True)
    if any(export_dir.iterdir()):
        raise FileExistsError(f'The directory {export_dir} already exists and is not empty.')

    for split in ['train', 'validation', 'test']:
        view = dataset.match_tags(split).match_tags('annotated').match_tags('no_face', bool=False)
        cur_file = export_dir / f'labels_{split}.txt'
        with cur_file.open(mode='w') as f:
            for sample in view:
                sample: fo.Sample
                sample.compute_metadata()
                img_height = sample.metadata.height
                img_width = sample.metadata.width
                lines = [f'# {sample.filename} {img_width} {img_height}\n']

                for det in sample.ground_truth.detections:
                    bbox = utils.convert_fo_bbox_to_absolute(det.bounding_box, img_height, img_width)
                    lines.append(f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n')

                f.writelines(lines)
            f.flush()


def create_all_datasets(export_dir: Path) -> None:
    fo.delete_non_persistent_datasets()
    export_dir = Path(export_dir)
    create_open_animal_images_dataset(export_dir, persistence=True)
    convert_open_animal_images_to_rf_detr(export_dir / 'OpenAnimalImages', export_dir / 'OpenAnimalImages_RF-DETR')
    create_open_animal_face_images_dataset(export_dir / 'OpenAnimalImages', export_dir, name='OAFI_full', persistence=True)


def fix_copy_anno_to_oafi() -> None:
    anno_oafi = fo.load_dataset('Anno-OAFI-2000')
    oafi = fo.load_dataset('OAFI_full')

    filenames = oafi.values('filepath')
    prefix_counts = dict()
    for n in filenames:
        k = Path(n).name.split('_')[0]
        if k in prefix_counts.keys():
            prefix_counts[k].append(n)
        else:
            prefix_counts[k] = [n]

    for anno_sample in anno_oafi:
        anno_sample.compute_metadata()
        anno_prefix = anno_sample.filename.split('_')[0]
        oafi_faces = prefix_counts[anno_prefix]
        matched = False
        for oafi_face in oafi_faces:
            oafi_sample = oafi[oafi_face]
            oafi_sample.compute_metadata()
            if oafi_sample.metadata == anno_sample.metadata:
                print(f'Found match {oafi_face} : {anno_sample.filename}')
                matched = True
                if 'annotated' in anno_sample.tags:
                    oafi_sample.tags.extend(anno_sample.tags)
                    oafi_sample.ground_truth = anno_sample.ground_truth
                    anno_time = anno_sample['annotation_time']
                    if anno_time is None:
                        anno_time = -1
                    oafi_sample['annotation_time'] = anno_time
                else:
                    oafi_sample.tags.append('anno_needed')
                oafi_sample.save()
                break
        if not matched:
            print(f'{anno_sample.filepath} does not have a match in OAFI...')


def print_matrix():
    oafi = fo.load_dataset('OAFI_full')

    counts = dict()
    labels = list(oafi.count_values('ground_truth.detections.label').keys())
    print(f'{'label':>14s} |  {'train':^20s}  :  {'validation':^20s}  :  {'test':^20s}  |  {'sum':^5s}  :  {'all':^5s}')
    for label in labels:
        label_view = oafi.filter_labels('ground_truth', F('label').is_in([label]))
        label_dict = dict()
        line = []
        sum_anno = 0
        sum_all = 0
        for split in ['train', 'validation', 'test']:
            ls_tags = label_view.match_tags(split).count_sample_tags()
            anno_needed = ls_tags.get('anno_needed', 0)
            annotated = ls_tags.get('annotated', 0)
            no_face = ls_tags.get('no_face', 0)
            s_count = ls_tags.get(split, 0)
            label_dict[split] = (anno_needed, annotated, no_face, s_count)
            sum_anno += anno_needed+annotated-no_face
            sum_all += s_count
            line.append(f'{anno_needed:4d} {annotated:4d} {no_face:4d} {s_count:5d}')
        print(f'{label:>14s} |  {line[0]}  :  {line[1]}  :  {line[2]}  |  {sum_anno:5d}  :  {sum_all:5d}')
        counts[label] = label_dict

    i = input('fix matrix?')
    if not i == 'yes':
        return

    take_by_split = {
        'train': 210,
        'validation': 45,
        'test': 45,
    }
    new_samples = oafi.match_tags(('annotated', 'anno_needed'), bool=False)
    for label in labels:
        label_view = new_samples.filter_labels("ground_truth", F("label").is_in([label]))
        for split in ['train', 'validation', 'test']:
            amount = counts[label][split][0] + counts[label][split][1]
            missing = take_by_split[split]-amount
            if missing <= 0:
                oafi_label_view = oafi.match_tags(split).filter_labels("ground_truth", F("label").is_in([label]))
                face_split_view = oafi_label_view.match_tags('anno_needed', bool=True)
                removed_samples = face_split_view.take(missing, seed=42)
                removed_samples.untag_samples('anno_needed')
            else:
                label_view.match_tags(split).take(missing, seed=42).tag_samples('anno_needed')


if __name__ == '__main__':
    # create_all_datasets(Path('/mnt/data/afarec/data'))
    # create_open_animal_images_dataset(Path('/mnt/data/afarec/data'), max_samples=100)
    # create_open_animal_face_images_dataset(Path('/mnt/data/afarec/data') / 'OpenAnimalImages', Path('/mnt/data/afarec/data'), name='OAFI_full')

    session = fo.launch_app(address='0.0.0.0')
    session.wait(-1)
