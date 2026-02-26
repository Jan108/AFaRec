import tempfile
from pathlib import Path

import fiftyone as fo
from PIL import Image
from fiftyone import ViewField as F
from tqdm import tqdm

from . import oai
from . import utils


def create_open_animal_face_images_dataset(oai_dir: Path, export_dir: Path, max_samples: int = None,
                                           name: str = None, persistence: bool = False, ) -> None:
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

        oai_dataset = utils.load_yolo_dataset_from_disk(oai_dir, max_samples=max_samples)
        new_dataset = fo.Dataset(name, persistent=persistence)

        # Iterate through all samples, adjust them and add to the new dataset
        for sample in tqdm(oai_dataset, desc=f'Crop images to new sizes', total=len(oai_dataset)):
            img_path = Path(sample.filepath)

            for i, det in enumerate(sample.ground_truth.detections):
                new_img_path = tmp_dir / f'{img_path.stem}_{i}{img_path.suffix}'
                img = Image.open(img_path)

                bbox = utils.convert_xyhwn_to_xyxy(det.bounding_box, img.height, img.width)
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
                classes=list(oai.get_open_image_mappings().keys()),
                split="val" if split == "validation" else split,
                overwrite=False,
            )

            print(f'Split {split} exported.')


def export_labels_to_yunet(dataset: fo.Dataset) -> None:
    """
    Export labels into txt file to use with Yunet.
    :param dataset: OpenAnimalFaceImages dataset
    :return: None
    """
    export_dir = Path(dataset.first().filepath).parent.parent.parent / 'labels_yunet'
    export_dir.mkdir(parents=True, exist_ok=True)
    if any(export_dir.iterdir()):
        raise FileExistsError(f'The directory {export_dir} already exists and is not empty.')

    for split in ['train', 'validation', 'test']:
        view = dataset.match_tags(split).match_tags('annotated').match_tags('no_face', bool=False)
        for classes in ['all', 'bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals']:
            cur_file = export_dir / f'labels_{classes}_{split}.txt'
            if classes == 'all':
                cls_view = view
            else:
                cls_view = view.filter_labels('ground_truth', F('label').is_in([classes]))
            with cur_file.open(mode='w') as f:
                for sample in cls_view:
                    sample: fo.Sample
                    sample.compute_metadata()
                    img_height = sample.metadata.height
                    img_width = sample.metadata.width
                    lines = [f'# {sample.filename} {img_width} {img_height}\n']

                    for det in sample.ground_truth.detections:
                        bbox = utils.convert_xyhwn_to_xyxy(det.bounding_box, img_height, img_width)
                        lines.append(
                            f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n')

                    f.writelines(lines)
                f.flush()


def import_prediction_from_yunet(dataset: fo.Dataset, import_dir: Path, field_name: str) -> None:
    """
    Import the predictions from yunet into the oafi dataset.
    :param dataset: the oafi dataset
    :param import_dir: the dir where the predictions are saved
    :param field_name: Name of the field to save the predictions in
    :return: None
    """
    dataset.compute_metadata()
    pred_dir = Path(import_dir)

    for sample in dataset:
        img_name = Path(sample.filepath).stem
        txt_path = pred_dir / f"{img_name}.txt"
        if not txt_path.exists():
            continue

        detections = []
        with txt_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                x, y, w, h, conf = map(float, parts)

                w_img, h_img = sample.metadata.width, sample.metadata.height
                nx = x / w_img
                ny = y / h_img
                nw = w / w_img
                nh = h / h_img
                bbox = [nx, ny, nw, nh]

                detections.append(
                    fo.Detection(
                        label='face',
                        bounding_box=bbox,
                        confidence=conf,
                    )
                )

        sample[field_name] = fo.Detections(detections=detections)
        sample.save()


def fix_copy_anno_to_oafi() -> None:
    """
    One use function to fix the import of the old dataformat.
    :return: None
    """
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


def print_matrix(dataset: fo.Dataset, fix_matrix: bool = False) -> None:
    """
    Print the annotation matrix for the oafi dataset.
    If fix_matrix is True, new samples per split and class will be tagged
    with 'anno_needed' so that the distribution matches 70-15-15.
    :return:
    """
    oafi = dataset

    counts = dict()
    labels = list(oafi.count_values('ground_truth.detections.label').keys())
    sum_all_anno = 0
    sum_all_useful = 0
    print(
        f'{'label':>14s} |  {'train':^20s}  :  {'validation':^20s}  :  {'test':^20s}  |  {'sum':^5s}  :  {'use':^5s}  :  {'all':^5s}')
    for label in labels:
        label_view = oafi.filter_labels('ground_truth', F('label').is_in([label]))
        label_dict = dict()
        line = []
        sum_anno = 0
        sum_useful = 0
        sum_all = 0
        for split in ['train', 'validation', 'test']:
            ls_tags = label_view.match_tags(split).count_sample_tags()
            anno_needed = ls_tags.get('anno_needed', 0)
            annotated = ls_tags.get('annotated', 0)
            no_face = ls_tags.get('no_face', 0)
            s_count = ls_tags.get(split, 0)
            label_dict[split] = (anno_needed, annotated, no_face, s_count)
            sum_anno += anno_needed + annotated
            sum_useful += anno_needed + annotated - no_face
            sum_all += s_count
            line.append(f'{anno_needed:4d} {annotated:4d} {no_face:4d} {s_count:5d}')
        print(
            f'{label:>14s} |  {line[0]}  :  {line[1]}  :  {line[2]}  |  {sum_anno:5d}  :  {sum_useful:5d}  :  {sum_all:5d}')
        counts[label] = label_dict
        sum_all_anno += sum_anno
        sum_all_useful += sum_useful
    print(f'{' ':^93s}{sum_all_anno:5d}  :  {sum_all_useful:5d}')

    if not fix_matrix:
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
            missing = take_by_split[split] - amount
            if missing <= 0:
                oafi_label_view = oafi.match_tags(split).filter_labels("ground_truth", F("label").is_in([label]))
                face_split_view = oafi_label_view.match_tags('anno_needed', bool=True)
                removed_samples = face_split_view.take(missing, seed=42)
                removed_samples.untag_samples('anno_needed')
            else:
                label_view.match_tags(split).take(missing, seed=42).tag_samples('anno_needed')
