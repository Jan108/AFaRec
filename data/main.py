from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

import oafi, utils, oai


utils.setup()


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


def yunet_stuff():
    data_oafi = fo.load_dataset('OAFI_full')

    work_dir = Path('/mnt/data/afarec/code/face_detection/yunet/work_dir/')
    oafi.import_prediction_from_yunet(
        data_oafi,
        work_dir / f'yunet_pretrained/results',
        f'yunet_pretrained_v1'
    )
    # for cls_name in ['all', 'bird', 'cat', 'cat_like', 'dog', 'dog_like', 'horse_like', 'small_animals', 'pretrained']:
    #     oafi.import_prediction_from_yunet(
    #         data_oafi,
    #         work_dir / f'yunet_{cls_name}/results',
    #         f'yunet_{cls_name}_v1'
    #     )


def create_all_datasets(export_dir: Path) -> None:
    fo.delete_non_persistent_datasets()
    export_dir = Path(export_dir)
    oai.create_open_animal_images_dataset(export_dir, persistence=True)
    oai.convert_open_animal_images_to_rf_detr(export_dir / 'OpenAnimalImages', export_dir / 'OpenAnimalImages_RF-DETR')
    oafi.create_open_animal_face_images_dataset(
        export_dir / 'OpenAnimalImages', export_dir, name='OAFI_full', persistence=True)


if __name__ == '__main__':
    # create_all_datasets(Path('/mnt/data/afarec/data'))
    yunet_stuff()

    session = fo.launch_app(address='0.0.0.0')
    session.wait(-1)
