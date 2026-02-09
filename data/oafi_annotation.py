from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import fiftyone as fo

from creation import load_yolo_dataset_from_disk

app = Flask(__name__)
auth = HTTPBasicAuth()
oafi_dataset: fo.Dataset
timed_out_samples: dict[str, datetime] = {}
include_skipped = False

users = {
    "jan": generate_password_hash("jan-nils-lutz-this-birgit-thomas"),
    "birgit": generate_password_hash("jan-nils-lutz-this-birgit-thomas"),
    "this": generate_password_hash("jan-nils-lutz-this-birgit-thomas"),
    "lutz": generate_password_hash("jan-nils-lutz-this-birgit-thomas"),
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


def bbox_fo_to_fabric(bbox: list[float], height, width) -> list[float]:
    """
    Convert relative bounding box to absolute.

    Args:
        bbox: List or tuple of (x_top_left, y_top_left, width, height) in [0, 1].
        width: Width of the image in pixels.
        height: Height of the image in pixels.

    Returns:
        List of (x_top_left, y_top_left, width, height) in pixels.
    """
    x_rel, y_rel, w_rel, h_rel = bbox
    x_tl = x_rel * width
    y_tl = y_rel * height
    w = w_rel * width
    h = h_rel * height
    return [x_tl, y_tl, w, h]

def bbox_fabric_to_fo(bbox: list[float], height, width) -> list[float]:
    """
    Convert absolute bounding box to relative.

    Args:
        bbox: List or tuple of (x_top_left, y_top_left, width, height) in pixels.
        width: Width of the image in pixels.
        height: Height of the image in pixels.

    Returns:
        List of (x_top_left, y_top_left, width, height) in relative coordinates [0, 1].
    """
    x_tl, y_tl, w, h = bbox
    x_rel = x_tl / width
    y_rel = y_tl / height
    w_rel = w / width
    h_rel = h / height
    return [x_rel, y_rel, w_rel, h_rel]


def update_annotation(img_id: str, no_face: bool, bbox: list[float],
                      img_class: str, skip_img: bool, anno_name: str):
    """

    :param img_id:
    :param no_face:
    :param bbox: need to be [<top-left-x>, <top-left-y>, <width>, <height>]
    :param img_class:
    :param skip_img:
    :param anno_name:
    :return:
    """
    sample = oafi_dataset[img_id]
    sample.tags.append(anno_name)
    if skip_img:
        print(f"Updating annotation for {img_id}: Skipped")
        sample.tags.append('skipped')
        sample.save()
        return

    if no_face:
        print(f"Updating annotation for {img_id}: No face present")
        sample.tags.append('no_face')
    else:
        print(f"Updating annotation for {img_id}: New bbox {bbox}")
        sample["ground_truth"] = fo.Detections(
            detections=[fo.Detection(label=img_class, bounding_box=list(bbox))]
        )

    if 'skipped' in sample.tags:
        sample.tags.remove('skipped')
    sample.tags.append('annotated')
    sample.save()


def get_unlabeled_sample(only_skipped: bool = False) -> fo.Sample | None:
    """
    Get an unlabeled sample from the dataset. Each sample has a timeout of 2min before its returned again.
    Except if there are no other samples left. Returns None when there are no samples left.
    :return: unlabeled sample or None
    """
    unlabeled_samples = oafi_dataset.match_tags('annotated', bool=False)
    if only_skipped:
        unlabeled_samples = unlabeled_samples.match_tags('skipped', bool=True)
    if unlabeled_samples.count() == 0:
        return None
    update_timeout()
    unlabeled_sample: fo.Sample = None
    for sample in unlabeled_samples:
        if sample.id in timed_out_samples:
            continue
        print(f'add {sample.id} to timeout')
        timed_out_samples[sample.id] = datetime.now() + timedelta(seconds=60 * 2)
        unlabeled_sample = sample
        break
    if unlabeled_sample is None:
        unlabeled_sample = unlabeled_samples.first()
        print(f'add {sample.id} to timeout')
        timed_out_samples[unlabeled_sample.id] = datetime.now() + timedelta(seconds=60 * 2)
    unlabeled_sample.compute_metadata()
    return unlabeled_sample


def update_timeout() -> None:
    ready_samples = [sample_id for sample_id, timeout in timed_out_samples.items() if timeout < datetime.now()]
    for sample_id in ready_samples:
        del timed_out_samples[sample_id]
    print(f'Timeout Samples: {timed_out_samples}; removed {ready_samples}')

@app.route('/', methods=['GET', 'POST'])
@auth.login_required
def index():
    if request.method == 'POST':
        img_id = request.form['img_id']
        img_skip = request.form['img_skip'] == 'skipped'
        a_img_height = int(request.form['img_height'])
        a_img_width = int(request.form['img_width'])
        annotated_bbox = bbox_fabric_to_fo(
            [float(request.form['bbox_x']), float(request.form['bbox_y']),
             float(request.form['bbox_w']), float(request.form['bbox_h'])],
            a_img_height, a_img_width
        )
        img_class = request.form['img_class']
        img_no_face = request.form['img_no_face'] == 'checked'
        anno_name = auth.current_user()

        update_annotation(img_id, bool(img_no_face), annotated_bbox, img_class, img_skip, anno_name)

        return redirect(url_for('index'))

    # Get unlabeled Sample and render annotation page
    unlabeled_sample = get_unlabeled_sample(only_skipped=include_skipped)
    if unlabeled_sample is None:
        if include_skipped and oafi_dataset.match_tags('skipped', bool=True).count() == 0:
            return 'Done! Some images where skipped.'
        return 'Done! ... for now'

    # Compute / Get relevant information
    filter_tags = ('annotated', 'skipped')
    if include_skipped:
        filter_tags = 'annotated'
    count_labeled = oafi_dataset.match_tags(filter_tags, bool=True).count()
    count_unlabeled = oafi_dataset.match_tags(filter_tags, bool=False).count()
    count_skipped = oafi_dataset.match_tags('skipped', bool=True).count()
    pct_labeled = round((count_labeled / oafi_dataset.count()) * 100, 2)
    img_id = unlabeled_sample.id
    img_height = unlabeled_sample.metadata['height']
    img_width = unlabeled_sample.metadata['width']
    img_bbox = [0, 0, 1, 1]
    img_class = 'None'
    possible_classes = ['cat', 'dog', 'cat_like', 'dog_like', 'bird', 'horse_like', 'small_animal']
    detections = unlabeled_sample.ground_truth.detections
    if len(detections) > 0:
        img_bbox = detections[0].bounding_box
        img_class = detections[0].label
        print(img_class)
        possible_classes.remove(img_class)
    return render_template('annotator.html',
                           img_id=img_id, img_bbox=bbox_fo_to_fabric(img_bbox, img_height, img_width), img_class=img_class,
                           img_height=img_height, img_width=img_width,
                           count_labeled=count_labeled, count_unlabeled=count_unlabeled,
                           pct_labeled=pct_labeled, count_skipped=count_skipped,
                           possible_classes=possible_classes)

@app.route('/images/<path:img_id>')
@auth.login_required
def serve_image(img_id):
    try:
        sample = oafi_dataset[img_id]
        return send_file(sample.filepath)
    except KeyError:
        return 404

if __name__ == '__main__':
    oafi_dataset = load_yolo_dataset_from_disk(
        Path('/mnt/data/afarec/data/OpenAnimalFaceImages'), 'train', max_samples=2000,
        persistence=True, name='Anno-OAFI-2000', unified_label_distribution=True
    )
    # Use to run skipped samples
    # include_skipped = True
    app.run(debug=False, host='0.0.0.0')
