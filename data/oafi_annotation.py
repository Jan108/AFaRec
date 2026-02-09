from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_file
import fiftyone as fo

from creation import load_yolo_dataset_from_disk

app = Flask(__name__)
oafi_dataset: fo.Dataset


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


def update_annotation(img_id: str, no_face: bool, bbox: list[float], img_class: str):
    """

    :param img_id:
    :param no_face:
    :param bbox: need to be [<top-left-x>, <top-left-y>, <width>, <height>]
    :param img_class:
    :return:
    """

    print(f"Updating annotation for {img_id} with bbox: {bbox}")

    sample = oafi_dataset[img_id]

    if no_face:
        sample.tags.append('no_face')
    else:
        sample["ground_truth"] = fo.Detections(
            detections=[fo.Detection(label=img_class, bounding_box=list(bbox))]
        )

    sample.tags.append('annotated')
    sample.save()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        a_img_height = int(request.form['img_height'])
        a_img_width = int(request.form['img_width'])
        annotated_bbox = bbox_fabric_to_fo(
            [float(request.form['bbox_x']), float(request.form['bbox_y']),
             float(request.form['bbox_w']), float(request.form['bbox_h'])],
            a_img_height, a_img_width
        )
        img_id = request.form['img_id']
        img_class = request.form['img_class']
        img_no_face = 'img_no_face' in request.form

        update_annotation(img_id, bool(img_no_face), annotated_bbox, img_class)

        return redirect(url_for('index'))

    # Get unlabeled Sample and render annotation page
    unlabeled_samples = oafi_dataset.match_tags('annotated', bool=False)
    if unlabeled_samples.count() == 0:
        return 'Done!!'
    unlabeled_sample: fo.Sample = unlabeled_samples.first()
    unlabeled_sample.compute_metadata()
    count_labeled = oafi_dataset.match_tags('annotated', bool=True).count()
    count_unlabeled = oafi_dataset.match_tags('annotated', bool=False).count()
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
                           pct_labeled=pct_labeled,
                           possible_classes=possible_classes)

@app.route('/images/<path:img_id>')
def serve_image(img_id):
    try:
        sample = oafi_dataset[img_id]
        return send_file(sample.filepath)
    except KeyError:
        return 404

if __name__ == '__main__':
    oafi_dataset = load_yolo_dataset_from_disk(
        Path('/mnt/data/afarec/data/OpenAnimalFaceImages'), 'train', max_samples=350,
        persistence=True, name='Anno-OAFI-350', unified_label_distribution=True
    )
    app.run(debug=True)
