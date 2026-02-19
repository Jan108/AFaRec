from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def test_mongo_connection():
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        # Force a connection to check if the server is available
        client.admin.command('ping')
        return
    except ConnectionFailure as e:
        msg = ("MongoDB connection failed, but FiftyOne needs one. Start the service with: \n"
              "sudo systemctl start mongod")
        raise ConnectionError(msg)


def convert_fo_bbox_to_absolute(bbox: tuple[int, int, int, int], img_height: int, img_width: int) -> tuple[int, int, int, int]:
    """
    Converts BBOX from FiftyOne format to xmin, ymin, xmax, ymax with absolut values based on image height and width

    :param bbox: BBox in FiftyOne format xmin, ymin, width, height relative
    :param img_height: Image height
    :param img_width: Image width
    :return: absolute xmin, ymin, xmax, ymax
    """
    def minmax_abs(v: float, a: int) -> int:
        return round(max(min(v, 1), 0) * a)
    xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    xmax = minmax_abs(xmin + w, img_width)
    ymax = minmax_abs(ymin + h, img_height)
    xmin = minmax_abs(xmin, img_width)
    ymin = minmax_abs(ymin, img_height)
    return xmin, ymin, xmax, ymax


def convert_xyxy_to_xyhwn(xyxy: tuple[int, int, int, int], img_height: int,
                          img_width: int) -> tuple[float, float, float, float]:
    """
    Convert bounding box from xyxy format to xywhn format
    :param xyxy: BBox in xyxy format xmin, ymin, xmax, ymax
    :param img_height: Image height
    :param img_width: Image width
    :return: normalized xmin, ymin, width, height
    """
    x_min, y_min, x_max, y_max = xyxy

    width = x_max - x_min
    height = y_max - y_min

    x_min_norm = x_min / img_width
    y_min_norm = y_min / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_min_norm, y_min_norm, width_norm, height_norm