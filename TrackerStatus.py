from utils import get_centroid


class TrackerStatus:
    def __init__(self, tracker_object, bounding_box, target_object_type):
        self.tracker = tracker_object
        self.bbox = bounding_box
        self.centroid = get_centroid(bounding_box)
        self.target = target_object_type
