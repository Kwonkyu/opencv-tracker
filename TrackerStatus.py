from utils import get_centroid


class TrackerStatus:
    def __init__(self, tracker_object):
        self.tracker = tracker_object
        self.target_bounding_box = (-1, -1, -1, -1)
        self.target_category = None
        self.centroid = (-1, -1)
        self.is_tracking = False
        self.failure_threshold = 30  # wait tracker to recover from failure at 30 frames.

    def init_tracker(self, image, bbox):
        self.target_bounding_box = bbox
        self.tracker.init(image, bbox)
        self.centroid = get_centroid(bbox)

    def is_tracker_tracking(self):
        return self.is_tracking

    def update_tracker(self, image):
        self.is_tracking, self.target_bounding_box = self.tracker.update(image)
        # When tracker updates, it returns float coordinates of bounding box.

    def get_bounding_box(self):
        self.target_bounding_box = [i for i in map(lambda x: int(x), self.target_bounding_box)]
        return self.target_bounding_box

    def get_target_category(self):
        return self.target_category

    def set_target_category(self, value):
        self.target_category = value

    def get_total_status(self):
        output = dict()
        output.update({"tracker":self.tracker})
        output.update({"bounding_box":self.target_bounding_box})
        output.update({"category":self.target_category})
        output.update({"centroid":self.centroid})
        output.update({"status":self.is_tracking})
        return output
