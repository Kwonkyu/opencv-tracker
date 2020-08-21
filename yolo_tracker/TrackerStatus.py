from utils import get_centroid
import math


class TrackerStatus:
    category_index = dict()

    def __init__(self, tracker_object):
        self.tracker = tracker_object
        self.bounding_box = (-1, -1, -1, -1)
        self.target_category = None
        self.centroid = (-1, -1)
        self.path = ((-2, -2), (-1, -1))
        self.index = -1
        self.tracking = False
        self.jumping = False
        self.fail_threshold = 30  # if threshold is 0, destroy this tracker.

    def init_tracker(self, image, bbox):
        self.bounding_box = bbox
        self.tracker.init(image, bbox)
        self.centroid = get_centroid(bbox)

    def is_tracker_jumping(self):
        return self.jumping

    def is_tracker_tracking(self):
        return self.tracking

    def is_tracker_threshold_limit(self):
        return self.fail_threshold <= 0

    def update_tracker(self, image):
        self.tracking, self.bounding_box = self.tracker.update(image)
        old_centroid = self.centroid
        if self.tracking is False:
            self.centroid = (-1, -1)
            self.bounding_box = (-1, -1, -1, -1)
            self.fail_threshold = self.fail_threshold - 1
            return
        else:
            self.centroid = get_centroid(self.bounding_box)
            old_centroid = [i for i in map(lambda _x: int(_x), old_centroid)]
            self.centroid = [i for i in map(lambda _x: int(_x), self.centroid)]
            self.bounding_box = [i for i in map(lambda _x: int(_x), self.bounding_box)]
            self.fail_threshold = 30
            # When tracker updates, it returns float coordinates of bounding box.
            self.path = (old_centroid, self.centroid)

        x, y, w, h = self.bounding_box
        if math.dist(*self.path) > math.dist((x, y), (x+w, y+h)) * 2:
            self.jumping = True

    def set_target_category(self, value):
        self.target_category = value
        tracker_index = TrackerStatus.category_index.get(value)
        if tracker_index is None:
            # This tracker is first one which tracks this kind of target.
            self.index = 1
            TrackerStatus.category_index.update({value: 1})
        else:
            # There're already other trackers which track this kind of target.
            tracker_index = tracker_index + 1
            self.index = tracker_index
            TrackerStatus.category_index.update({value: tracker_index})

    def get_bounding_box(self):
        return self.bounding_box

    def get_target_category(self):
        return self.target_category

    def get_centroid(self):
        return self.centroid

    def get_path(self):
        return self.path

    def get_index(self):
        return self.index

    def get_total_status(self):
        output = dict()
        output.update({"tracker": self.tracker})
        output.update({"bounding_box": self.bounding_box})
        output.update({"category": self.target_category})
        output.update({"index": self.index})
        output.update({"centroid": self.centroid})
        output.update({"tracking": self.tracking})
        output.update({"jumping": self.jumping})
        return output
