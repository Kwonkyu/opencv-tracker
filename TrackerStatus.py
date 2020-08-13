from utils import get_centroid
import math


class TrackerStatus:
    category_index = dict()

    def __init__(self, tracker_object):
        self.tracker = tracker_object
        self.bounding_box = (-1, -1, -1, -1)
        self.target_category = None
        self.centroid = (-1, -1)
        self.index = -1
        self.tracking = False
        self.jumping = False

    def init_tracker(self, image, bbox):
        self.bounding_box = bbox
        self.tracker.init(image, bbox)
        self.centroid = get_centroid(bbox)

    def is_tracker_jumping(self):
        return self.jumping

    def is_tracker_tracking(self):
        return self.tracking

    def update_tracker(self, image):
        self.tracking, self.bounding_box = self.tracker.update(image)
        old_centroid = self.centroid
        if self.tracking is False:
            self.centroid = (-1, -1)
            self.bounding_box = (-1, -1, -1, -1)
        else:
            self.centroid = get_centroid(self.bounding_box)
            self.centroid = [i for i in map(lambda x: int(x), self.centroid)]
            self.bounding_box = [i for i in map(lambda x: int(x), self.bounding_box)]
            # When tracker updates, it returns float coordinates of bounding box.

        x, y, w, h = self.bounding_box
        if math.dist(old_centroid, self.centroid) > math.dist((x, y), (x+w, y+h)) * 2:
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

    def get_index(self):
        return self.index

    def get_total_status(self):
        output = dict()
        output.update({"tracker":self.tracker})
        output.update({"bounding_box":self.bounding_box})
        output.update({"category":self.target_category})
        output.update({"index": self.index})
        output.update({"centroid":self.centroid})
        output.update({"tracking":self.tracking})
        output.update({"jumping":self.jumping})
        return output
