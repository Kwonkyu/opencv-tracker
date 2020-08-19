from multitracker.utils import get_centroid
import math


class TrackerStatus:
    def __init__(self, tracker_object, tracker_name="InstanceOfTrackerClass", tracker_color=(0, 0, 0)):
        self.tracker = tracker_object
        self.instance_name = tracker_name
        self.bounding_box = (-1, -1, -1, -1)
        self.centroid = (-1, -1)
        self.is_tracker_tracking = False
        self.is_tracker_jumping = False
        self.unique_color = tracker_color
        # manually disable tracker.
        self.disabled = False

    def init_tracker(self, image, bbox):
        self.bounding_box = bbox
        self.tracker.init(image, bbox)
        self.centroid = get_centroid(bbox)

    def update_tracker(self, image):
        if self.disabled:
            return

        self.is_tracker_tracking, self.bounding_box = self.tracker.update(image)
        old_centroid = self.centroid
        if self.is_tracker_tracking is False:
            self.centroid = (-1, -1)
            self.bounding_box = (-1, -1, -1, -1)
            return
        else:
            self.centroid = get_centroid(self.bounding_box)
            old_centroid = [i for i in map(lambda _x: int(_x), old_centroid)]
            self.centroid = [i for i in map(lambda _x: int(_x), self.centroid)]
            self.bounding_box = [i for i in map(lambda _x: int(_x), self.bounding_box)]
            # When tracker updates, it returns float coordinates of bounding box.
            path = (old_centroid, self.centroid)

            x, y, w, h = self.bounding_box
            if math.dist(*path) > math.dist((x, y), (x+w, y+h)) * 2:
                self.is_tracker_jumping = True
