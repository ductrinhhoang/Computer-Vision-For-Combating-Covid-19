from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import random


class InfoManager:
    # property
    track_ids = []
    count = 0
    polygon = []
    info_dict = {}
    speeds = []

    # method
    def __init__(self, polygon):
        self.polygon = Polygon(polygon)
        self.track_ids = []
        self.count = 0
        self.speeds = []
        self.info_dict = {}

    def update(self, centroid, track_id, detect_time=0):
        if track_id not in self.info_dict:
            self.info_dict[track_id] = {
                "speed": 0,
                "is_in_polygon": False,
                "detect_time": detect_time
            }

        cen_x, cen_y = centroid
        cen_point = Point((cen_x, cen_y))
        is_in_polygon = self.polygon.contains(cen_point)

        if is_in_polygon and not self.info_dict[track_id]["is_in_polygon"]:
            self.info_dict[track_id]["speed"] = 60 + 10*random.random()
            self.info_dict[track_id]["is_in_polygon"] = True
            self.count += 1
        else:
            self.info_dict[track_id]["speed"] = 60 + 10*random.random()
        return is_in_polygon

    def clear_old_id(self, min_track_id):
        new_info_dict = {}
        for track_id in self.info_dict:
            if track_id >= min_track_id:
                new_info_dict[track_id] = self.info_dict[track_id]

        self.info_dict = new_info_dict
