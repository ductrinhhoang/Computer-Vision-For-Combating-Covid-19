import time


class InOutManager:
    # Property
    '''
    in_out_data is a dictionary of:
        - key: id
        - value: 
            + is_in_polygon
            + count_in_polygon
            + in_time
            + out_time
    '''
    in_out_data = {}
    polygon_count = 0

    # Method
    def update(self, inout_info):
        '''
        Parameter:
            dictionary of:
                - id
                - is_in_polygon
        Ouput:
            update for in out id
        '''
        proc_id = inout_info["id"]
        if proc_id not in self.in_out_data:
            self.in_out_data[proc_id] = {
                "is_in_polygon": inout_info["is_in_polygon"],
                "count_in_polygon": 0,
                "in_time": "",
                "out_time": ""
            }
            if self.in_out_data[proc_id]["is_in_polygon"]:
                # when start, object is in polygon
                self.in_out_data[proc_id]["in_time"] = time.asctime()
                self.polygon_count += 1
        else:
            if not self.in_out_data[proc_id]["is_in_polygon"]:
                if inout_info["is_in_polygon"]:
                    self.in_out_data[proc_id]["is_in_polygon"] = True
                    self.in_out_data[proc_id]["in_time"] = time.asctime()
                    self.polygon_count += 1

            if self.in_out_data[proc_id]["is_in_polygon"]:
                if not inout_info["is_in_polygon"]:
                    self.in_out_data[proc_id]["is_in_polygon"] = False
                    self.in_out_data[proc_id]["out_time"] = time.asctime()
                    self.in_out_data[proc_id]["count_in_polygon"] += 1
                    self.polygon_count -= 1
