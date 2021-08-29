class ShowIdsManager:
    # Property
    max_id = 0
    show_ids_dict = {}

    # Method
    def __init__(self, predicted_classes=["person"]):
        for pred_class in predicted_classes:
            self.show_ids_dict[pred_class] = {}

    def get_show_id(self, class_key, pred_id):
        if pred_id not in self.show_ids_dict[class_key]:
            self.show_ids_dict[class_key][pred_id] = self.max_id
            self.max_id += 1
        return self.show_ids_dict[class_key][pred_id]
