import json
import logging

import cv2

from app.utils.image_utils import extract_gold, parse_gold, preprocess_image


class FrameProcessor:
    def __init__(self, model, minimap_template, gold_template, map_bounds_min, map_bounds_max, fps):
        self.model = model
        self.minimap_template = minimap_template
        self.gold_template = gold_template
        self.map_bounds_min = map_bounds_min
        self.map_bounds_max = map_bounds_max
        self.fps = fps  
        self.initial_bbox = None
        self.gold_bbox = None

    def find_template(self, frame, template):
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = template.shape[:2]
        return top_left[0], top_left[1], top_left[0] + w, top_left[1] + h

    def pixel_to_map_coordinates(self, x, y, width, height):
        map_width = self.map_bounds_max[0] - self.map_bounds_min[0]
        map_height = self.map_bounds_max[1] - self.map_bounds_min[1]
        map_x = self.map_bounds_min[0] + (x / width) * map_width
        map_y = self.map_bounds_min[1] + (y / height) * map_height
        return map_x, map_y

    def process_frame(self, frame, frame_idx, json_file):
        if self.initial_bbox is None:
            self.initial_bbox = self.find_template(frame, self.minimap_template)
            if self.initial_bbox is None:
                logging.error("Initial minimap detection failed.")
                return None

        if self.gold_bbox is None:
            self.gold_bbox = self.find_template(frame, self.gold_template)
            if self.gold_bbox is None:
                logging.error("Gold table detection failed.")
                return None

        x_min, y_min, x_max, y_max = self.gold_bbox
        gold_table = frame[y_min:y_max, x_min:x_max]

        boxes = [
            (50, 12, 136, 39),  # Blue 1
            (561, 12, 644, 38),  # Red 1
            (50, 55, 136, 83),  # Blue 2
            (561, 55, 644, 83),  # Red 2
            (50, 99, 136, 126),  # Blue 3
            (561, 99, 644, 126),  # Red 3
            (50, 146, 136, 170),  # Blue 4
            (561, 146, 644, 170),  # Red 4
            (50, 186, 136, 215),  # Blue 5
            (561, 186, 644, 215)  # Red 5
        ]
        players_gold_data = {"frame_index": frame_idx, "time": frame_idx / self.fps, "players": []}

        for i in range(5):
            blue_player_data = {"player": f"Blue {i + 1}", "gold": []}
            red_player_data = {"player": f"Red {i + 1}", "gold": []}

            blue_box = boxes[i * 2]
            red_box = boxes[i * 2 + 1]

            for player_data, box in zip([blue_player_data, red_player_data], [blue_box, red_box]):
                gold_column = gold_table[box[1]:box[3], box[0]:box[2]]
                preprocessed_gold = preprocess_image(gold_column)
                gold_text = extract_gold(preprocessed_gold)

                gold_data = parse_gold(gold_text)
                if not gold_data:
                    gold_data = [{"Gold_inventory": None, "Gold_total": None}]
                player_data["gold"].extend(gold_data)
                cv2.rectangle(gold_table, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            players_gold_data["players"].extend([blue_player_data, red_player_data])

        json.dump(players_gold_data, json_file, indent=4)
        json_file.write(",\n")

        minimap = frame[self.initial_bbox[1]:self.initial_bbox[3], self.initial_bbox[0]:self.initial_bbox[2]]
        results = self.model(minimap)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, map(round, box.xyxy.cpu().numpy()[0][:4]))
                _ = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])
                class_name = self.model.names[cls]

                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                map_center_x, map_center_y = self.pixel_to_map_coordinates(center_x, center_y, minimap.shape[1], minimap.shape[0])

                label = f"{class_name} ({int(map_center_x)}, {int(map_center_y)})"
                color = (0, 255, 0)
                cv2.rectangle(minimap, (x1, y1), (x2, y2), color, 2)
                cv2.putText(minimap, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return minimap
