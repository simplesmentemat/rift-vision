import argparse
import json
import logging
import re

import cv2
import easyocr
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


class RiftVision:
    def __init__(self, input_video_path, output_video_path, model_path, num_frames=None):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_path = model_path

        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.basicConfig(level=logging.DEBUG)

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.num_frames = num_frames if num_frames else self.total_frames

        self.minimap_template_path = 'input_video/image3.png'
        self.gold_template_path = 'input_video/image.png'

        self.minimap_template = self.load_template(self.minimap_template_path)
        self.gold_template = self.load_template(self.gold_template_path)

        self.initial_bbox = None
        self.gold_bbox = None
        self.map_bounds_min = np.array([-120, -120])
        self.map_bounds_max = np.array([14870, 14980])

        # Initialize EasyOCR reader with a custom character list
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.reader.lang_char = '0123456789()'

    def load_template(self, template_path):
        template = cv2.imread(template_path)
        if template is None:
            raise IOError(f"Template image not found at {template_path}.")
        return template

    def find_template(self, frame, template):
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = template.shape[:2]
        return top_left[0], top_left[1], top_left[0] + w, top_left[1] + h

    def pixel_to_map_coordinates(self, x, y, width, height, map_bounds_min, map_bounds_max):
        map_width = map_bounds_max[0] - map_bounds_min[0]
        map_height = map_bounds_max[1] - map_bounds_min[1]
        map_x = map_bounds_min[0] + (x / width) * map_width
        map_y = map_bounds_min[1] + (y / height) * map_height
        return map_x, map_y

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(resized, -1, kernel)
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 30, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def extract_gold(self, preprocessed_image):
        result = self.reader.readtext(preprocessed_image)
        text = ' '.join([item[1] for item in result])
        return text

    def parse_gold(self, gold_text):
        cleaned_text = re.sub(r'[^0-9() ]', '', gold_text)
        gold_list = cleaned_text.split()
        gold_data = []
        for gold in gold_list:
            parts = gold.split('(')
            if len(parts) == 2:
                current_gold = parts[0].strip()
                total_gold = parts[1].strip(')')
                gold_data.append({"Gold_inventory": current_gold, "Gold_total": total_gold})
        return gold_data

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
                preprocessed_gold = self.preprocess_image(gold_column)
                gold_text = self.extract_gold(preprocessed_gold)

                gold_data = self.parse_gold(gold_text)
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
                class_name = self.class_names[cls]

                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                map_center_x, map_center_y = self.pixel_to_map_coordinates(center_x, center_y, minimap.shape[1], minimap.shape[0], self.map_bounds_min, self.map_bounds_max)

                label = f"{class_name} ({int(map_center_x)}, {int(map_center_y)})"
                color = (0, 255, 0)
                cv2.rectangle(minimap, (x1, y1), (x2, y2), color, 2)
                cv2.putText(minimap, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return minimap

    def process_video(self):
        progress_bar = tqdm(total=self.num_frames, desc="Processing Frames")

        frame_idx = 0
        out = None

        with open("output_video/detections.json", "w") as json_file:
            json_file.write("[\n")

            while frame_idx < self.num_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break

                minimap = self.process_frame(frame, frame_idx, json_file)
                if minimap is None:
                    break

                if out is None:
                    x_min, y_min, x_max, y_max = self.initial_bbox
                    minimap_width, minimap_height = x_max - x_min, y_max - y_min
                    out = cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (minimap_width, minimap_height))

                out.write(minimap)
                frame_idx += 1
                progress_bar.update(1)

            json_file.write("{}\n]") 

        self.cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        progress_bar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Video Object Detection with YOLOv5")
    parser.add_argument('--input', type=str, default='input_video/teste2.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_video/result.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, default='model/best.pt', help='Path to YOLO model')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames to process (default: entire video)')
    return parser.parse_args()


def main():
    args = parse_args()
    detector = RiftVision(args.input, args.output, args.model, args.frames)
    detector.process_video()


if __name__ == "__main__":
    main()
