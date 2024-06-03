
import logging

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from app.processors.frame_processor import FrameProcessor
from app.utils.template_loader import TemplateLoader

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("easyocr").setLevel(logging.ERROR)

class VideoProcessor:
    def __init__(self, input_video_path, output_video_path, model_path, num_frames=None):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.num_frames = num_frames

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

        template_loader = TemplateLoader('app/config/minimap_template.png', 'app/config/gold_table_template.png')
        self.minimap_template, self.gold_template = template_loader.load_templates()

        self.frame_processor = FrameProcessor(
            self.model, self.minimap_template, self.gold_template, 
            np.array([-120, -120]), np.array([14870, 14980]), self.fps
        )

    def process_video(self):
        progress_bar = tqdm(total=self.num_frames, desc="Processing Frames")

        frame_idx = 0
        out = None

        with open("output/output_video/data.json", "w") as json_file:
            json_file.write("[\n")

            while frame_idx < self.num_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break

                minimap = self.frame_processor.process_frame(frame, frame_idx, json_file)
                if minimap is None:
                    break

                if out is None:
                    x_min, y_min, x_max, y_max = self.frame_processor.initial_bbox
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
