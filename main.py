import argparse

from app.processors.video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Video Object Detection with YOLO")
    parser.add_argument('--input', type=str, default='app/config/teste2.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output/output_video/result.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, default='app/model/best.pt', help='Path to YOLO model')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames to process (default: entire video)')
    return parser.parse_args()

def main():
    args = parse_args()
    processor = VideoProcessor(args.input, args.output, args.model, args.frames)
    processor.process_video()

if __name__ == "__main__":
    main()
