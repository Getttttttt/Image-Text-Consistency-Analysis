import json
import os
from mmdet.apis import DetInferencer

class Config:
    def __init__(self):
        self.model = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        self.weights = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.output_directory = './Output'
        self.device = 'cpu'
        self.prediction_score_threshold = 0.3
        self.batch_size = 1
        self.show_results = False
        self.save_visualization = True
        self.save_predictions = True
        self.print_results = False
        self.palette = 'none'
        self.custom_entities = False
        self.chunked_size = -1
        self.tokens_positive = None
        self.texts = None

def load_image_paths(json_file_path):
    """Load image paths from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        image_paths_dict = {key: paths for item in data['pic_path'] for key, paths in item.items()}
    return image_paths_dict

def process_images(config, image_paths_dict):
    """Process images using the configuration and image paths."""
    image_index = 1
    for key, paths in image_paths_dict.items():
        if image_index < 4068:
            image_index += 1
            continue

        output_key_dir = os.path.join(config.output_directory, key)
        os.makedirs(output_key_dir, exist_ok=True)

        for image_path in paths:
            inferencer = DetInferencer(
                model=config.model,
                weights=config.weights,
                device=config.device,
                palette=config.palette
            )
            results = inferencer(
                inputs=image_path,
                out_dir=output_key_dir,
                show=config.show_results,
                no_save_vis=not config.save_visualization,
                no_save_pred=not config.save_predictions,
                print_result=config.print_results,
                batch_size=config.batch_size,
                pred_score_thr=config.prediction_score_threshold
            )
        image_index += 1

def main():
    config = Config()
    image_paths_dict = load_image_paths("./Data/final_sample.json")
    process_images(config, image_paths_dict)

if __name__ == '__main__':
    main()
