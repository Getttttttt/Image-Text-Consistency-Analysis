import json
from random import sample
from PIL import Image

def load_json(json_file):
    """Load JSON file and return the data."""
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def try_load_all_images(image_base_path, image_count):
    """Attempt to load all images for a given ID to ensure paths are valid."""
    all_image_paths = [f"{image_base_path}\\{i}.jpg" for i in range(image_count)]
    for path in all_image_paths:
        new_path = path.replace('I:\\', 'F:\\20201016ImgData\\20201016ImgData\\')
        try:
            Image.open(new_path)
        except (IOError, OSError):
            return False
    return True

def filter_valid_images(data, max_samples=5000):
    """Extract entries with valid images and return a dictionary of valid entries."""
    valid_entries = {'pic_path': {}, 'imageCount': {}}
    pic_paths = data.get('pic_path', {})
    image_counts = data.get('imageCount', {})

    for image_id, path in pic_paths.items():
        image_count = image_counts.get(image_id, 1)
        base_path = path.rsplit('\\', 1)[0]
        if try_load_all_images(base_path, image_count):
            valid_entries['pic_path'][image_id] = path.replace('I:\\', 'F:\\20201016ImgData\\20201016ImgData\\')
            valid_entries['imageCount'][image_id] = image_count

    if len(valid_entries['pic_path']) > max_samples:
        sampled_ids = sample(list(valid_entries['pic_path'].keys()), max_samples)
        valid_entries['pic_path'] = {id_: valid_entries['pic_path'][id_] for id_ in sampled_ids}
        valid_entries['imageCount'] = {id_: valid_entries['imageCount'][id_] for id_ in sampled_ids}

    return valid_entries

def save_json(data, json_file):
    """Save the given data to a JSON file."""
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    input_json_file = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'
    output_json_file = 'samples_step1_dispose.json'
    
    data = load_json(input_json_file)
    valid_data = filter_valid_images(data)
    save_json(valid_data, output_json_file)

if __name__ == '__main__':
    main()
