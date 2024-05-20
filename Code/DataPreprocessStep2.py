import json

def load_json(json_file):
    """Load JSON file and return the data."""
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_matching_attributes(first_json, second_json):
    """Extract matching attributes from the second JSON file based on IDs in the first JSON file."""
    first_data = load_json(first_json)
    pic_path_ids = set(first_data.get('pic_path', {}).keys())
    image_count_ids = set(first_data.get('imageCount', {}).keys())

    assert pic_path_ids == image_count_ids, "ID sets in pic_path and imageCount do not match"

    second_data = load_json(second_json)
    matching_attributes = {key: {id_: value[id_] for id_ in value if id_ in pic_path_ids}
                           for key, value in second_data.items() if key not in {'pic_path', 'imageCount'}}

    matching_attributes['pic_path'] = first_data['pic_path']
    matching_attributes['imageCount'] = first_data['imageCount']

    return matching_attributes

def save_json(data, json_file):
    """Save the given data to a JSON file."""
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    first_json_file = 'samples_step1_dispose.json'
    second_json_file = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'
    output_json_file = 'final_sample.json'

    matching_data = extract_matching_attributes(first_json_file, second_json_file)
    save_json(matching_data, output_json_file)

if __name__ == '__main__':
    main()
