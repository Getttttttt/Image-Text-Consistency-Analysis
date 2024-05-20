# Image-Text-Consistency-Analysis
This repository provides tools for downloading, analyzing, and visualizing the co-occurrence of objects in images and text comments, with a focus on measuring and evaluating image-text consistency.

## 题目描述


1. 阅读本周参考文献。

2. 从https://data.mendeley.com/datasets/x54j8vxdmp/2下载相关评论数据，包括图像数据和评论其他属性。

(1) 利用detectron2或mmlab得到图像物体检测结果；

(2) 构建物体共现网络，并将其进行可视化，并观察随着共现阈值增加（小于阈值的边将被删除），该网络结构的变化；

(3) 从物体检测及文本的视角，计算图文一致性，结合表格数据，分析该指标和评论有用性的相关性；

(4) 思考在大模型背景下，有无其他度量图文一致性的方法？


## 实现思路

1. 数据下载和预处理：

对下载的数据进行预处理，确保每个评论对应的图像路径是有效的。将有效的数据保存到新的JSON文件中，供后续步骤使用。同时抽取5k张图片以减少数据处理量。

2. 图像物体检测：

使用mmlab进行图像物体检测，提取图像中包含的物体标签、置信度分数和边界框信息。将检测结果保存，便于后续分析。

3. 物体共现网络构建和可视化：

根据检测到的物体标签，构建物体共现网络。设置不同的共现阈值，生成不同稀疏程度的共现网络。将网络结构保存为.gexf文件，以便于可视化。

4. 图文一致性计算和相关性分析：

从物体检测和文本中提取关键词，计算图文一致性。结合数据，分析图文一致性与评论有用性（UsefulVoteCount）的相关性。通过计算相关系数和P值，评估相关性显著性。

5. 在大模型背景下的图文一致性度量探索：

探讨使用大模型（如CLIP、视觉语言模型）来度量图文一致性的方法。

## 具体实现代码

数据预处理：

```python
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


```

第二步数据预处理：

```python
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
```

图像检测：

```python
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

```

生成物体共现网络

```python
import networkx as nx
import numpy as np
import json
import os

def load_json(json_file_path):
    """Load JSON file and return the data."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_results_and_filter(json_file_path, score_threshold):
    """Load results from JSON file and filter by score threshold."""
    data = load_json(json_file_path)
    filtered_labels = [data['labels'][i] for i, score in enumerate(data['scores']) if score >= score_threshold]
    return filtered_labels

def get_objects_detected(score_threshold):
    """Get detected objects from output directory with a given score threshold."""
    out_dir = './Output'
    objects_detected = {}

    for key in os.listdir(out_dir):
        key_dir = os.path.join(out_dir, key)
        if os.path.isdir(key_dir):
            objects_detected[key] = []
            preds_path = os.path.join(key_dir, 'preds')
            if os.path.exists(preds_path):
                for json_file in os.listdir(preds_path):
                    if json_file.endswith('.json'):
                        json_file_path = os.path.join(preds_path, json_file)
                        results = load_results_and_filter(json_file_path, score_threshold)
                        objects_detected[key].extend(results)

    for key in objects_detected:
        objects_detected[key] = list(set(objects_detected[key]))

    with open('./objects_detected.json', 'w', encoding='utf-8') as json_file:
        json.dump(objects_detected, json_file, ensure_ascii=False, indent=4)

    detected_list = list(objects_detected.values())
    return detected_list

def read_nodes_from_file(file_path):
    """Read nodes from a text file and return as a list."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()

def save_network_to_gexf(G, threshold):
    """Save the network graph to a GEXF file."""
    gexf_path = f'.CoOccurrenceNetwork/Co-Occurrence_network_{threshold}.gexf'
    nx.write_gexf(G, gexf_path)

def build_and_save_network(detected_list, threshold):
    """Build the co-occurrence network and save it as a GEXF file."""
    object_classes = list(set(sum(detected_list, [])))
    co_occurrence_matrix = np.zeros((len(object_classes), len(object_classes)))
    occurrences = {obj: [] for obj in object_classes}

    for img_idx, objects in enumerate(detected_list):
        for obj in objects:
            occurrences[obj].append(img_idx)

    for i, obj_i in enumerate(object_classes):
        for j, obj_j in enumerate(object_classes):
            if i != j:
                intersection = len(set(occurrences[obj_i]) & set(occurrences[obj_j]))
                union = len(set(occurrences[obj_i]) | set(occurrences[obj_j]))
                co_occurrence_matrix[i][j] = intersection / union if union > 0 else 0

    G = nx.Graph()
    for i, obj_i in enumerate(object_classes):
        for j, obj_j in enumerate(object_classes):
            if i < j and co_occurrence_matrix[i][j] > threshold:
                G.add_edge(obj_i, obj_j, weight=co_occurrence_matrix[i][j])

    nodes_file_path = "./nodes.txt"
    nodes_list = read_nodes_from_file(nodes_file_path)
    mapping = {i: nodes_list[i] for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    save_network_to_gexf(G, threshold)

def main(threshold):
    detected_list = get_objects_detected(0.8)
    build_and_save_network(detected_list, threshold)

if __name__ == '__main__':
    main(0.2)
    main(0.5)
    main(0.8)
```

图文一致性计算和相关性计算：

```python
import json
import os
import time
from nltk.corpus import wordnet
from googletrans import Translator
import jieba
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def get_english_synonyms(word, limit=19):
    """Retrieve English synonyms for a given word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
                if len(synonyms) == limit:
                    return synonyms
    return synonyms

def translate_words(words):
    """Translate a list of English words to Chinese using Google Translate."""
    translator = Translator()
    translations = []
    for word in words:
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                translated = translator.translate(word, src='en', dest='zh-cn').text
                translations.append(translated)
                success = True
            except Exception as e:
                retries -= 1
                print(f"Failed to translate {word}: {str(e)}")
                time.sleep(1)
                if retries == 0:
                    translations.append(word)
    return translations

def process_categories(filename):
    """Process categories from a file, get synonyms, and translate them."""
    with open(filename, 'r') as file:
        categories = [line.strip() for line in file.readlines()]

    result = {}
    result_en = {}
    for index, category in enumerate(categories):
        english_synonyms = get_english_synonyms(category)
        english_synonyms.add(category)
        result_en[index] = list(english_synonyms)
        translated_words = translate_words(english_synonyms)
        result[index] = translated_words

    with open('categories_synonyms_en.json', 'w', encoding='utf-8') as f:
        json.dump(result_en, f, ensure_ascii=False, indent=4)

    with open('categories_synonyms.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def load_synonyms(filepath):
    """Load synonyms from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_texts(filepath):
    """Load texts from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)['content']

def text_contains_keywords(text, keywords):
    """Check if a text contains any of the specified keywords."""
    words = set(jieba.cut(text))
    return any(keyword in words for keyword in keywords)

def process_texts(synonyms_filepath, texts_filepath, output_filepath):
    """Process texts and determine their categories based on synonyms."""
    synonyms_data = load_synonyms(synonyms_filepath)
    content_data = load_texts(texts_filepath)

    results = {}
    for item in content_data:
        for key, text in item.items():
            matched_categories = []
            for category_index, synonyms in synonyms_data.items():
                if text_contains_keywords(text, synonyms):
                    matched_categories.append(int(category_index))
            results[key] = matched_categories

    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def get_key(json_file_path):
    """Get keys from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return [key for item in data['pic_path'] for key in item.keys()]

def extract_usefulness_scores(keys, input_filepath, output_filepath):
    """Extract usefulness scores from a JSON file based on given keys."""
    with open(input_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    usefulness_scores = {key: int(data['usefulVoteCount'][key]) for key in keys if key in data['usefulVoteCount']}
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(usefulness_scores, f, ensure_ascii=False, indent=4)

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = set(set1).intersection(set(set2))
    union = set(set1).union(set(set2))
    return len(intersection) / len(union) if union else 0

def compute_consistency_and_correlation():
    """Compute text-image consistency and its correlation with usefulness scores."""
    with open('no_image_data_class.json', 'r') as file:
        text_categories = json.load(file)

    with open('objects_detected.json', 'r') as file:
        image_categories = json.load(file)

    keys = text_categories.keys() & image_categories.keys()
    data = pd.DataFrame({
        'key': list(keys),
        'text_categories': [text_categories[key] for key in keys],
        'image_categories': [image_categories[key] for key in keys]
    })

    data['jaccard_similarity'] = data.apply(
        lambda row: jaccard_similarity(row['text_categories'], row['image_categories']),
        axis=1
    )
    data[['key', 'jaccard_similarity']].to_csv('text_image_consistency.csv', index=False)

    plt.figure(figsize=(8, 5))
    data['jaccard_similarity'].plot(kind='density')
    plt.title('Density Distribution of Text-Image Consistency (Jaccard Similarity)')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('consistency.jpg')

    with open('usefulness_scores.json', 'r') as file:
        usefulness_scores = json.load(file)

    data['usefulness_score'] = data['key'].apply(lambda key: usefulness_scores.get(key, 0))
    data['usefulness_score_standard'] = (data['usefulness_score'] - data['usefulness_score'].min()) / \
                                        (data['usefulness_score'].max() - data['usefulness_score'].min())

    correlation, p_value = pearsonr(data['jaccard_similarity'], data['usefulness_score'])
    correlation2, p_value2 = pearsonr(data['jaccard_similarity'], data['usefulness_score_standard'])

    print(f'Standard: Correlation coefficient: {correlation2}, P-value: {p_value2}')
    print(f'Correlation coefficient: {correlation}, P-value: {p_value}')

def main():
    categories_filename = "./nodes.txt"
    process_categories(categories_filename)

    synonyms_filepath = 'categories_synonyms.json'
    texts_filepath = 'valid_samples_formatted2.json'
    output_filepath = 'no_image_data_class.json'
    process_texts(synonyms_filepath, texts_filepath, output_filepath)

    key_list = get_key('valid_samples_formatted2.json')
    path = 'Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/20201016ImgData/target_comment_seed2021.json'
    extract_usefulness_scores(key_list, path, 'usefulness_scores.json')

    compute_consistency_and_correlation()

if __name__ == '__main__':
    main()

```

## 结果与讨论

### mmlab图像物体检测示例

使用mmlab检测图片示例的demo为：

![image = ](OutputSample\vis\demo.jpg)

生成的对应数据内容为：

```json
{"labels": [2, 2, 2, 13, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 56, 2, 2, 2, 2, 2, 2, 2, 2, 13, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 28, 2, 56, 7, 7, 0, 2, 2, 7, 2, 0, 2, 2, 2, 2], "scores": [0.9887662529945374, 0.98715740442276, 0.9832060933113098, 0.9777801632881165, 0.9713270664215088, 0.9678022265434265, 0.9594148397445679, 0.9592534303665161, 0.956841230392456, 0.951012372970581, 0.9458499550819397, 0.9439922571182251, 0.9331439137458801, 0.8663246035575867, 0.8267839550971985, 0.7778761386871338, 0.7534804344177246, 0.7166000008583069, 0.600783109664917, 0.5919992327690125, 0.5540292263031006, 0.5435200929641724, 0.4767574369907379, 0.46119236946105957, 0.4169999659061432, 0.4009988605976105, 0.3438500463962555, 0.2879628539085388, 0.2612017095088959, 0.2584044337272644, 0.2560497522354126, 0.25596293807029724, 0.2099705934524536, 0.2039610892534256, 0.19742421805858612, 0.1499723196029663, 0.14838436245918274, 0.14762884378433228, 0.14173544943332672, 0.13545489311218262, 0.13270887732505798, 0.12589052319526672, 0.11515292525291443, 0.1107218787074089, 0.10123585164546967, 0.09750967472791672, 0.09688187390565872, 0.08887842297554016, 0.0813654363155365, 0.07645413279533386, 0.0743645653128624, 0.07326328754425049, 0.0720720961689949, 0.06474228948354721, 0.0632171705365181, 0.05161796137690544, 0.05153827369213104], "bboxes": [[609.6500244140625, 113.805908203125, 634.5115966796875, 136.95191955566406], [481.773681640625, 110.48099517822266, 522.4596557617188, 130.40711975097656], [1.0182130336761475, 112.14472198486328, 60.43746566772461, 144.17376708984375], [219.20883178710938, 174.56265258789062, 460.1087951660156, 377.04669189453125], [294.62371826171875, 117.03523254394531, 378.02264404296875, 150.55087280273438], [396.3289489746094, 111.20333099365234, 432.49053955078125, 132.72926330566406], [590.976318359375, 110.80265808105469, 615.40185546875, 126.4935531616211], [267.5820007324219, 105.6860122680664, 328.8187561035156, 128.2265625], [166.8567352294922, 108.00660705566406, 219.10069274902344, 140.19480895996094], [189.76959228515625, 109.80110931396484, 300.3108215332031, 153.78189086914062], [429.8224792480469, 105.65538024902344, 482.74151611328125, 132.37672424316406], [555.0009155273438, 109.78498840332031, 592.7617797851562, 127.8084945678711], [59.679039001464844, 93.18280029296875, 83.45455932617188, 106.24291229248047], [97.84465026855469, 89.654296875, 118.17234802246094, 101.0111083984375], [143.8990020751953, 96.18698120117188, 164.5998077392578, 104.97926330566406], [372.6518249511719, 136.14308166503906, 432.0538024902344, 188.44647216796875], [85.58942413330078, 89.944580078125, 98.89207458496094, 98.5285415649414], [97.82826232910156, 90.74437713623047, 110.29804992675781, 99.7373275756836], [223.57919311523438, 98.51847839355469, 249.84512329101562, 107.50985717773438], [168.92861938476562, 95.94691467285156, 182.84344482421875, 105.69495391845703], [135.02134704589844, 90.87393951416016, 150.60702514648438, 102.79887390136719], [0.0, 111.52195739746094, 14.53266716003418, 125.85028076171875], [553.8966064453125, 116.17054748535156, 562.602294921875, 126.39092254638672], [375.80975341796875, 119.57904815673828, 382.37646484375, 132.11390686035156], [372.06634521484375, 136.31430053710938, 432.19525146484375, 187.17288208007812], [137.9241180419922, 93.79753875732422, 154.49716186523438, 104.65968322753906], [555.009033203125, 110.95270538330078, 574.9256591796875, 126.91203308105469], [554.0431518554688, 100.95907592773438, 561.2978515625, 110.92771911621094], [614.7410278320312, 101.98706817626953, 635.4816284179688, 112.59370422363281], [570.7603149414062, 109.67938232421875, 590.2860717773438, 127.2488784790039], [0.4785451292991638, 111.56817626953125, 22.504091262817383, 142.62355041503906], [375.0931396484375, 111.69644165039062, 420.53692626953125, 133.69107055664062], [262.74725341796875, 107.56563568115234, 326.7658996582031, 143.92529296875], [79.13125610351562, 90.3788833618164, 100.24787902832031, 101.08089447021484], [609.3134765625, 113.30851745605469, 625.9619750976562, 125.3425064086914], [135.30484008789062, 92.37714385986328, 164.08018493652344, 104.99246215820312], [67.35405731201172, 88.50080871582031, 82.9853515625, 97.39421081542969], [540.8524169921875, 113.84894561767578, 561.85546875, 126.19877624511719], [351.7349853515625, 109.43265533447266, 439.3101806640625, 134.81973266601562], [375.3485412597656, 119.17100524902344, 381.95086669921875, 134.4606170654297], [96.31792449951172, 89.87806701660156, 153.28778076171875, 101.7763671875], [45.449501037597656, 117.44498443603516, 61.89558029174805, 144.27505493164062], [91.32599639892578, 107.15576934814453, 106.02936553955078, 119.77730560302734], [606.4075317382812, 112.21598052978516, 618.9356689453125, 124.95723724365234], [218.4040985107422, 175.13783264160156, 462.1076354980469, 365.5412902832031], [188.20008850097656, 109.94707489013672, 300.4713134765625, 156.33583068847656], [427.7491149902344, 105.11559295654297, 483.4554443359375, 132.9432830810547], [532.3619995117188, 109.55472564697266, 540.5265502929688, 125.22264099121094], [102.15262603759766, 93.6143798828125, 141.08184814453125, 101.59896850585938], [398.3648376464844, 112.08146667480469, 409.38983154296875, 132.8977508544922], [294.5048522949219, 117.64228820800781, 378.6327819824219, 150.46356201171875], [539.245849609375, 112.39482879638672, 548.7567138671875, 121.9644546508789], [361.1242980957031, 109.04920959472656, 368.6256103515625, 122.48306274414062], [609.1565551757812, 104.0174560546875, 635.4721069335938, 126.77718353271484], [3.758938789367676, 98.57452392578125, 74.58482360839844, 135.15501403808594], [168.16648864746094, 91.4260482788086, 220.30313110351562, 107.9556884765625], [70.97237396240234, 90.26848602294922, 105.39813232421875, 103.82550811767578]]}
```

mmlab检测图片的示例展示包含多个检测到的物体。生成的数据内容包含每个检测到的物体的标签、置信度分数和边界框坐标。这些数据对于理解图像中包含的物体类型和位置非常有用。例如，在提供的JSON数据中，标签为2的物体出现了多次，且置信度分数较高，这表明这些物体在图像中被高度准确地识别出来。mmlab的检测示例展示了其在复杂图像场景中识别和标注多种物体的能力，为后续的图像分析提供支持。

### 物体共现网络

得到的物体共现网络分别为：

阈值为0.2时

![image = ](CoOccurrenceNetwork\CoOccuranceImage\image0_2.png)

阈值为0.5时

![image = ](CoOccurrenceNetwork\CoOccuranceImage\image0_5.png)

阈值为0.8时

![image = ](CoOccurrenceNetwork\CoOccuranceImage\image0_8.png)

当阈值设置为0.2时，共现网络包含了较多的边，显示出多个物体之间的共现关系。此时网络较为复杂，节点之间的连接密集，物体之间的广泛联系，但是可能会引入较多的噪音。当阈值提高到0.5时，网络中的边数量减少，图像中只保留了较强的共现关系。此时的网络更清晰，强调了那些频繁一起出现的物体对。当阈值进一步提高到0.8时，网络变得更加稀疏，仅保留了最强的共现关系。

### 图文一致性和有用性的相关性分析

绘制的散点图为：

![image = ](Consistency\Output.png)

Correlation coefficient: 0.6460, P-value: 0.00049

绘制的散点图展示了有用性投票数（UsefulVoteCount）与一致性（Consistency）之间的关系。通过对数据进行对数变换，图像显示了不同类别商品在这两个维度上的分布情况。蓝色点代表搜索商品（search_goods），红色点代表体验商品（experience_goods）。从图中可以看到，虽然大部分数据点集中在低有用性投票数和低一致性区域，但在有用性投票数较高时，数据点的一致性也相对较高。计算得到的相关系数（0.6460）和P值（0.00049）表明，一致性和有用性投票数之间存在正相关关系，且相关性具有统计显著性。

### 图文一致性度量的新思考

在大模型（如GPT-4、CLIP等）的背景下，有一些新的方法可以用于度量图文一致性：

 - 基于CLIP的相似度度量：CLIP（Contrastive Language–Image Pretraining）是一种能够将图像和文本映射到同一个向量空间的大模型，可以直接用于计算图文相似度。具体方法是将图像和文本分别输入到CLIP模型中，得到对应的向量表示，然后计算这两个向量之间的余弦相似度，作为图文一致性的度量。

 - 视觉语言模型（VLM）：使用预训练的视觉语言模型（如VL-BERT、ViLBERT）来度量图文一致性。这些模型通过联合训练图像和文本数据，可以捕捉更丰富的跨模态关系。可以将图像和文本输入模型，得到联合表示，然后计算其相似度。

 - 语义相似度计算：使用大语言模型（如GPT-4）来生成图像描述，然后与文本进行语义相似度计算。具体方法是首先使用图像生成描述（Captioning），然后将生成的描述与原文本进行语义相似度计算，例如使用BERT等模型计算两个文本描述的相似度。

 - 多模态对比学习：使用多模态对比学习方法，通过训练图像和文本对的模型，使得相关的图像和文本对在向量空间中更接近，而不相关的对则更远。这种方法可以在训练过程中就考虑图文的一致性，有助于提升度量的准确性。

 - 基于注意力机制的度量：使用注意力机制，将图像和文本中的重要信息进行对齐，然后计算一致性。比如，利用图像中的关键区域和文本中的关键词，计算它们之间的注意力权重，从而得出图文一致性分数。