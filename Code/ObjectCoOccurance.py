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
