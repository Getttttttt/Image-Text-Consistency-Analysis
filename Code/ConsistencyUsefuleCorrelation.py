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
