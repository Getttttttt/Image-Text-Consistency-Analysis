import json
import os

def read_json(file_path):
    """
    Read JSON file and return the data.
    
    :param file_path: Path to the JSON file
    :return: Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def generate_report(data):
    """
    Generate a report based on the JSON data.
    
    :param data: Parsed JSON data
    :return: Report string
    """
    report = []
    total_records = len(data)
    report.append(f"Total records: {total_records}")
    
    # Example of generating some statistics
    if total_records > 0:
        sample_record = data[0]
        report.append("\nSample record:")
        for key, value in sample_record.items():
            report.append(f"{key}: {value}")
        
        # Example: Count occurrences of a specific key if exists
        if 'category' in sample_record:
            category_count = {}
            for record in data:
                category = record.get('category', 'Unknown')
                category_count[category] = category_count.get(category, 0) + 1
            report.append("\nCategory counts:")
            for category, count in category_count.items():
                report.append(f"{category}: {count}")
    
    return "\n".join(report)

def main():
    file_path = './Data for Effect of User-Generated Image on Review Helpfulness Perspectives from Object Detection/tabular data/target_comment_seed2021.json'
    if os.path.exists(file_path):
        data = read_json(file_path)
        report = generate_report(data)
        print(report)
    else:
        print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()
