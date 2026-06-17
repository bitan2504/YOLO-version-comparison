import csv
import os
from typing import List, Dict, Any

def save_to_csv(data: List[Dict[str, Any]], filepath: str) -> bool:
    """
    Saves a list of dictionaries to a CSV file.
    Creates parent directories automatically if they do not exist.
    """
    if not data:
        print("Warning: No data provided to save.")
        return False
        
    try:
        # Automatically create the folder path if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Extract headers from the keys of the first dictionary
        headers = list(data[0].keys())
        
        # Open file with utf-8 encoding and correct newline handling
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            
        print(f"Success: Data successfully saved to '{filepath}'")
        return True
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False
