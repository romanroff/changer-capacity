import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_json_path = os.path.join(BASE_DIR, "default", "default.json")

def get_min_capacity():
    df = pd.read_json(default_json_path)
    
    result_df = df.apply(lambda row: {
        'name': row['name'],
        'capacity': min(unit['capacity'] for unit in row['units']) if row['units'] else None
    }, axis=1)

    return pd.DataFrame(result_df.tolist())