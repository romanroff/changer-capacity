import pandas as pd

def get_min_capacity():
    # Читаем JSON как DataFrame
    df = pd.read_json('changer_capacity/default/default.json')
    
    # Применяем функцию к каждой строке
    result_df = df.apply(lambda row: {
        'name': row['name'],
        'capacity': min(unit['capacity'] for unit in row['units']) if row['units'] else None
    }, axis=1)

    # Преобразуем в DataFrame
    return pd.DataFrame(result_df.tolist())