
import re
import pandas as pd
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Простая функция очистки текста.
    """
    text = text.lower()  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def load_and_clean_data(raw_path: Path) -> pd.DataFrame:
    """
    Загружает сырые данные из txt файла, очищает их и возвращает 
    в виде pandas DataFrame.
    """
    print(f"1. Чтение сырого файла: {raw_path}")
    
    
    with open(raw_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    
    df = pd.DataFrame(lines, columns=['text'])
    df.dropna(inplace=True)

    print("2. Очистка текста...")
    df['text'] = df['text'].apply(clean_text)

    # Удаляем пустые строки, которые могли появиться после очистки
    df = df[df['text'].str.len() > 0]
    
    print(f"Готово. Получено {len(df)} чистых строк.")
    
    return df