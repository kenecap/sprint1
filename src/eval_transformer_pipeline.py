
import torch
import evaluate
from tqdm import tqdm
import pandas as pd
from transformers import pipeline

def evaluate_transformer(file_path, device, limit_rows=2560):
    
    generator = pipeline(
        "text-generation", 
        model="distilgpt2",
        device=0 if device == "cuda" else -1
    )

    df = pd.read_csv(file_path).head(limit_rows)
    rouge_metric = evaluate.load('rouge')

    progress_bar = tqdm(df['text'], desc="Evaluating Transformer")

    for text in progress_bar:
        tokens = text.split()
        if len(tokens) <= 1:
            continue
        
        prompt_len = int(len(tokens) * 0.75)
        if prompt_len == 0:
            prompt_len = 1
            
        prompt_text = " ".join(tokens[:prompt_len])
        
        try:
            generated_output = generator(
                prompt_text,
                max_new_tokens=len(tokens) - prompt_len + 5,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50
            )
            generated_text = generated_output[0]['generated_text']
        except Exception as e:
            print(f"Error generating for: '{prompt_text}'. Error: {e}")
            continue

        rouge_metric.add(prediction=generated_text, reference=text)

    results = rouge_metric.compute()
    return results