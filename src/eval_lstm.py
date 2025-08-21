import torch
import evaluate
from tqdm import tqdm

def calculate_rouge(model, dataloader, vocab, idx2word, device, limit_batches=20):
    model.eval()
    rouge_metric = evaluate.load('rouge')
    
    progress_bar = tqdm(
        enumerate(dataloader), 
        total=min(len(dataloader), limit_batches), 
        desc="Calculating ROUGE"
    )

    for i, (inputs, targets) in progress_bar:
        if i >= limit_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            for j in range(inputs.size(0)):
                true_len = (inputs[j] != vocab.get("<pad>")).sum().item()
                if true_len <= 1: continue

                prompt_len = int(true_len * 0.75)
                if prompt_len == 0: continue
                
                prompt_indices = inputs[j, :prompt_len].tolist()
                
                prompt_text = " ".join([idx2word.get(idx, "") for idx in prompt_indices if idx != vocab.get("<bos>")])
                
                generated_text = model.generate(
                    start_seq=prompt_text.strip(),
                    max_len=true_len - prompt_len + 5,
                    vocab=vocab,
                    idx2word=idx2word,
                    device=device
                )

                reference_indices = targets[j, :true_len-1].tolist()
                reference_text = " ".join([idx2word.get(idx, "") for idx in reference_indices if idx != vocab.get("<bos>")])
                
                clean_generated = generated_text.replace("<bos>", "").replace("<eos>", "").strip()
                clean_reference = reference_text.replace("<bos>", "").replace("<eos>", "").strip()
                
                rouge_metric.add(prediction=clean_generated, reference=clean_reference)

    results = rouge_metric.compute()
    return results