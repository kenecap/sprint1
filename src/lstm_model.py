import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

    def generate(self, start_seq, max_len, vocab, idx2word, device):
        self.eval() 
        
        tokens = start_seq.lower().split()
        
        unk_idx = vocab.get("<unk>")
        bos_idx = vocab.get("<bos>")
        eos_idx = vocab.get("<eos>")
        
        indices = [bos_idx] + [vocab.get(token, unk_idx) for token in tokens]
        
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        
        with torch.no_grad():
            for _ in range(max_len):
                output_logits = self.forward(input_tensor)
                
                last_logits = output_logits[0, -1, :]
                predicted_index = torch.argmax(last_logits).item()
                
                if predicted_index == eos_idx:
                    break
                
                indices.append(predicted_index)
                input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

        generated_text = " ".join([idx2word.get(idx, "<unk>") for idx in indices])
        return generated_text