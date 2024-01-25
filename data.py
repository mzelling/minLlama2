import json
import torch
from torch.utils.data import Dataset, DataLoader

class JSONLData(Dataset):
    """ Dataset with examples like {"prompt": str, "completion": str}. """

    def __init__(self, jsonl_file, prompt_template="{}"):
        """
        Args:
            jsonl_file (string): Path to the jsonl file
            prompt_template: Prompt template with which
                to format the raw prompt string.
        """
        self.raw_data = []
        with open(jsonl_file, 'r') as file:
            for line in file:
                self.raw_data.append(json.loads(line.rstrip()))
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        raw_sample = self.raw_data[idx]
        sample = {'prompt': self.prompt_template.format(raw_sample['prompt']), 
                  'completion': raw_sample['completion']}
        return sample

class JSONLDataLoader():
    """ Data loader for JSONL dataset. """
    def collate_jsonl(self, batch):
        # tokenize plaintext prompts and completions
        p_toks = [self.tokenizer.encode(sample['prompt'], bos=True) for sample in batch]
        c_toks = [self.tokenizer.encode(sample['completion'], eos=True) for sample in batch]
        
        # Set up input and target tokens
        input_tokens = [ p + c[:-1] for p, c in zip(p_toks, c_toks) ]
        target_tokens = [ [self.tokenizer.ignore_token]*(len(p) - 1) + c for p, c in zip(p_toks, c_toks) ]
        
        max_len = max(len(toks) for toks in input_tokens)
        batch_sz = len(input_tokens)
        tensor_config = {"dtype": torch.long, "device": "cuda"}
        
        # pad shorter token sequences on the right with eos token
        input_tensor = torch.full((batch_sz, max_len), self.tokenizer.eos_token, **tensor_config)
        output_tensor = torch.full((batch_sz, max_len), self.tokenizer.eos_token, **tensor_config)
        
        for batch_idx, (i_toks, t_toks) in enumerate(zip(input_tokens, target_tokens)):
            input_tensor[batch_idx, :len(i_toks)] = torch.tensor(i_toks, **tensor_config)
            output_tensor[batch_idx, :len(t_toks)] = torch.tensor(t_toks, **tensor_config)
            
        return (input_tensor, output_tensor)
    
    def __init__(self, dataset, config, tokenizer):
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(
            dataset, 
            shuffle = config.shuffle_data,
            batch_size = config.batch_size,
            num_workers = config.num_workers,
            collate_fn = lambda batch: self.collate_jsonl(batch)
        )
        
    def __iter__(self):
        return self.dataloader.__iter__()
