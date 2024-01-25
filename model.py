import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from tokenizer import LlamaTokenizer

class LlamaModel(torch.nn.Module):
    """ PyTorch class for initializing Llama2 with HuggingFace weights. """
    def __init__(self, model):
        super().__init__()
        self.lm_body = model.model
        self.use_cache = model.config.use_cache
        self.lm_head = model.lm_head
        self.cached = None
        self.tokenizer = LlamaTokenizer()
        self._MAX_CONTEXT_WINDOW = 4096
        # make sure CUDA is available
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise Exception("CUDA not available")
        
        
    @staticmethod
    def from_pretrained(model_path, use_cache=False, **kwargs):
        """ Initialize a LlamaModel from pretrained weights (local .pth file or from HuggingFace). """
        if model_path == "meta-llama/Llama-2-7b-hf":
            model = LlamaModel(
                LlamaForCausalLM.from_pretrained(model_path, use_cache=use_cache, **kwargs)
            )
            return model.to(device=model.device, dtype=torch.bfloat16)
        else: 
            config = LlamaConfig(use_cache=use_cache) # have to set is_decoder to TRUE? investigate
            model = LlamaModel(LlamaForCausalLM(config=config)).to(device='cuda', dtype=torch.bfloat16)
            model.load_state_dict(torch.load(model_path, map_location=model.device))
            return model
    

    def forward(self, x, targets=None, position_ids=None, past_key_values=None):
        body_output = self.lm_body.forward(
            **LlamaForCausalLM.prepare_inputs_for_generation(
                self.lm_body,
                input_ids=x, 
                position_ids=position_ids,
                past_key_values=past_key_values
            )
        )
        logits = self.lm_head(body_output['last_hidden_state'])
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                   ignore_index=self.tokenizer.ignore_token)

        return logits, loss, body_output.past_key_values
    
    def configure_optimizer(self, train_config):
        """ Configure AdamW optimizer. """
        no_decay_words = ["embedding", "norm", "bias"]
        no_decay = set()
        decay = set()

        for pn, _ in self.named_parameters():
            if any([(word in pn) for word in no_decay_words]):
                no_decay.add(pn)
            else:
                decay.add(pn)

        # check that union of decay and no_decay parameter groups gives all params
        assert no_decay | decay == {pn for pn, _ in self.named_parameters()}
        assert no_decay & decay == set()

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    @staticmethod
    def pad_sequences(sequences, pad_id):
        """ Helper function to pad shorter token sequences on the right. """
        max_length = max(len(seq) for seq in sequences)
        min_length = min(len(seq) for seq in sequences)
        padded_sequences = [seq + [pad_id] * (max_length - len(seq)) for seq in sequences]
        return min_length, max_length, padded_sequences
    
    @torch.no_grad()
    def generate_from_tokens(self, idx, min_length, max_length, max_new_tokens, 
                             temperature=1.0, do_sample=False, top_k=None):
        """ Generate new tokens starting from the provided token tensor. """
        # activate evaluation mode
        self.eval()
        
        next_pos = min_length
        past_key_values = None
        
        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it
            if idx.size(1) > self._MAX_CONTEXT_WINDOW:
                idx = idx[:, -self._MAX_CONTEXT_WINDOW:]
            
            # forward the model to get the logits for the index in the sequence
            logits, _, past_key_values = self(idx[:, :next_pos],
                                              past_key_values=past_key_values)
            
            # scale last logits by desired temperature
            if (i > 0) and self.use_cache:
                # with caching, iterations after the first only return incremental logits
                assert logits.size(1) == 1
                last_idx = 0
            else:
                last_idx = next_pos-1 if next_pos < max_length else -1
            logits = logits[:, last_idx, :] / temperature
                
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # if some sequences in our batch already have a token at this position, don't overwrite
            if next_pos < max_length:
                idx_next = torch.where(idx[:, [next_pos]] == self.tokenizer.pad_token, idx_next, idx[:, [next_pos]])
                idx[:, [next_pos]] = idx_next
            else:
                idx = torch.cat((idx, idx_next), dim=1)
                # check if all sequences have reached EOS
                if (idx_next == self.tokenizer.eos_token).all().item():
                    break
                
            # increment the position at which to predict tokens
            next_pos += 1

        return idx
    
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, output_only=True, **kwargs):
        """ Continue a list of strings and return the output as longer strings. """
        
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if not isinstance(prompts, list):
            raise ValueError("input should be a list of strings")
        
        min_length, max_length, padded_sequences = self.__class__.pad_sequences(
            list(map(lambda prompt: self.tokenizer.encode(prompt, bos=True), prompts)), 
            self.tokenizer.pad_token
        )
        
        self.eval()
        idx = torch.tensor(padded_sequences).to(device=self.device, dtype=torch.long)
        new_idx_list = self.generate_from_tokens(idx, min_length, max_length, max_new_tokens, **kwargs).tolist()

        if output_only:
            start_idx = [ len(t) for t in padded_sequences ]
        else:
            start_idx = [0] * len(padded_sequences)
    
        # decode tokens and return strings
        return list(map(lambda i: self.tokenizer.decode(new_idx_list[i][start_idx[i]:]),
                        range(len(padded_sequences))))
    
    
### Example

if __name__ == "__main__":
    MODEL_PATHS = []
    for model_path in MODEL_PATHS:
        model = LlamaModel.from_pretrained(model_path, use_cache=True)
        B = 2
        T = 100
        x = torch.randint(low=0, high=32000, size=(B, T)).to('cuda')
        targets = torch.randint(low=0,high=32000,size=(B, T)).to('cuda')

        output, loss = model(x, targets=targets)
        print('Successfully loaded model weights from {} and ran inference.'.format(model_path))