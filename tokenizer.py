from typing import List
from sentencepiece import SentencePieceProcessor
        
class LlamaTokenizer():
    """ Thin wrapper for SentencePieceProcessor. """
    @staticmethod
    def get_tokenizer_file_from_hf():
        from transformers import LlamaTokenizer as HFLlamaTokenizer
        hf_tokenizer = HFLlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        return hf_tokenizer.vocab_file

    def __init__(self, model_file: str = None, ignore_token=-100):
        if model_file is None:
            model_file = LlamaTokenizer.get_tokenizer_file_from_hf()
        self.sp_model = SentencePieceProcessor(model_file=model_file)
        self.bos_token = self.sp_model.bos_id() # signify the beginning of model generation
        self.eos_token = self.sp_model.eos_id() # signify the end of model generation
        self.pad_token = self.sp_model.pad_id()
        self.ignore_token = ignore_token # ignore this token when computing cross-entropy loss
                                               
    def encode(self, plaintext: str, bos: bool = False, eos: bool = False) -> List[int]:
        """ Encode plaintext into a list of tokens. """
        tokens = self.sp_model.encode(plaintext)
        if bos:
            tokens = [self.sp_model.bos_id()] + tokens
        if eos:
            tokens = tokens + [self.sp_model.eos_id()]
        return tokens
            
    def decode(self, tokens: List[int]) -> str:
        """ Decode a list of tokens into plaintext. """
        return self.sp_model.decode(tokens)
    
    
### EXAMPLE

if __name__ == "__main__":
    tok = LlamaTokenizer()

    from transformers import LlamaTokenizer as HFLlamaTokenizer
    tok_hf = HFLlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    for in_str in ["Hi, I'm Mike.", "Welcome to the Llama OS!"]:
        print("INPUT:", '"{}"'.format(in_str))
        print("-"*80)
        print('\tLlamaTokenizer   ->', tok.encode(in_str, bos=True))
        print('\tHFLlamaTokenizer ->', tok_hf(in_str)['input_ids'], '\n')