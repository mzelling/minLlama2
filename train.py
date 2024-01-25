import argparse
import json
import torch
from trainer import Trainer
from model import LlamaModel
from templates import PromptTemplateCollection
from data import JSONLData, JSONLDataLoader

# CLI parser
parser = argparse.ArgumentParser(description="Train your model.")

parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                    help="model path (either HuggingFace or local file). default: \"meta-llama/Llama-2-7b-hf\"")
parser.add_argument("--prompt_template", type=str, default="summarize",
                    help="Name of prompt template to use. default: \"summarize\"")
parser.add_argument("--jsonl_filename", type=str, help="file path of JSONL training dataset.")
parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
parser.add_argument("--max_epochs", type=int, help="maximum number of epochs")
parser.add_argument("--save_path", type=str, help="where to save checkpoint for the trained model.")
args = parser.parse_args()
print("Preparing training run with:")
for arg in vars(args):
    print(f"  {arg}: {getattr(args, arg)}")

# Instantiate the model
MODEL = LlamaModel.from_pretrained(model_path=args.model_path)
PROMPT_TEMPLATE = PromptTemplateCollection[args.prompt_template]
JSONL_FILENAME = args.jsonl_filename
SAVE_PATH = args.save_path

# Create a training config
train_config = Trainer.get_default_config()
train_config.learning_rate = 1e-5
train_config.max_iters = args.max_iters
train_config.max_epochs = args.max_epochs
train_config.num_workers = 0
train_config.batch_size = 2

# Prepare the data
train_dataset = JSONLData(JSONL_FILENAME, prompt_template=PROMPT_TEMPLATE)
train_dataloader = JSONLDataLoader(train_dataset, train_config, MODEL.tokenizer)

# Create a Trainer object
trainer = Trainer(train_config, MODEL, train_dataloader)

# Set callbacks
def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(
            f"epoch {trainer.epoch_num}; iter_dt {trainer.iter_dt * 1000:.2f}ms; " 
            + f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )

trainer.set_callback('on_batch_end', batch_end_callback)
trainer.run()

torch.save(MODEL.state_dict(), SAVE_PATH)