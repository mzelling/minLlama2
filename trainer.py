import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode

class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.device = "auto"
        C.num_workers = 4
        C.max_iters = None
        C.max_epochs = None
        C.shuffle_data = False
        C.batch_size = 8
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        return C
    
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.optimizer = None
        self.dataloader = dataloader
        self.callbacks = defaultdict(list)
        
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Running on:", self.device)
        
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    
    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
            
        
    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizer(config)
        
        model.train()
        self.iter_num = 0
        self.epoch_num = 0
        self.iter_time = time.time()
        data_iter = iter(self.dataloader)
        
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                self.trigger_callbacks('on_epoch_end')
                self.epoch_num += 1
                if self.epoch_num >= config.max_epochs:
                    break
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
                
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss, _ = model(x, targets=y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
