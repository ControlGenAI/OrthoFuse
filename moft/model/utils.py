import torch


# there is no such function in last versions of transformers
def build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


class SaveOutput:
    def __init__(self, register_outputs=True, register_inputs=False):
        self.outputs = {}
        self.inputs = {}
        self.counter = {}
        self.register_outputs = register_outputs
        self.register_inputs = register_inputs
        
    def __call__(self, module, module_in, module_out):
        if not hasattr(module, 'module_name'):
            raise AttributeError('All modules should have name attr')
        if module.module_name not in self.counter:
            self.counter[module.module_name] = 0
        else:
            self.counter[module.module_name] += 1
        if self.register_outputs:
            self.outputs[(module.module_name, self.counter[module.module_name])] = module_out[0].cpu().clone().detach()
        if self.register_inputs:
            self.inputs[(module.module_name, self.counter[module.module_name])] = module_in[0].cpu().clone().detach()

    def clear(self):
        self.outputs = {}
        self.inputs = {}
        self.counter = 0
