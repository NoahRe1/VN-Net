import torch
import torch.nn as nn
from .seq2seq import Seq2SeqAttrs

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, conv, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        self.conv = conv

    def forward_once(self, inputs, hidden_state=None):
        """
        inputs: [B, N, C]
        hidden_state: [num_layers, B, N, C]
        """
        if isinstance(inputs, torch.Tensor):
            batch_size = inputs.size(0)
        else:
            batch_size = inputs[0].size(0) # For MTNGCN

        if hidden_state is None:
            hidden_state = torch.zeros((self.layer_num, batch_size, self.node_num, self.rnn_units))
            if isinstance(inputs, torch.Tensor):
                hidden_state = hidden_state.to(inputs.device)
            else:
                hidden_state = hidden_state.to(inputs[0].device) # For MTNGCN
        hidden_states = []
        output = inputs
        others = {}
        for layer_num, layer in enumerate(self.conv):
            next_hidden_state, others = layer(X=output, H=hidden_state[layer_num], **others)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state            

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slows
    
    def forward(self, inputs):
        """
        inputs: [T, B, N, C]
        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.forward_once(inputs[t], encoder_hidden_state)

        return encoder_hidden_state