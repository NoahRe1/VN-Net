import numpy as np
import torch
import torch.nn as nn
from .seq2seq import Seq2SeqAttrs

class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, conv, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)
        self.conv = conv
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
    
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward_once(self, inputs, hidden_state=None):
        """
        inputs: [B, N, C]
        hidden_state: [num_layers, B, N, C]
        """
        hidden_states = []
        output = inputs
        others = {}
        for layer_num, layer in enumerate(self.conv):
            next_hidden_state, others = layer(X=output, H=hidden_state[layer_num], **others)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output)
        output = projected.reshape(-1, self.node_num, self.output_dim)

        return output, torch.stack(hidden_states)
    

    def forward(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        encoder_hidden_state: [Layer, B, N, C]
        labels: [T, B, N, C]
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.node_num, self.output_dim))
        go_symbol = go_symbol.to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.forward_once(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs