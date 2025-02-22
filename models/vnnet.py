import numpy as np
import torch
import torch.nn as nn

from .seq2seq import Seq2SeqAttrs
from .encoder import EncoderModel
from .decoder import DecoderModel
from .image import get_image_encoder
from .graph import get_graph_encoder_decoder
from .fusion import get_fusion_method

from lib.utils_g import Meter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VNNET(nn.Module, Seq2SeqAttrs):
    def __init__(self, sparse_idx, logger=None, region=None, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, sparse_idx, **model_kwargs)

        # graph_encoder
        graph_encoder_type = model_kwargs.get("graph_encoder").get("type")
        conv = []
        for i in range(self.layer_num):
            if i == 0:
                conv.append(
                    get_graph_encoder_decoder(graph_encoder_type)(
                        in_channels=self.input_dim,
                        out_channels=self.rnn_units,
                        first_layer=True,
                        **model_kwargs
                    )
                )
            else:
                conv.append(
                    get_graph_encoder_decoder(graph_encoder_type)(
                        in_channels=self.rnn_units,
                        out_channels=self.rnn_units,
                        K=self.max_view,
                        **model_kwargs
                    )
                )
        self.conv = nn.ModuleList(conv)
        self.graph_encoder = EncoderModel(sparse_idx, self.conv, **model_kwargs)

        # graph_decoder
        graph_decoder_type = model_kwargs.get("graph_decoder").get("type")
        conv2 = []
        for i in range(self.layer_num):
            if i == 0:
                conv2.append(
                    get_graph_encoder_decoder(graph_decoder_type)(
                        in_channels=1,
                        out_channels=self.rnn_units,
                        first_layer=True,
                        **model_kwargs
                    )
                )
            else:
                conv2.append(
                    get_graph_encoder_decoder(graph_decoder_type)(
                        in_channels=self.rnn_units,
                        out_channels=self.rnn_units,
                        K=self.max_view,
                        **model_kwargs
                    )
                )

        self.conv2 = nn.ModuleList(conv2)
        self.graph_decoder = DecoderModel(sparse_idx, self.conv2, **model_kwargs)

        # image_encoder
        image_encoder_kwargs = model_kwargs.get("image_encoder")
        image_encoder_type = image_encoder_kwargs.get("type")
        image_encoder_id = image_encoder_kwargs.get("id")
        self.image_encoder = get_image_encoder(image_encoder_type, image_encoder_id)

        # fusion_method
        fusion_method_type = model_kwargs.get("fusion_method").get("type")
        self.fusion = get_fusion_method(fusion_method_type)(
            rnn_units=self.rnn_units, node_num=self.node_num, region=region
        )

        self._logger = logger
        self.fm = Meter()
        self.bm = Meter()

    def forward(self, graph, image, labels=None, batches_seen=None, **kwargs):
        """
        graph: [B, T, N, C]
        image: [B, T, H, W, C]
        labels: [B, T, N, C]
        """

        graph = graph.permute(1, 0, 2, 3)  # (T, B, N, C)
        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)  # (T, B, N, C)

        # graph_encoder
        encoder_hidden_state = self.graph_encoder(graph)  # (Layer, B, N, C)

        # image_encoder
        image = image.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        image_feature = self.image_encoder(image)  # (B, C, H, W)

        # fusion
        fusion_output = self.fusion(
            encoder_hidden_state, image_feature, inputs=graph
        )  # (Layer, B, N, C)
        dec_input = fusion_output

        # graph_decoder
        outputs = self.graph_decoder(
            dec_input, labels, batches_seen=batches_seen
        )  # (T, B, N, C)

        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        outputs = outputs.permute(1, 0, 2, 3)  # (B, T, N, C)
        return outputs
