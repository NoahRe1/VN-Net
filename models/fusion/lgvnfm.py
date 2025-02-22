import numpy as np
import torch
import torch.nn as nn

class LGVNFM(nn.Module):
    def __init__(self, rnn_units, region, num_heads=4, **kwargs):
        super().__init__()

        if region == 'd60':
            self.region_lat = [38.3, 44.7]
            region_lat_mean = 41.000166666666686
            region_lat_std = 1.273302126580945
            self.region_lon = [117.3, 123.7]
            region_lon_mean = 120.285
            region_lon_std = 1.5803296069702264
        elif region == 'd96':
            self.region_lat = [26.3, 32.7]
            region_lat_mean = 29.570937499999985
            region_lat_std = 1.2827745629794365
            self.region_lon = [100.3, 106.7]
            region_lon_mean = 103.7905208333334
            region_lon_std = 1.2183940675055178
        elif region == 'd139':
            self.region_lat = [26.8, 33.2]
            region_lat_mean = 30.202949640287823
            region_lat_std = 1.3724302098529955
            self.region_lon = [116.8, 123.2]
            region_lon_mean = 119.94877697841724
            region_lon_std = 1.2297589752922053
        else:
            NotImplementedError
        
        self.region_lat = [
            (self.region_lat[0] - region_lat_mean) / region_lat_std,
            (self.region_lat[1] - region_lat_mean) / region_lat_std
        ]
        self.region_lon = [
            (self.region_lon[0] - region_lon_mean) / region_lon_std,
            (self.region_lon[1] - region_lon_mean) / region_lon_std
        ]

        self.hs = None
        self.ws = None

        self.local_proj = nn.Sequential(
            nn.Linear(32 + rnn_units, rnn_units),
            nn.ReLU(),
        )

        self.attn=nn.MultiheadAttention(rnn_units, num_heads, batch_first=True)
    
    def forward(self, graph_feature, image_feature, inputs, **kwargs):
        """
        graph_feature: [Layer, B, N, C]
        image_feature: [B, C, H, W]
        output: [Layer, B, N, C]
        """

        layer_num, bs, n_points, c_station = graph_feature.shape # (Layer, B, N, C)
        bs, c_image, h, w = image_feature.shape

        if self.hs is None or self.ws is None:
            lats = inputs[0, 0, :, 0]
            lons = inputs[0, 0, :, 1]
            hs = h - ((lats - self.region_lat[0]) / (self.region_lat[1] - self.region_lat[0]) * h)
            hs = torch.round(hs).long()
            ws = (lons - self.region_lon[0]) / (self.region_lon[1] - self.region_lon[0]) * w
            ws = torch.round(ws).long()
            self.hs = hs
            self.ws = ws

        local_feats = image_feature[:, :, self.hs, self.ws].permute(0, 2, 1) # (B, N, C)
        local_feats = torch.cat([
            graph_feature,
            local_feats.unsqueeze(0).repeat(graph_feature.size(0), 1, 1, 1)], dim=-1) # (Layer, B, N, 32+C)
        local_feats = self.local_proj(local_feats) + graph_feature # (Layer, B, N, C)
        
        image_feature = image_feature.reshape(bs,c_image,h*w).permute(0,2,1) # (B, H*W, C)
        local_feats_ = local_feats.permute(1,0,2,3).reshape(bs,-1,c_station) # (B, Layer*N, C)

        attn_output, attn = self.attn(query=local_feats_, key=image_feature, value=image_feature) # (B, Layer*N, C)
        attn_output = attn_output.reshape(bs,layer_num,n_points,c_station).permute(1,0,2,3) # (Layer, B, N, C)
        attn_output = attn_output + local_feats # (Layer, B, N, C)

        return attn_output