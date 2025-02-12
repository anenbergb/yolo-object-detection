from typing import List, Dict
import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops import Conv2dNormActivation
from functools import partial


class YoloBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        act_layer = partial(nn.LeakyReLU, negative_slope=0.1)

        self.layers = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                out_channels // 2,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
            Conv2dNormActivation(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
            Conv2dNormActivation(
                out_channels,
                out_channels // 2,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
            Conv2dNormActivation(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
            Conv2dNormActivation(
                out_channels,
                out_channels // 2,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
        )
        self.final_layer = Conv2dNormActivation(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            norm_layer=nn.BatchNorm2d,
            activation_layer=act_layer,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        intermediate = self.layers(x)
        return intermediate, self.final_layer(intermediate)


class YoloUpsample(nn.Sequential):
    def __init__(self, in_channels: int):
        act_layer = partial(nn.LeakyReLU, negative_slope=0.1)
        out_channels = in_channels // 2
        layers = [
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=act_layer,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
        ]
        super().__init__(*layers)
        self.out_channels = out_channels


class YoloFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int] = [512, 1024, 2048],
        out_channels_list: List[int] = [256, 512, 1024],
    ):
        super().__init__()
        # FPN must have 3 scales
        assert len(in_channels_list) == len(out_channels_list)
        assert len(out_channels_list) == 3

        lateral_convs = []
        for i, (in_channels, out_channels) in enumerate(
            zip(in_channels_list, out_channels_list)
        ):
            lat_out_channels = (
                out_channels if i == len(in_channels_list) - 1 else out_channels // 2
            )
            lateral_convs.append(
                Conv2dNormActivation(
                    in_channels,
                    lat_out_channels,
                    kernel_size=1,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=None,
                )
            )

        self.lateral_convs = nn.ModuleList(lateral_convs)

        # YoloBlocks will be applied after the concatenated features
        self.stages = nn.ModuleList(
            [
                YoloBlock(out_channels, out_channels)
                for out_channels in out_channels_list
            ]
        )
        self.upsamples = nn.ModuleList(
            [YoloUpsample(out_channels // 2) for out_channels in out_channels_list[1:]]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(x) == len(self.lateral_convs)
        names = list(x.keys())
        feature_maps = list(x.values())

        results = []
        x_up = None
        for idx in range(len(x) - 1, -1, -1):
            x_lateral = self.lateral_convs[idx](feature_maps[idx])
            if idx < len(x) - 1:
                x_lateral = torch.cat([x_up, x_lateral], dim=1)
            x_to_up, x_to_head = self.stages[idx](x_lateral)
            if idx > 0:
                # because there is always 1 fewer upsample
                x_up = self.upsamples[idx - 1](x_to_up)
            results.insert(0, x_to_head)
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out
