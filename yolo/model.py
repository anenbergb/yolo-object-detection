from typing import List, Dict
import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops import Conv2dNormActivation
from functools import partial
from torchvision.models import get_model
from torchvision.models._utils import IntermediateLayerGetter


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
        """
        names: "0", "1", "2"
        example feature_maps: [1, 256, 52, 76], [1, 512, 26, 38], [1, 1024, 13, 19]
        for input shape (416,608)
        """
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


class YoloHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 3, num_classes: int = 80):
        super().__init__()
        # tx, ty, tw, th, objectness, 80 classes
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        feat_dim = num_anchors * (5 + num_classes)
        self.fc = nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds_BCHW = self.fc(x)
        preds_BHWC = preds_BCHW.permute(0, 2, 3, 1)  # .shape [10,13,19,255]
        shape_BHWAC = [
            *preds_BHWC.shape[:-1],
            self.num_anchors,
            self.num_classes + 5,
        ]  # .shape [10,13,19,3,85]
        preds_BHWAC = preds_BHWC.reshape(shape_BHWAC)
        preds_BflatC = preds_BHWAC.flatten(
            start_dim=1, end_dim=-2
        )  # .shape [10,13,19,3,85] -> [10,13*19*3,85]
        return preds_BflatC


class Yolo(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        num_anchors_per_scale: int = 3,
        backbone_name: str = "resnext50_32x4d",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_scale = num_anchors_per_scale
        backbone = get_model(
            backbone_name, weights="DEFAULT", norm_layer=torch.nn.BatchNorm2d
        )

        # modifed from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py#L118
        returned_layers = [2, 3, 4]  # final 3 layers
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        fpn_channel_list = [256, 512, 1024]
        self.fpn = YoloFPN(in_channels_list, out_channels_list=fpn_channel_list)
        self.heads = nn.ModuleList(
            [
                YoloHead(
                    in_channels=ch,
                    num_anchors=num_anchors_per_scale,
                    num_classes=num_classes,
                )
                for ch in fpn_channel_list
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        preds_per_scale = [head(tensor) for head, tensor in zip(self.heads, x.values())]
        preds = torch.cat(preds_per_scale, dim=1)  # [N,num_spatial_anchors,85]
        tx_ty_tw_th, objectness, class_logits = torch.split(
            preds, [4, 1, self.num_classes], dim=-1
        )
        return {
            "tx_ty_tw_th": tx_ty_tw_th,
            "objectness": objectness,
            "class_logits": class_logits,
        }
