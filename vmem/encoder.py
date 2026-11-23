"""Spiking frame encoder built from the original Mixer-SNN network, unmodified.

The encoder sees frames one time step each, in order. It is trained to recognise
the activity of short windows. Its last stage emits spikes. Their mean over the
spatial grid, a spike rate per channel, is the feature every memory model reads.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from models.mixers_sparse_patchcell_origin import sMLPNet

SIZES = {
    # Same blocks as the paper, fewer and narrower layers for real-time use.
    "lite": dict(dim=48, alpha=3, patch_size=4, depths=[1, 2, 2, 1]),
    # The paper's Mixer-SNN-Small configuration (models/configs.py).
    "small": dict(dim=78, alpha=3, patch_size=4, depths=[2, 8, 14, 2]),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FrameEncoder(nn.Module):
    def __init__(self, size="lite", image_size=64, num_classes=24):
        super().__init__()
        config = SIZES[size]
        self.net = sMLPNet(in_channels=3, num_classes=num_classes, image_size=image_size, **config)
        functional.set_step_mode(self.net, "m")
        self.feature_dim = config["dim"] * 2 ** (len(config["depths"]) - 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 1, 3, 1, 1))

    def prepare(self, frames):
        """uint8 [T, B, H, W, 3] to normalised float [T, B, 3, H, W]."""
        x = frames.permute(0, 1, 4, 2, 3).float() / 255.0
        return (x - self.mean) / self.std

    def spikes(self, x):
        """The original sMLPNet.forward up to its last spiking layer."""
        net = self.net
        shift = 0
        for i in range(len(net.depths)):
            h = net.to_patch_embedding[i](x)
            x = net.to_patch_embedding_lif[i](h)
            for j in range(net.depths[i]):
                token_h = net.token_mix[j + shift](x)
                x = net.token_mix_lif[j + shift](h + token_h)
                x = net.channel_mix_lif[j + shift](h + token_h + net.channel_mix[j + shift](x))
            shift += net.depths[i]
        return x  # [T, B, C, h, w], binary

    def forward(self, x):
        """Class scores for every time step, as in the original network."""
        return self.net.mlp_head(self.net.batch_norm(self.spikes(x)))

    def features(self, x):
        """Spike rate of each last-stage channel, [T, B, feature_dim]."""
        return self.spikes(x).mean(dim=(3, 4))


class SpikingResNetEncoder(nn.Module):
    """Spiking ResNet18 from spikingjelly, started from ImageNet ResNet18 weights.

    Used only by the upgraded model. Each block adds its shortcut before the
    neuron, so every signal between blocks is a binary spike. SEW-ResNet was tried
    first: its `ADD` mode sends spike counts, not spikes, and its binary `IAND`
    mode left the last block almost silent (rate 0.0001), so it barely learned.
    IF neurons: with LIF the ImageNet weights leave the deeper layers silent.
    """

    def __init__(self, image_size=112, num_classes=24, pretrained=True):
        super().__init__()
        from spikingjelly.activation_based import neuron
        from spikingjelly.activation_based.model import spiking_resnet

        self.net = spiking_resnet.spiking_resnet18(pretrained=pretrained, spiking_neuron=neuron.IFNode,
                                                   detach_reset=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        functional.set_step_mode(self.net, "m")
        self.feature_dim = self.net.fc.in_features
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 1, 3, 1, 1))

    prepare = FrameEncoder.prepare

    def features(self, x):
        """Spike rate of each last-block channel, [T, B, 512]."""
        net = self.net
        x = net.maxpool(net.sn1(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return torch.flatten(net.avgpool(x), 2)

    def forward(self, x):
        return self.net.fc(self.features(x))


def build_encoder(size, image_size, pretrained=True):
    if size == "sresnet18":
        return SpikingResNetEncoder(image_size=image_size, pretrained=pretrained)
    return FrameEncoder(size, image_size=image_size)
