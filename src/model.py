"""Classic ("normal") U-Net, trained from scratch.

        input (3,256,256)
   enc1  32 ch  ───────────────────────────────┐ skip        dec1 32 ch  → 1x1 conv → 4 classes
    ↓ pool                                     │                ↑ up
   enc2  64 ch  ─────────────────────────┐ skip│              dec2 64 ch
    ↓ pool                               │     │                ↑ up
   enc3 128 ch  ───────────────────┐ skip  │     │              dec3 128 ch
    ↓ pool                         │       │     │                ↑ up
   enc4 256 ch  ─────────────┐ skip│       │     │              dec4 256 ch
    ↓ pool                   │     │       │     │                ↑ up
            bottleneck 512 ch  ────┴───────┴─────┴────────────────┘

Each block = [3x3 conv → BatchNorm → ReLU] x 2.  Down-sampling = 2x2 max-pool.
Up-sampling = 2x2 transposed conv, then concatenate the skip connection and apply a block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, classes=4, width=32, levels=4, dropout=0.1):
        super().__init__()
        w = [width * 2 ** i for i in range(levels + 1)]  # e.g. 32, 64, 128, 256, 512
        self.encoders = nn.ModuleList()
        c = in_channels
        for i in range(levels):
            self.encoders.append(DoubleConv(c, w[i]))
            c = w[i]
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(w[levels - 1], w[levels])
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(levels)):
            self.ups.append(nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2))
            self.decoders.append(DoubleConv(2 * w[i], w[i]))  # skip (w[i]) + upsampled (w[i])
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(w[0], classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # only matters if H or W is not divisible by 2^levels
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(self.drop(x))


# =====================================================================================
#  Attention U-Net  (Oktay et al., 2018): same as the classic U-Net, but every skip
#  connection passes through an ATTENTION GATE before it is concatenated.
# =====================================================================================
#   skip  (from the encoder: fine detail, but also lots of irrelevant background)
#   gate  (from the decoder, one level deeper: knows the context, "what are we looking for here?")
#         a = sigmoid( conv1x1( ReLU( conv1x1(skip) + conv1x1(gate) ) ) )    -> one map, values 0..1
#         skip_out = skip * a                                                -> background suppressed
#   Note: the gate is computed at the skip's resolution using the up-sampled decoder features
#   (the usual practical implementation; the paper computes it at the coarser resolution).

class AttentionGate(nn.Module):
    def __init__(self, c_skip, c_gate, c_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(c_skip, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wg = nn.Sequential(nn.Conv2d(c_gate, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.store, self.last = False, None  # set store=True to keep the attention map (for plotting)

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.relu(self.wx(skip) + self.wg(gate)))  # (B,1,H,W) in [0,1]
        if self.store:
            self.last = a.detach()
        return skip * a


class AttentionUNet(UNet):
    """Classic U-Net (same widths / depth / loss as `UNet`) + an attention gate on every skip."""

    def __init__(self, in_channels=3, classes=4, width=32, levels=4, dropout=0.1):
        super().__init__(in_channels, classes, width, levels, dropout)
        w = [width * 2 ** i for i in range(levels + 1)]
        self.gates = nn.ModuleList([AttentionGate(w[i], w[i], max(w[i] // 2, 8))
                                    for i in reversed(range(levels))])

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, gate, dec, skip in zip(self.ups, self.gates, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = gate(skip, x)  # <- the only difference from the classic U-Net
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(self.drop(x))


# =====================================================================================
#  U-Net with a ResNet encoder (optionally ImageNet-pretrained)
# =====================================================================================
#  tile (3,256,256)
#   ├─ full-resolution skip: one small conv block on the raw tile        (32,256,256)
#   └─ ResNet encoder:  f2 (64,128,128) → f4 (64,64,64) → f8 (128,32,32) → f16 (256,16,16) → f32 (512,8,8)
#                       (channel counts shown for resnet18/34; resnet50 is wider)
#  decoder: f32 → up + f16 → up + f8 → up + f4 → up + f2 → up + full-res skip → 1x1 conv → classes
#
#  The ResNet stem down-samples 4x straight away, which can blur thin objects at 10 m.
#  The full-resolution skip gives the decoder back the fine detail of the original tile.

ARCHS = {  # name: (torchvision builder, weights enum name, channels at strides 2,4,8,16,32)
    "resnet18": (torchvision.models.resnet18, "ResNet18_Weights", [64, 64, 128, 256, 512]),
    "resnet34": (torchvision.models.resnet34, "ResNet34_Weights", [64, 64, 128, 256, 512]),
    "resnet50": (torchvision.models.resnet50, "ResNet50_Weights", [64, 256, 512, 1024, 2048]),
}


class ResNetEncoder(nn.Module):
    def __init__(self, name="resnet34", pretrained=True):
        super().__init__()
        fn, wname, ch = ARCHS[name]
        weights = getattr(torchvision.models, wname).IMAGENET1K_V1 if pretrained else None
        r = fn(weights=weights)  # downloads the ImageNet weights on first use when pretrained
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.out_channels = ch

    def forward(self, x):
        f2 = self.stem(x)
        f4 = self.l1(self.pool(f2))
        f8 = self.l2(f4)
        f16 = self.l3(f8)
        f32 = self.l4(f16)
        return [f2, f4, f8, f16, f32]


class UpBlock(nn.Module):
    """Up-sample to the skip's size, concatenate the skip, two 3x3 convs."""

    def __init__(self, c_in, c_skip, c_out, attention=False):
        super().__init__()
        self.gate = AttentionGate(c_skip, c_in, max(c_skip // 2, 8)) if attention else None
        self.conv = DoubleConv(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gate is not None:
            skip = self.gate(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True, classes=4, dropout=0.1, attention=False):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained)
        ec = self.encoder.out_channels
        self.full = DoubleConv(3, 32)  # full-resolution skip
        self.up16 = UpBlock(ec[4], ec[3], 256, attention)
        self.up8 = UpBlock(256, ec[2], 128, attention)
        self.up4 = UpBlock(128, ec[1], 64, attention)
        self.up2 = UpBlock(64, ec[0], 32, attention)
        self.up1 = UpBlock(32, 32, 32, attention)
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(32, classes, 1)

    def forward(self, x):
        f2, f4, f8, f16, f32 = self.encoder(x)
        d = self.up16(f32, f16)
        d = self.up8(d, f8)
        d = self.up4(d, f4)
        d = self.up2(d, f2)
        d = self.up1(d, self.full(x))
        return self.head(self.drop(d))


def build_model(model="unet", encoder="resnet34", pretrained=False, width=32, levels=4,
                dropout=0.1, classes=4):
    if model == "unet":
        return UNet(3, classes, width, levels, dropout)
    if model == "attn_unet":
        return AttentionUNet(3, classes, width, levels, dropout)
    if model == "resnet_unet":
        return ResNetUNet(encoder, pretrained, classes, dropout)
    if model == "attn_resnet_unet":
        return ResNetUNet(encoder, pretrained, classes, dropout, attention=True)
    raise ValueError(f"unknown model '{model}'")


def param_groups(model, lr_enc, lr_dec, wd):
    """Pretrained encoder gets a smaller learning rate than the freshly initialised decoder."""
    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": enc, "lr": lr_enc, "weight_decay": wd},
            {"params": dec, "lr": lr_dec, "weight_decay": wd}]
