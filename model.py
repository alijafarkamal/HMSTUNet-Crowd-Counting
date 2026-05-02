import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MSViTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, scales=(1, 2), dropout=0.1):
        super().__init__()
        self.scales = scales
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 1)
        self.scale_fac = self.head_dim ** -0.5
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in scales])
        self.fusion = nn.Linear(dim * len(scales), dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def _attn(self, x, idx):
        B, N, C = x.shape
        q, k, v = self.qkv[idx](x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4).unbind(0)
        a = (q @ k.transpose(-2,-1)) * self.scale_fac
        a = a.softmax(-1)
        return (a @ v).transpose(1,2).reshape(B, N, C)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = self.norm1(x.flatten(2).transpose(1,2))
        outs = []
        for i, s in enumerate(self.scales):
            if s == 1: tok = flat
            else:
                tok = F.avg_pool2d(x, s, s).flatten(2).transpose(1,2)
                tok = self.norm1(tok)
            a = self._attn(tok, i)
            if s > 1:
                ph, pw = max(H//s, 1), max(W//s, 1)
                a = a.transpose(1,2).reshape(B, C, ph, pw)
                a = F.interpolate(a, (H, W), mode='bilinear', align_corners=False)
                a = a.flatten(2).transpose(1,2)
            outs.append(a)
        fused = self.drop(self.fusion(torch.cat(outs, -1)))
        xr = flat + fused
        xr = xr + self.ffn(self.norm2(xr))
        return xr.transpose(1,2).reshape(B, C, H, W)

class DCAB(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        mid = max(dim // r, 4)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, mid), nn.ReLU(), nn.Linear(mid, dim), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x * self.ca(x).view(B, C, 1, 1)
        avg, mx = x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]
        x = x * self.sa(torch.cat([avg, mx], 1))
        return self.act(self.bn(self.pw(self.dw(x))) + x)

class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))

class HMSTUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.enc = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        edims = [96, 192, 384, 768]
        self.msv  = MSViTBlock(edims[3], num_heads=8, scales=(1,2))
        self.dcab = DCAB(edims[3])
        dd = [256, 128, 64, 32]
        self.l0, self.l1, self.l2, self.l3 = nn.Conv2d(edims[3], dd[0], 1), nn.Conv2d(edims[2], dd[0], 1), nn.Conv2d(edims[1], dd[1], 1), nn.Conv2d(edims[0], dd[2], 1)
        self.d3, self.d2, self.d1 = DecBlock(dd[0], dd[0], dd[1]), DecBlock(dd[1], dd[1], dd[2]), DecBlock(dd[2], dd[2], dd[3])
        self.head = nn.Sequential(nn.Conv2d(dd[3], 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        f0, f1, f2, f3 = self.enc(x)
        f3 = self.dcab(self.msv(f3))
        p3, p2, p1, p0 = self.l0(f3), self.l1(f2), self.l2(f1), self.l3(f0)
        d = self.d1(self.d2(self.d3(p3, p2), p1), p0)
        return self.head(d)
