
import torch
import torch.nn as nn
import torch.nn.functional as F


# 2.Transformer的自注意
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        residual = x
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return self.layer_norm(x + residual)

# 3.Transformer的Mlp层 暂时不要，简单实现
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.layer_norm = nn.LayerNorm(out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return self.layer_norm(x + residual)

class Transformer_block(nn.Module):
    def __init__(self, dim=5, num_heads=5):
        super(Transformer_block, self).__init__()
        
        self.trans1 = nn.Sequential(
            Attention(dim=dim, num_heads=num_heads, qkv_bias=False, attn_drop_ratio=0.,proj_drop_ratio=0.),
            Mlp(in_features=dim, hidden_features=dim*2, drop=0.),
            )

    def forward(self, x):

        x = self.trans1(x) + x
        
        return x


class Transformer_encoder(nn.Module):
    def __init__(self, dim=5, num_heads=5, tran_num=3):
        super(Transformer_encoder, self).__init__()

        self.trans = torch.nn.ModuleList([Transformer_block(dim=dim, num_heads=num_heads) for _ in range(tran_num)])

    def forward(self, x):

        for trans in self.trans:
            x = trans(x) + x

        return x

class self_attention(nn.Module):
    def __init__(self, dim=5, num_heads=5):
        super(self_attention, self).__init__()
        
        self.atten = Attention(dim=dim, num_heads=num_heads, qkv_bias=False, attn_drop_ratio=0.,proj_drop_ratio=0.)

    def forward(self, x):
        x = self.atten(x)

        return x



if __name__=='__main__':
    x = torch.randn(1,62,5)
    net = Transformer_encoder(dim=5)
    y = net(x)
    print(y.shape)