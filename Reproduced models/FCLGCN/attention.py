import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: Queries, with dimension [B, L_q, D_q]
        :param k: Keys, with dimension [B, L_k, D_k]
        :param v: Values, with dimension [B, L_v, D_v], generally is k
        :param scale: Scaling factor, a float scalar
        :param attn_mask: Masking, with dimension [B, L_q, L_k]
        :return context: Context
        :return attention: Attention
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(attn_drop)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(proj_drop)
        # layer norm after multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # residual connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, num_heads, -1, dim_per_head).transpose(1, 2) \
            .reshape(batch_size, -1, num_heads * dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)


        return output, attention