import torch
import torch.nn as nn
import numpy as np

"""Class to create Patches for given image."""
class Patch_Embedding(nn.Module):

    def __init__(self, in_channels: int, patch_size: int, emb_dim: int):
        super(Patch_Embedding, self).__init__()
        """
        Args:
        in_channels -> Image channels.
        patch_size -> Size of the patch in which images have to be divided.
        emb_dim -> Dimension of the patches divided. It is given by square(patch_size) * in_channels [1].

        """
        self.patch_creator = nn.Conv2d(in_channels = in_channels, 
                                       out_channels = emb_dim,
                                       kernel_size = patch_size,
                                       stride = patch_size
                                      )
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.patch_creator(x)
        out = self.flatten(out)
        out = out.transpose(1, 2)
        return out
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create Multi-Layer Perceptron[MLP] Network."""
class MLP(nn.Module):

    def __init__(self, Embedding_dim: int, dropout_rate: float):
        super(MLP, self).__init__()
        """
        This MLP block is similiar to Positionwise FeedForward block of Transformer.

        """
        self.d_model = Embedding_dim
        self.d_ff = self.d_model * 4
        self.linear_1 = nn.Linear(self.d_model, self.d_ff)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           
        out = self.linear_1(x)
        out = self.gelu(out)
        out = self.dropout_1(out)
        out = self.linear_2(out)
        out = self.dropout_2(out)
        return out
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create Multi-Head Self-Attention block."""
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, Embedding_dim: int, attn_drop: float = 0.0):
        super(MultiHeadAttention, self).__init__()

        self.inf = 1e9
        self.dmodel = Embedding_dim
        self.h = num_heads
        self.dk = self.dv = self.dmodel // self.h
        self.Wo = nn.Linear(self.h * self.dv, self.dmodel)
        self.Wq = nn.Linear(self.dmodel, self.h * self.dk)
        self.Wk = nn.Linear(self.dmodel, self.h * self.dk)
        self.Wv = nn.Linear(self.dmodel, self.h * self.dv)
        self.dropout = nn.Dropout(attn_drop)

    
    # Function to perform attention
    def attention(self, Wq: nn.Module, Wk: nn.Module, Wv: nn.Module, x: torch.Tensor, mask = None) -> torch.Tensor:

        """
        An attention function can be described as mapping a query and a set of key-value pairs to an output,
        where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
        of the values, where the weight assigned to each value is computed by a compatibility function of the
        query with the corresponding key.

        Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
        we found it beneficial to linearly project the queries, keys and values h times with different, learned
        linear projections to dk, dk and dv dimensions, respectively.

        Args:
        Wq -> Query Weight Matrix.
        Wk -> Key Weight Matrix.
        Wv -> Value Weight Matrix.
        x -> Input sequence with embeddings.
        mask -> Attention scores to be masked.

        """

        q = Wq(x)
        k = Wk(x)
        v = Wv(x)

        q = q.view(x.size(0), x.size(1), self.h, self.dk).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.h, self.dk).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.h, self.dv).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1) / np.sqrt(self.dk) # Calculation of Attention Scores

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask == 0, -self.inf)
        
        attn_scores = attn_scores.softmax(dim = -1)
        attn_scores = self.dropout(attn_scores)
        attn_values = attn_scores @ v # Calculation of Attention values.
        attn_values = attn_values.transpose(1, 2)
        attn_values = attn_values.flatten(2)

        return attn_values, attn_scores
    
    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
    
        x, mask = x, mask
        # Calculation of attention values for number of heads
        attn_values, attn_scores = self.attention(self.Wq, self.Wk, self.Wv, x, mask)
        multiheadattn_values = self.Wo(attn_values) 
    
        return multiheadattn_values, attn_scores     
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create single Encoder Block."""    
class Transformer_Encoder_Block(nn.Module):

    def __init__(self, emb_dim: int, dropout_rate: float, n_heads: int, attn_dropout_rate: float):
        super(Transformer_Encoder_Block, self).__init__()
        """
        Single Transformer Encoder block comprises of
            1. Layer Normalization
            2. Multi-head Self-attention
            3. Residual Connection
            4. Layer Normalization
            5. MLP
            6. Residual Connection [1].

        """
        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadAttention(num_heads = n_heads, Embedding_dim = emb_dim, attn_drop = attn_dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(Embedding_dim = emb_dim, dropout_rate = dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        residue = x
        out = self.layer_norm_1(x)
        out, attn_scores = self.attention(out)
        residue = out + residue

        out = self.layer_norm_2(residue)
        out = self.mlp(out)
        out = residue + out

        return out, attn_scores
    
"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create Vision Transformer."""    
class ViT(nn.Module):

    def __init__(self, in_channels: int, emb_dim: int, patch_size: int, num_patches: int, dropout_rate: float, attn_dropout_rate: float, 
                 num_encoder: int, n_heads: int, n_class: int):
        super(ViT, self).__init__()
        """
        Vision Transformer comprises of 
            1. Patch Embedding - Create patches of given image.
            2. CLS Token - This learnable embedding gets concate with patches, which holds image representation like BERT's [CLASS] token, which is then 
                           used to classify the image.
            3. Positional Embedding - Learnable embedding which adds position information for the patches.
            4. Transformer Encoder Block - L number of Encoder blocks.
            5. MLP Head - Use CLS token embedding to linearly project to number of output classes.

        """
        self.patch_emb = Patch_Embedding(in_channels, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(size = (1, 1, emb_dim)), requires_grad = True)
        self.pos_emb = nn.Parameter(torch.zeros(size = (1, num_patches + 1, emb_dim)), requires_grad = True)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoders = nn.ModuleList([Transformer_Encoder_Block(emb_dim, dropout_rate, n_heads, attn_dropout_rate) for _ in range(num_encoder)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_dim),
                                      nn.Linear(emb_dim, n_class)
                                     )
        
        #self.apply(self._init_weights)
        
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patches = self.patch_emb(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        patches = torch.cat([cls_token, patches], dim = 1)
        out = self.pos_emb + patches
        out = self.dropout(out)
        
        Attentions = []
        for encoder in self.encoders:
            out, attn_scores = encoder(out)
            Attentions.append(attn_scores)

        out = out[:, 0]
        out = self.mlp_head(out)

        return out, Attentions


if __name__ == '__main__':

    image_size = (176, 176)
    patch_size = 16
    in_channels = 3
    num_patches = (image_size[0] * image_size[1]) // (patch_size ** 2)
    emb_dim = (patch_size ** 2) * in_channels
    num_class = 10
    
    vit = ViT(in_channels = in_channels, emb_dim = emb_dim, patch_size = patch_size, num_patches = num_patches, dropout_rate = 0.001, 
          attn_dropout_rate = 0.0, num_encoder = 4, n_heads = 8, n_class = num_class)
    Num_of_parameters = sum(p.numel() for p in vit.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6)) # Prints Total number of Model Parameters.


    """
    Reference:
    [1] https://arxiv.org/abs/2010.11929

    """