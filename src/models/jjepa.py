import torch
import torch.nn as nn
import torch.nn.functional as F

import src.util.positional_embedding.create_pos_emb_fn as create_pos_emb_fn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        print("Initializing Attention module")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        print(f"Attention forward pass with input shape: {x.shape}")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        print(f"Attention output shape: {x.shape}")
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        print("Initializing MLP module")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        print(f"MLP forward pass with input shape: {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        print(f"MLP output shape: {x.shape}")
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        print("Initializing Block module")
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        print(f"Block forward pass with input shape: {x.shape}")
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print(f"Block output shape: {x.shape}")
        return x

class JetsTransformer(nn.Module):
    def __init__(self, num_features, embed_dim, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        print("Initializing JetsTransformer module")
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.calc_pos_emb = create_pos_emb_fn(embed_dim)
        
        # Adjust the input dimensions based on the new input shape
        self.patch_embed = nn.Linear(num_features * 30, embed_dim)  # num_features * subjet_length
        
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, subjet_ftrs):
        print(f"JetsTransformer forward pass with input shape: {x.shape}")
        # B: batch size
        # N: Number of subjets
        # C: Number of particle features
        # L: Number of particles
        B, N, C, L = x.shape
        x = x.view(B, N, -1)  # Flatten last two dimensions to [B, N, C*L]
        print(f"Flattened input shape: {x.shape}")

        # emb
        x = self.patch_embed(x)

        # pos emb
        x += self.calc_pos_emb(subjet_ftrs)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        print(f"JetsTransformer output shape: {x.shape}")
        return x.view(B, N, -1)  # Reshape back if necessary

class JetsTransformerPredictor(nn.Module):
    def __init__(self, num_features, embed_dim, predictor_embed_dim, depth, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        print("Initializing JetsTransformerPredictor module")
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.calc_predictor_pos_emb = create_pos_emb_fn(predictor_embed_dim)
        self.predictor_blocks = nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer) for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, 8 * 30, bias=True)  # Match target dimensions
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_x, context_subjet_ftrs, target_subjet_ftrs):
        print(f"JetsTransformerPredictor forward pass with input shape: {x.shape}")
        # calcualte context positional embedding
        x = self.predictor_embed(x)
        x += self.calc_predictor_pos_emb(context_subjet_ftrs)

        # concat conditional jet token to x
        # TODO: add an learnable token
        pred_token = self.calc_predictor_pos_emb(target_subjet_ftrs)

        # TODO: repeat?
        x = torch.cat([x, pred_token], axis=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_proj(x)
        print(f"JetsTransformerPredictor output shape: {x.shape}")
        return x.view(x.size(0), x.size(1), 8, 30)  # Reshape to match target_repr shape

class JJEPA(nn.Module):
    def __init__(self, input_dim, embed_dim, depth, num_heads, mlp_ratio, dropout=0.1, use_predictor=True):
        super(JJEPA, self).__init__()
        print("Initializing JJEPA module")
        self.use_predictor = use_predictor
        self.context_transformer = JetsTransformer(num_features=input_dim, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=dropout)
        if self.use_predictor:
            self.predictor_transformer = JetsTransformerPredictor(num_features=input_dim, embed_dim=embed_dim, predictor_embed_dim=embed_dim//2, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=dropout)


        # Debug Statement - Dimension check
        self.input_check = DimensionCheckLayer("Model Input", 3)
        self.context_check = DimensionCheckLayer("After Context Transformer", 3)
        self.predictor_check = DimensionCheckLayer("After Predictor", 3)

    def forward(self, context, target):
        print(f"JJEPA forward pass with context shape: {context.shape} and target shape: {target.shape}")
        context = context.to(next(self.parameters()).device)
        target = target.to(next(self.parameters()).device)
        
        context_repr = self.context_transformer(context)
        # Debug Statement
        context_repr = self.context_check(context_repr)
        if self.use_predictor:
            pred_repr = self.predictor_transformer(context_repr, None, None)
            pred_repr = self.predictor_check(pred_repr)
            print(f"JJEPA output - pred_repr shape: {pred_repr.shape}, context_repr shape: {context_repr.shape}, target shape: {target.shape}")
            return pred_repr, context_repr, target
        
        print(f"JJEPA output - context_repr shape: {context_repr.shape}, target shape: {target.shape}")
        return context_repr, target

