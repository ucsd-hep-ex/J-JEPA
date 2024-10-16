import torch
import torch.nn as nn
from functools import partial
from .utils import pairwise_lv_fts


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation="gelu"):
        super().__init__()

        self.input_bn = nn.LayerNorm(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            # x = x.permute(2, 0, 1).contiguous()
            x = x.contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(  # noqa: C901
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        mode="sum",
        normalize_input=True,
        activation="gelu",
        eps=1e-8,
        for_onnx=False,
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(
            pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx
        )
        self.out_dim = dims[-1]

        if self.mode == "concat":
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                    ]
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == "sum":
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend(
                        [
                            nn.Conv1d(input_dim, dim, 1),
                            nn.BatchNorm1d(dim),
                            nn.GELU() if activation == "gelu" else nn.ReLU(),
                        ]
                    )
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend(
                        [
                            nn.Conv1d(input_dim, dim, 1),
                            nn.BatchNorm1d(dim),
                            nn.GELU() if activation == "gelu" else nn.ReLU(),
                        ]
                    )
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError("`mode` can only be `sum` or `concat`")

    def forward(self, x, uu=None):  # noqa: C901
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert x is not None or uu is not None
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(
                    seq_len,
                    seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=(x if x is not None else uu).device,
                )
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == "concat":
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == "concat":
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == "sum":
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(
                batch_size,
                self.out_dim,
                seq_len,
                seq_len,
                dtype=elements.dtype,
                device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y
