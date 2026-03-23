import numpy as np

import torch
from torch.nn import functional as F

import torch.nn as nn


class BlockdiagButterflyMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Implementation was taken from https://github.com/HazyResearch/fly/blob/master/src/models/layers/blockdiag_butterfly_multiply.py
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        dx, dw1_bfly, dw2_bfly = None, None, None
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly


class MonarchOrthogonal(nn.Module):
    def __init__(self, n: int, nblocks: int, orthogonal=True, method="cayley", device='cuda'):
        """
        Monarch matrix layer

        Args:
            n: input and output number of features
            nblocks: number of blocks on diagonal in factors
            orthogonal: flag of whether to use orthogonal monarch matrix of any orthogonal matrix
            method: method of obtaining orthogonal matrix. If "cayley" we use self.cayley_batch,
                otherwise self.exp_full. Useful only in case orthogonal=True
        """
        assert n % nblocks == 0

        super().__init__()
        self.device = device
        self.L = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks, device=self.device))
        self.R = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks, device=self.device))

        self.orthogonal = orthogonal
        self.n = n
        self.nblocks = nblocks
        self.block_size = n // nblocks
        self.method = method

        self.blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

        self.reset_parameters()

    def reset_parameters(self):
        # initialize whole layer as identity matrix in both cases

        if self.orthogonal:
            torch.nn.init.zeros_(self.L)
            torch.nn.init.zeros_(self.R)
        else:
            block_size = self.n // self.nblocks
            self.L.data = torch.eye(block_size).unsqueeze(0).expand(self.nblocks, block_size, block_size)
            self.R.data = torch.eye(block_size).unsqueeze(0).expand(self.nblocks, block_size, block_size)

    def cayley_batch(self, data):
        # Cayley transform of skew-symmetric matrix
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(I - skew, I + skew, left=False)
        return Q

    def exp_full(self, data):
        # matrix exponent of skew-symmetric matrix
        skew = 0.5 * (data - data.transpose(1, 2))
        return torch.matrix_exp(skew)
    
    def get_next_index(self, save_dir):
        import re
        import os
        # Get the list of files in the directory
        existing_files = os.listdir(save_dir)
        
        # Filter for files with the expected pattern and extract indices
        indices = []
        for filename in existing_files:
            if filename.startswith('L_matrix_') and filename.endswith('.pt'):
                try:
                    index = int(filename.split('_')[2].split('.')[0])  # Extract the number from the filename
                    indices.append(index)
                except (ValueError, IndexError):
                    continue  # Ignore files that don't match the pattern

        # Return the next index as one greater than the max found, or start with 1
        return max(indices, default=0) + 1

    def forward(self, x):
        if self.orthogonal:
            # different methods for obtaining orthogonal matrix
            if self.method == "cayley":
                L = self.cayley_batch(self.L)
                R = self.cayley_batch(self.R)
            elif self.method == "exp":
                L = self.exp_full(self.L)
                R = self.exp_full(self.R)
            elif self.method == "already_orthogonal":
                L = self.L
                R = self.R
            else:
                raise NotImplementedError("Method is not supported. Use 'cayley' or 'exp'.")
        else:
            # non-orthogonal case
            L = self.L
            R = self.R
        
        return self.blockdiag_butterfly_multiply(x, R, L)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.L.data = self.L.data.to(*args, **kwargs)
        self.R.data = self.R.data.to(*args, **kwargs)
        return self
