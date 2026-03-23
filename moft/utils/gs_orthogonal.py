import math
import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
from typing import List
from ..model.monarch_orthogonal import BlockdiagButterflyMultiply
from .utils import newton_schulz
import time


class GSOrthogonal(nn.Module):
    def __init__(self, n: int, nblocks: int, orthogonal=True, method="cayley", block_size=None):

        if block_size is not None:
            assert n % block_size == 0
            nblocks = n // block_size

        assert n % nblocks == 0

        super().__init__()

        self.R = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))
        self.L = nn.Parameter(torch.empty(nblocks, n // nblocks, n // nblocks))

        self.orthogonal = orthogonal
        self.n = n
        self.nblocks = nblocks
        self.block_size = n // nblocks
        self.method = method

        self.blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply

        self.reset_parameters()

    def reset_parameters(self):
        # initialize whole layer as identity matrix

        if self.orthogonal:
            torch.nn.init.zeros_(self.L)
            torch.nn.init.zeros_(self.R)

        else:
            block_size = self.n // self.nblocks
            self.L.data = (
                torch.eye(block_size)
                .unsqueeze(0)
                .expand(self.nblocks, block_size, block_size)
            )
            self.R.data = (
                torch.eye(block_size)
                .unsqueeze(0)
                .expand(self.nblocks, block_size, block_size)
            )

    def exp_full(self, data):
        skew = 0.5 * (data - data.transpose(1, 2))
        return torch.matrix_exp(skew)

    def cayley_batch(self, data):
        '''
        преобр Кэли переводит любую кососимметрическую матрицу в ортогональную
        '''
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2)) # делаем кососимметрическую по построению
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(I - skew, I + skew, left=False)
        return Q

    def forward(self, x):
        if self.orthogonal:
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
            L = self.L
            R = self.R

        return self.blockdiag_butterfly_multiply(x, R, L)


@torch.no_grad()
def postprocess_blocks(
        gs_orthogonal,
        method,
        parameter
    ):
    """
    Function for modifying eigenvalues of the orthogonal matrix.
    We support several methods to modify eigenvalues, for each we need
    a specific parameter, whose range differs from method to method.
    1. If method == "rotation", we rotate eigenvalues in the opposite direction
    from the point (1, 0) of the 2D plane by "parameter" angle. In this case,
    parameter range is [0, pi].
    2. If method == "eta", we replace the eigenvalue x + iy of an orthogonal matrix with
    (1 + parameter * y) / (1 - parameter * y). There is no specific conditions on parameter
    here.
    3. If method == "exp", we replace the eigenvalue x + iy of an orthogonal matrix with
    exp(2i * y * parameter). There is no specific conditions on parameter here.
    4. If method == "truncation", we truncate eigenvalues with the smallest imaginary part.
    In this case, parameter represents the percentage of the smallest imaginary parts we want to
    truncate. parameter value should be in [0, 1].
    5. If method == "uniform", we set imaginary part of eigenvalues to be uniformly distributed
    across the segment [-parameter, parameter]. In this case, parameter should be in [-1, 1].
    6. If method == "no_modification", we do not perform any eigenvalue modification.
    7. If method == "curve_over_id", we calculate a specific curve going further from Identity matrix.
    In this case, parameter value should be in [0, 1]. parameter=1/2 is equal to setting method="eta" 
    with parameter=1.
    """
    if gs_orthogonal.method != "already_orthogonal": 
        gs_orthogonal = cast_to_already_orthogonal_mode(gs_orthogonal)
    
    if method == "no_modification":
        return gs_orthogonal
    
    if method == "curve_over_id":
        assert ((parameter >= 0) and (parameter <= 1)), "parameter {parameter} for curve_over_id is invalid."
        eta = 0.5 + 2 * parameter * (1 - parameter)
        new_orthogonal_l = gs_orthogonal.cayley_batch(eta * gs_orthogonal.L.data) 
        new_orthogonal_r = gs_orthogonal.cayley_batch(eta * gs_orthogonal.R.data)
        gs_orthogonal.L.data = new_orthogonal_l
        gs_orthogonal.R.data = new_orthogonal_r
        return gs_orthogonal
    
    Kl = (gs_orthogonal.L.data - gs_orthogonal.L.data.transpose(-1, -2)) / 2
    Kr = (gs_orthogonal.R.data - gs_orthogonal.R.data.transpose(-1, -2)) / 2
    
    Dl, Vl = torch.linalg.eigh(-Kl * 1j)
    Dr, Vr = torch.linalg.eigh(-Kr * 1j)

    if method == "rotation": 
        assert ((parameter >= 0) and (parameter <= math.pi)), f"parameter {parameter} for rotation is invalid."
        rotation_matrix_l = torch.ones_like(Dl) * math.cos(parameter) + torch.ones_like(Dl) * torch.sign(Dl) * math.sin(parameter) * 1j
        rotation_matrix_r = torch.ones_like(Dr) * math.cos(parameter) + torch.ones_like(Dr) * torch.sign(Dr) * math.sin(parameter) * 1j
        
        gs_orthogonal.L.data = Vl @ torch.diag_embed((torch.sqrt(1 - torch.square(Dl)) + Dl * 1j) * rotation_matrix_l) @ Vl.mH
        gs_orthogonal.R.data = Vr @ torch.diag_embed((torch.sqrt(1 - torch.square(Dr)) + Dr * 1j) * rotation_matrix_r) @ Vr.mH

    elif method == "eta": 
        new_eigvals_l = (1 + 1j * Dl * parameter) / (1 - 1j * Dl * parameter)
        new_eigvals_r = (1 + 1j * Dr * parameter) / (1 - 1j * Dr * parameter)

        gs_orthogonal.L.data = Vl @ torch.diag_embed(new_eigvals_l) @ Vl.mH
        gs_orthogonal.R.data = Vr @ torch.diag_embed(new_eigvals_r) @ Vr.mH

    elif method == "exp":
        new_eigvals_l = torch.exp(2 * 1j * parameter * Dl)
        new_eigvals_r = torch.exp(2 * 1j * parameter * Dr)
        
        gs_orthogonal.L.data = Vl @ torch.diag_embed(new_eigvals_l) @ Vl.mH
        gs_orthogonal.R.data = Vr @ torch.diag_embed(new_eigvals_r) @ Vr.mH

    elif method == "truncation":
        assert ((parameter >= 0) and (parameter <= 1)), f"parameter {parameter} for truncation is invalid."
        n_truncation = int(parameter * gs_orthogonal.block_size)
        qs_l = torch.quantile(torch.abs(Dl), parameter, dim=1).unsqueeze(1) 
        qs_r = torch.quantile(torch.abs(Dr), parameter, dim=1).unsqueeze(1) 
        # here we make truncation of the smallest n_truncation imaginary parts for each set of eigenvalues
        Dl[torch.abs(Dl) < qs_l] = 0
        Dr[torch.abs(Dr) < qs_r] = 0
        
        gs_orthogonal.L.data = Vl @ torch.diag_embed((torch.sqrt(1 - torch.square(Dl)) + Dl * 1j)) @ Vl.mH
        gs_orthogonal.R.data = Vr @ torch.diag_embed((torch.sqrt(1 - torch.square(Dr)) + Dr * 1j)) @ Vr.mH

    elif method == "uniform":
        l_imaginary_parts = torch.linspace(start=-parameter, end=parameter, steps=Dl.shape[1]).repeat(Dl.shape[0], 1)
        r_imaginary_parts = torch.linspace(start=-parameter, end=parameter, steps=Dr.shape[1]).repeat(Dr.shape[0], 1)

        gs_orthogonal.L.data = Vl @ torch.diag_embed((torch.sqrt(1 - torch.square(l_imaginary_parts)) + l_imaginary_parts * 1j)) @ Vl.mH
        gs_orthogonal.R.data = Vr @ torch.diag_embed((torch.sqrt(1 - torch.square(r_imaginary_parts)) + r_imaginary_parts * 1j)) @ Vr.mH
    
    else:
        raise NotImplementedError(f"Method is not supported {method}")
    return gs_orthogonal


@torch.no_grad()
def cast_to_already_orthogonal_mode(y1: GSOrthogonal):
    if y1.orthogonal:
        if y1.method == "cayley":
            L = y1.cayley_batch(y1.L)
            R = y1.cayley_batch(y1.R)
        elif y1.method == "exp":
            L = y1.exp_full(y1.L)
            R = y1.exp_full(y1.R)
        else:
            L = y1.L
            R = y1.R

        gs_final = GSOrthogonal(y1.n, y1.nblocks, orthogonal=True, method="already_orthogonal")
        gs_final.L = nn.Parameter(L)
        gs_final.R = nn.Parameter(R)
        return gs_final

    else:
        raise NotImplementedError("Impossible to cast non orthogonal matrix to already_orthogonal mode.")


@torch.no_grad()
def merge_inside_cayley_space(
    gs_orthogonals,
    ts,
    ):
    """
    Merge Cayley factors of several GSOrthogonal matrices
    All GSOrthogonal matrices should be in cayley format
    Args:
        gs_orthogonals - list of GSOrthogonal matrices
        ts - list of coefficients (should be positive and sum into 1)
    Returns:
        matrix in GSOrthogonal format with orthogonal blocks
        whose each block is obtained from Cayley transform from
        the weighted sum of Cayley factors.
    """
    assert torch.all(ts >= 0) and (torch.sum(ts) == 1)
    assert len(gs_orthogonals) == ts.numel()
    device = gs_orthogonals[0].L.data.device
    n = gs_orthogonals[0].n
    nblocks = gs_orthogonals[0].nblocks
    for gs_orthogonal in gs_orthogonals:
        assert gs_orthogonal.method == "cayley"
        assert (n == gs_orthogonal.n) and (nblocks == gs_orthogonal.nblocks)

    gs_merge = GSOrthogonal(n=n, nblocks=nblocks, orthogonal=True, method="cayley").to(device)
    for i in range(len(gs_orthogonals)):
        gs_merge.L.data += ts[i] * gs_orthogonals[i].L.data
        gs_merge.R.data += ts[i] * gs_orthogonals[i].R.data

    return cast_to_already_orthogonal_mode(gs_merge)


@torch.no_grad()
def merge_inside_cayley_space_v2(
    gs_orthogonals,
    ts,
    ):
    """
    Merge Cayley factors of several GSOrthogonal matrices
    All GSOrthogonal matrices should be in cayley format
    Args:
        gs_orthogonals - list of GSOrthogonal matrices
        ts - list of coefficients (should be positive and sum into 1)
    Returns:
        matrix in GSOrthogonal format with orthogonal blocks
        whose each block is obtained from Cayley transform from
        the weighted sum of Cayley factors.
    """
    assert torch.all(ts >= 0) and (torch.sum(ts) == 1)
    assert len(gs_orthogonals) == ts.numel()
    device = gs_orthogonals[0].L.data.device
    n = gs_orthogonals[0].n
    nblocks = gs_orthogonals[0].nblocks
    for gs_orthogonal in gs_orthogonals:
        assert gs_orthogonal.method == "cayley"
        assert (n == gs_orthogonal.n) and (nblocks == gs_orthogonal.nblocks)

    gs_merge = GSOrthogonal(n=n, nblocks=nblocks, orthogonal=True, method="cayley").to(device)
    left_norms = torch.zeros(nblocks)[:, None, None].to(device)
    right_norms = torch.zeros(nblocks)[:, None, None].to(device)
    for i in range(len(gs_orthogonals)):
        left_skew = (gs_orthogonals[i].L.data - gs_orthogonals[i].L.data.transpose(-1, -2)) * 0.5
        right_skew = (gs_orthogonals[i].R.data - gs_orthogonals[i].R.data.transpose(-1, -2)) * 0.5

        gs_merge.L.data += ts[i] * left_skew
        gs_merge.R.data += ts[i] * right_skew
        left_norms += ts[i] * torch.norm(left_skew, dim=(-1, -2), keepdim=True)
        right_norms += ts[i] * torch.norm(right_skew, dim=(-1, -2), keepdim=True)

    gs_merge.L.data /= torch.norm(gs_merge.L.data, dim=(-1, -2), keepdim=True)
    gs_merge.R.data /= torch.norm(gs_merge.R.data, dim=(-1, -2), keepdim=True)
    gs_merge.L.data *= left_norms
    gs_merge.R.data *= right_norms

    gs_merge.L.data = torch.tril(gs_merge.L.data) * 2
    gs_merge.R.data = torch.tril(gs_merge.R.data) * 2

    return cast_to_already_orthogonal_mode(gs_merge)


@torch.no_grad()
def blocked_geodesic_combination(y1, y2, t):  
    """
    Minimizing geodesic between two orthogonal matrices y1, y2
    can be given by expression y(t) = y1 * exp(-t * X), t in [0, 1] where X is the smallest
    (in terms of Frobenius norm) skew-symmetric matrix such that exp(X) = y2^T * y1. 
    Args:
        y1, y2 - GSOrthogonal matrices
        t - a point which corresponds to some GSOrthogonal on a path
    Returns:
        matrix in GSOrthogonal format (with the same number of blocks as y1 and y2) where
        between each pair of corresponding blocks we draw a minimizing geodesic.
    """
    assert y1.nblocks == y2.nblocks, "The number of blocks must be equal"
    
    start_time = time.time()
    already_orthogonal_y1 = cast_to_already_orthogonal_mode(y1)
    already_orthogonal_y2 = cast_to_already_orthogonal_mode(y2)
    end_time = time.time()
    time_for_cast_to_already_orthogonal_mode = end_time - start_time
    
    start_time = time.time()
    left_blocks_product = torch.bmm(
        already_orthogonal_y2.L.transpose(-1, -2),
        already_orthogonal_y1.L
    )
    right_blocks_product = torch.bmm(
        already_orthogonal_y2.R.transpose(-1, -2),
        already_orthogonal_y1.R
    )
    end_time = time.time()
    time_for_blocks_product = end_time - start_time

    start_time = time.time()
    left_eigvals, left_eigvecs = torch.linalg.eig(left_blocks_product)
    right_eigvals, right_eigvecs = torch.linalg.eig(right_blocks_product)
    end_time = time.time()
    time_for_eig_decomposition = end_time - start_time
    
    start_time = time.time()
    left_matrix_log = torch.bmm(
        left_eigvecs,
        torch.bmm(
            torch.diag_embed(torch.log(left_eigvals).imag) * 1j,
            left_eigvecs.transpose(-1, -2).conj(),
        ),
    )
    right_matrix_log = torch.bmm(
        right_eigvecs,
        torch.bmm(
            torch.diag_embed(torch.log(right_eigvals).imag) * 1j,
            right_eigvecs.transpose(-1, -2).conj(),
        ),
    )
    end_time = time.time()
    time_for_matrix_log = end_time - start_time

    start_time = time.time()
    combination = GSOrthogonal(
        y1.n, y1.nblocks, orthogonal=True, method="already_orthogonal"
    )
    end_time = time.time()
    time_for_combination_initialization = end_time - start_time
    
    start_time = time.time()
    combination.L.data = torch.bmm(
        already_orthogonal_y1.L, torch.matrix_exp(-t * left_matrix_log).real
    )
    combination.R.data = torch.bmm(
        already_orthogonal_y1.R, torch.matrix_exp(-t * right_matrix_log).real
    )
    end_time = time.time()
    time_for_combination = end_time - start_time
    all_time = time_for_cast_to_already_orthogonal_mode + time_for_blocks_product + time_for_eig_decomposition + time_for_matrix_log + time_for_combination_initialization + time_for_combination
    return combination


@torch.no_grad()
def full_matrix_geodesic_combination(y1, y2, t):
    """
    Minimizing geodesic between two orthogonal matrices y1, y2
    can be given by expression y(t) = y1 * exp(-t * X), t in [0, 1] where X is the smallest
    (in terms of Frobenius norm) skew-symmetric matrix such that exp(X) = y2^T * y1. 
    Args:
        y1, y2 - GSOrthogonal matrices
        t - a point which corresponds to some GSOrthogonal on a path
    Returns:
        matrix in GSOrthogonal format (with one orthogonal block) where the second block is identity
        matrix and the first one is whole orthogonal matrix y(t).
    """
    already_orthogonal_y1 = cast_to_already_orthogonal_mode(y1)
    already_orthogonal_y2 = cast_to_already_orthogonal_mode(y2)
    eye = torch.eye(n=y1.n, device=y1.L.device).unsqueeze(0)
    first_full_matrix = already_orthogonal_y1(eye)
    second_full_matrix = already_orthogonal_y2(eye)

    full_matrix_product = torch.bmm(
        second_full_matrix.transpose(-1, -2),
        first_full_matrix
    )

    full_matrix_eigvals, full_matrix_eigvecs = torch.linalg.eig(
        full_matrix_product
    )

    full_left_matrix = torch.bmm(
        full_matrix_eigvecs,
        torch.matmul(
            torch.diag_embed(torch.log(full_matrix_eigvals).imag) * 1j,
            full_matrix_eigvecs.transpose(-1, -2).conj(),
        )
    )

    combination = GSOrthogonal(
        y1.n, nblocks=1, orthogonal=True, method="already_orthogonal"
    )

    combination.L.data = torch.bmm(
        first_full_matrix, torch.matrix_exp(-t * full_left_matrix).real
    )
    combination.R.data = torch.eye(
        y1.n
    )
    return combination



@torch.no_grad()
def cast_to_already_orthogonal_mode_batch(L: torch.Tensor, R: torch.Tensor):
    """
    Batch version of cast_to_already_orthogonal_mode for Cayley-parameterized GSOrthogonal.
    Args:
        L, R: tensors of shape [B, nblocks, d, d] containing Cayley factors.
    Returns:
        List[GSOrthogonal] with method="already_orthogonal".
    """
    # Применяем батчевый Кэли ко всем матрицам сразу
    L_ortho = cayley_batch_ultra_fast(L)
    R_ortho = cayley_batch_ultra_fast(R)
    return L_ortho, R_ortho


@torch.no_grad()
def merge_inside_cayley_space_batch(
    L1,
    R1,
    L2,
    R2,
    ts,
    ):
    """
    Merge Cayley factors of several GSOrthogonal matrices
    All GSOrthogonal matrices should be in cayley format
    Args:
        gs_orthogonals - list of GSOrthogonal matrices
        ts - list of coefficients (should be positive and sum into 1)
    Returns:
        matrix in GSOrthogonal format with orthogonal blocks
        whose each block is obtained from Cayley transform from
        the weighted sum of Cayley factors.
    """
    assert torch.all(ts >= 0) and (torch.sum(ts) == 1)
    merged_L = ts[0] * L1 + ts[1] * L2
    merged_R = ts[0] * R1 + ts[1] * R2
    return cast_to_already_orthogonal_mode_batch(merged_L, merged_R)


def cayley_batch_ultra_fast(data):
    '''
    преобр Кэли переводит любую кососимметрическую матрицу в ортогональную
    '''
    _, b, r, c = data.shape
    # Ensure the input matrix is skew-symmetric
    skew = 0.5 * (data - data.transpose(2, 3)) # делаем кососимметрическую по построению
    I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

    # Perform the Cayley parametrization
    Q = torch.linalg.solve(I - skew, I + skew, left=False)
    return Q


@torch.no_grad()
def postprocess_blocks_batch(
        L_ortho,
        R_ortho,
        method,
        parameter
    ):

    if method == "curve_over_id":
        assert ((parameter >= 0) and (parameter <= 1)), "parameter {parameter} for curve_over_id is invalid."
        eta = 0.5 + 2 * parameter * (1 - parameter)
        new_orthogonal_l = cayley_batch_ultra_fast(eta * L_ortho) 
        new_orthogonal_r = cayley_batch_ultra_fast(eta * R_ortho)
    else:
        raise NotImplementedError(f"Method is not supported {method}")
    return new_orthogonal_l, new_orthogonal_r