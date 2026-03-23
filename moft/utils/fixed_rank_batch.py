import torch
from typing import List, Tuple, Optional
import torch.nn as nn

class FixedRank(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        device: str = "cpu",
        u_param: Optional[torch.nn.Parameter] = None,
        v_param: Optional[torch.nn.Parameter] = None,
        orthogonalization=-1,
    ):
        # The point is stored as UV^\top
        # U size is (out_dim, r), V size is (in_dim, r)
        #print("in_dim:", in_dim)
        #print("out_dim:", out_dim)
        #print("rank:", rank)
        assert in_dim >= rank and out_dim >= rank
        super().__init__()
        self._orthogonalization = orthogonalization
        if not ((u_param is None) or (v_param is None)):
            self.U = u_param
            self.V = v_param

            self.out_dim = u_param.shape[0]
            self.in_dim = v_param.shape[0]
            self.rank = u_param.shape[1]
        else:
            self.U = nn.Parameter(torch.empty(out_dim, rank, device=device))
            self.V = nn.Parameter(torch.empty(in_dim, rank, device=device))

            self.out_dim = out_dim
            self.in_dim = in_dim
            self.rank = rank
            self.reset_parameters()

    def reset_parameters(self):
        self.U.data = torch.linalg.qr(
            torch.randn(self.out_dim, self.rank, device=self.device)
        ).Q
        self.V.data = torch.randn(self.in_dim, self.rank, device=self.device)
        self._orthogonalization = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) @ self.U.T

    @property
    def orthogonalization(self) -> int:
        return self._orthogonalization

    @property
    def device(self) -> torch.device:
        return self.U.device

    def clone_data(self) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        return (self.U.clone(), self.V.clone())

    def clone(self) -> "FixedRank":
        param = self.clone_data()
        return FixedRank(
            self.in_dim,
            self.out_dim,
            self.rank,
            u_param=param[0],
            v_param=param[1],
            orthogonalization=self.orthogonalization,
        )

    def to_dense(self) -> torch.Tensor:
        return self.U @ (self.V.T)

    @torch.no_grad()
    def orthogonalize(self, mu: int):
        mu = mu % 2
        if self._orthogonalization == mu:
            return
        if mu == 1:
            Q, R = torch.linalg.qr(self.U.data)
            self.U.data = Q
            self.V.data = self.V.data @ R.T
        else:
            Q, R = torch.linalg.qr(self.V.data)
            self.V.data = Q
            self.U.data = self.U.data @ R.T
        self._orthogonalization = mu

    def __imul__(self, a: float) -> "FixedRank":
        if not isinstance(a, (int, float, torch.Tensor)):
            raise TypeError(
                f"Multiplication only supported for scalars, got {type(a)}"
            )
        if self.orthogonalization == 0:
            self.U *= a
        else:
            self.V *= a
        return self

    def __mul__(self, a: float) -> "FixedRank":
        fr = self.clone()
        fr *= a
        return fr

    def __rmul__(self, a: float) -> "FixedRank":
        return self.__mul__(a)

    def __truediv__(self, a: float) -> "FixedRank":
        fr = self.clone()
        fr *= 1 / a
        return fr

    def __itruediv__(self, a: float) -> "FixedRank":
        if not isinstance(a, (int, float, torch.Tensor)):
            raise TypeError(
                f"Division only supported for scalars, got {type(a)}"
            )
        self *= 1 / a
        return self

    def __matmul__(self, C: torch.Tensor) -> "FixedRank":
        fr = self.clone()
        fr.V = C.T @ fr.V
        if self.orthogonalization == 0:
            fr._orthogonalization = -1
        return fr

    def __repr__(self) -> str:
        msg = (
            f"FixedRank(in_dim={self.in_dim}, out_dim={self.out_dim},"
            "rank={self.rank}).orthogonalization={self.orthogonalization}"
        )
        return msg

    def norm(self) -> torch.Tensor:
        if self.orthogonalization == 0:
            return torch.linalg.norm(self.U)
        elif self.orthogonalization == 1:
            return torch.linalg.norm(self.V)
        else:
            fr = self.clone()
            fr.orthogonalize(0)
            return fr.norm()

    def transpose(self) -> "FixedRank":
        fr = self.clone()
        fr.U, fr.V = fr.V, fr.U
        fr._orthogonalization = (
            fr._orthogonalization
            if (fr._orthogonalization < 0)
            else (fr._orthogonalization + 1) % 2
        )

        return fr

class FixedRankBatch:
    def __init__(
        self,
        fr_batch: List[Tuple[torch.Tensor, torch.Tensor]],
        conv_coefs: torch.Tensor,
    ):
        """
        :param fr_batch: list of tuples of factors u, v
        :param conv_coefs: mixture coeffs for fr_batch:
            sum_1^n fr[i]conv_coefs[i]
        """
        assert fr_batch
        self.batch_size = len(fr_batch)
        assert (
            self.batch_size == conv_coefs.numel()
        ), f"Sizes should be equal. {self.batch_size}, {conv_coefs.numel()}"
        sum_coefs = torch.sum(conv_coefs)
        assert torch.all(conv_coefs >= 0) and (
            torch.allclose(
                sum_coefs,
                torch.ones_like(sum_coefs),
            )
        ), "conv_coefs should be non-negative and their sum equal to 1."

        self.device = fr_batch[0][0].device
        self.in_dim = fr_batch[0][1].shape[0]  # V.shape[0]
        self.out_dim = fr_batch[0][0].shape[0]  # U.shape[0]
        self.rank = fr_batch[0][0].shape[1]  # V.shape[1]
        self._orthogonalization = [-1] * self.batch_size

        self._U_batch = torch.empty(
            self.batch_size, self.out_dim, self.rank, device=self.device
        )
        self._V_batch = torch.empty(
            self.batch_size, self.in_dim, self.rank, device=self.device
        )

        self._prepare_batch(fr_batch)
        self.apply_conv_weights(conv_coefs)
        self.orthogonalize(-1)

    @property
    def orthogonalization(self) -> Tuple[int, ...]:
        return tuple(self._orthogonalization)

    def _prepare_batch(
        self, fr_batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        for i, (u, v) in enumerate(fr_batch):
            self._U_batch[i, :, :] = u  # (b, out_dim, rank)
            self._V_batch[i, :, :] = v  # (b, in_dim, rank)

    def matmul(self, mat: torch.Tensor, transposed=True) -> torch.Tensor:
        """
        :param mat: 2d tensor (out/in_dim, rank) of mats
        :param transposed: if True computes fr_batch[i].T @ mat else
            fr_batch[i] @ mat
        :return: 3d tensor (b, in/out_dim, rank) of matmul fr_batch[i] and mat
        """
        A, B = self._U_batch, self._V_batch
        if not transposed:
            A, B = B, A
        return torch.matmul(B, torch.matmul(A.transpose(2, 1), mat))

    def orthogonalize(self, mu: int):
        """
        orthogonalizes all fr_batch elemets
        :param mu: non orthogonal core position
        """
        mu = mu % 2
        for i in range(self.batch_size):
            if self.orthogonalization[i] != mu:
                fr = FixedRank(
                    self.in_dim,
                    self.out_dim,
                    self.rank,
                    self.device,
                    u_param=self._U_batch[i, :, :],
                    v_param=self._V_batch[i, :, :],
                )
                fr.orthogonalize(mu)
                self._U_batch[i, :, :] = fr.U
                self._V_batch[i, :, :] = fr.V
                self._orthogonalization[i] = fr.orthogonalization

    def apply_conv_weights(self, conv_coefs: torch.Tensor):
        """
        :param conv_coefs: mixture coeffs for fr_batch:
            sum_1^n fr[i] * conv_coefs[i]
        """
        for i, t in enumerate(conv_coefs):
            orthogonalization = self.orthogonalization[i]
            if orthogonalization == 0:
                self._U_batch[i, :, :] *= t
            else:
                self._V_batch[i, :, :] *= t

    def sum(self) -> torch.Tensor:
        """
        :return: 2d tensor (out_dim, in_dim) of sum U[i] @ V[i].T
        """
        return torch.sum(
            torch.bmm(self._U_batch, self._V_batch.transpose(2, 1)), 0
        )

    def to_dense(self, i: int) -> torch.Tensor:
        """
        :param i: position of mats in batch
        :return: i-th matrix in dense format
        """
        i = i % self.batch_size
        return self._U_batch[i, :, :] @ self._V_batch[i, :, :].T

    @torch.no_grad()
    def riemannian_barycenter_approximation(
        self,
        als_steps=10,
        device="cpu",
    ) -> FixedRank:
        # init in orthogonalization = 1
        U = torch.linalg.qr(
            torch.randn((self.out_dim, self.rank), device=device)
        ).Q
        V = torch.randn((self.in_dim, self.rank), device=device)

        for _ in range(als_steps):
            # V = sum_i t_i X_{(i)}.T U
            V = torch.sum(self.matmul(U, transposed=True), 0)
            V = torch.linalg.qr(V).Q
            # U = sum_i t_i X_{(i)} V
            U = torch.sum(self.matmul(V, transposed=False), 0)
            U, R = torch.linalg.qr(U)
        V = V @ R.T

        return FixedRank(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            rank=self.rank,
            device=device,
            u_param=U,
            v_param=V,
            orthogonalization=1,
        )