import torch
import torch.nn as nn

import torch.nn.functional as F

from .monarch_orthogonal import MonarchOrthogonal


class MOFTLayer(nn.Module):
    def __init__(
            self,
            n: int,
            nblocks: int = 16,
            orthogonal: bool = True,
            method: str = 'cayley',
            device: str = 'cuda'    
    ):
        # n - in_features
        super().__init__()

        self.n = n
        self.nblocks = nblocks
        self.device = device
        self.ort_monarch = MonarchOrthogonal(n, nblocks, orthogonal, method, self.device)

    def forward(self, x: torch.Tensor):
        x = self.ort_monarch(x)
        return x


class MOFTCrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, nblocks=16, method='cayley', scale=True, device='cuda'):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_moft = MOFTLayer(hidden_size, nblocks=nblocks, method=method, device=device)
        self.to_k_moft = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method, device=device)
        self.to_v_moft = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method, device=device)
        self.to_out_moft = MOFTLayer(hidden_size, nblocks=nblocks, method=method, device=device)

        if scale:
            self.q_scale = nn.Parameter(torch.ones(hidden_size))
            self.k_scale = nn.Parameter(torch.ones(hidden_size))
            self.v_scale = nn.Parameter(torch.ones(hidden_size))
            self.out_scale = nn.Parameter(torch.ones(hidden_size))

        self.scale = scale

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = self.to_q_moft(hidden_states) # Rx
        query = attn.to_q(query) # WR x 
        if self.scale: # True
            query = self.q_scale * query
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = self.to_k_moft(encoder_hidden_states)
        key = attn.to_k(key)
        if self.scale:
            key = self.k_scale * key
        value = self.to_v_moft(encoder_hidden_states)
        value = attn.to_v(value)
        if self.scale:
            value = self.v_scale * value
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out_moft(hidden_states)
        hidden_states = F.linear(hidden_states, attn.to_out[0].weight)
        if self.scale:
            hidden_states = self.out_scale * hidden_states
        if attn.to_out[0].bias is not None:
            hidden_states = hidden_states + attn.to_out[0].bias
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class DoubleMOFTCrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, nblocks=16, method='cayley', scale=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_moft_in = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_k_moft_in = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_v_moft_in = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_out_moft_in = MOFTLayer(hidden_size, nblocks=nblocks, method=method)

        self.to_q_moft_out = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_k_moft_out = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_v_moft_out = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_out_moft_out = MOFTLayer(hidden_size, nblocks=nblocks, method=method)

        if scale:
            self.q_scale = nn.Parameter(torch.ones(hidden_size))
            self.k_scale = nn.Parameter(torch.ones(hidden_size))
            self.v_scale = nn.Parameter(torch.ones(hidden_size))
            self.out_scale = nn.Parameter(torch.ones(hidden_size))
        
        self.scale = scale

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = self.to_q_moft_in(hidden_states)
        query = attn.to_q(query)
        query = self.to_q_moft_out(query)
        if self.scale:
            query = self.q_scale * query
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = self.to_k_moft_in(encoder_hidden_states)
        key = attn.to_k(key)
        key = self.to_k_moft_out(key)
        if self.scale:
            key = self.k_scale * key

        value = self.to_v_moft_in(encoder_hidden_states)
        value = attn.to_v(value)
        value = self.to_v_moft_out(value)
        if self.scale:
            value = self.v_scale * value

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out_moft_in(hidden_states)
        hidden_states = F.linear(hidden_states, attn.to_out[0].weight)
        hidden_states = self.to_out_moft_out(hidden_states)  # + scale * self.to_out_lora(hidden_states)
        if self.scale:
            hidden_states = self.out_scale * hidden_states
        if attn.to_out[0].bias is not None:
            hidden_states = hidden_states + attn.to_out[0].bias
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class MOFTDoubleCrossAttnProcessor(nn.Module):
    '''
    for direct merge, Vera's code
    '''
    def __init__(self, hidden_size, cross_attention_dim=None, nblocks=16, method='cayley', scale=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_moft_content = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_k_moft_content = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_v_moft_content = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_out_moft_content = MOFTLayer(hidden_size, nblocks=nblocks, method=method)

        if scale:
            self.q_scale_content = nn.Parameter(torch.ones(hidden_size))
            self.k_scale_content = nn.Parameter(torch.ones(hidden_size))
            self.v_scale_content = nn.Parameter(torch.ones(hidden_size))
            self.out_scale_content = nn.Parameter(torch.ones(hidden_size))

        self.scale = scale
        
        self.to_q_moft_style = MOFTLayer(hidden_size, nblocks=nblocks, method=method)
        self.to_k_moft_style = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_v_moft_style = MOFTLayer(cross_attention_dim or hidden_size, nblocks=nblocks, method=method)
        self.to_out_moft_style = MOFTLayer(hidden_size, nblocks=nblocks, method=method)

        if scale:
            self.q_scale_style = nn.Parameter(torch.ones(hidden_size))
            self.k_scale_style = nn.Parameter(torch.ones(hidden_size))
            self.v_scale_style = nn.Parameter(torch.ones(hidden_size))
            self.out_scale_style = nn.Parameter(torch.ones(hidden_size))


    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = self.to_q_moft_content(hidden_states) # Rx
        query = self.to_q_moft_style(query) # Rx
        query = attn.to_q(query) # WR x 
        if self.scale:
            query = self.q_scale_style * self.q_scale_content * query
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = self.to_k_moft_content(encoder_hidden_states)
        key = self.to_k_moft_style(key)
        key = attn.to_k(key)
        if self.scale:
            key = self.k_scale_style * self.k_scale_content * key

        value = self.to_v_moft_content(encoder_hidden_states)
        value = self.to_v_moft_style(value)
        value = attn.to_v(value)
        if self.scale:
            value = self.v_scale_style * self.v_scale_content * value

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out_moft_content(hidden_states)
        hidden_states = self.to_out_moft_style(hidden_states)
        hidden_states = F.linear(hidden_states, attn.to_out[0].weight)
        if self.scale:
            hidden_states = self.out_scale_style * self.out_scale_content * hidden_states
        if attn.to_out[0].bias is not None:
            hidden_states = hidden_states + attn.to_out[0].bias
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states