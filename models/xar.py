from functools import partial



import numpy as np
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn as nn
from timm.layers import SwiGLU
from timm.models.vision_transformer import LayerScale, DropPath
from typing import Optional
import torch.nn.functional as F
from .rope import *
from .flow import SILoss
import models.sampler as sampler
import torch.nn as nn
import math
import torch.utils.checkpoint
from flash_attn import flash_attn_func
from scipy.stats import norm

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight=None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        else:
            return output * self.weight




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            scale=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )
        self.resolusion = scale
        self.k=None
        self.v=None

    def forward(self, x: torch.Tensor, mask, update_cache=True,scale_index=None) -> torch.Tensor:
        B, N, C = x.shape
        if self.training:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = self.rope(q)
            k = self.rope(k)
            x = F.scaled_dot_product_attention(
                    q, k, v,attn_mask=mask,
                    dropout_p=self.attn_drop if self.training else 0.,
                )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = self.rope(q, scale_index)
            k = self.rope(k, scale_index)
            if self.k is not None:
                k,v = torch.cat([self.k[:k.shape[0]],k], dim=2), torch.cat([self.v[:k.shape[0]], v], dim=2)
            if update_cache:
                self.k,self.v=k[:, :, :-64], v[:, :, :-64]
            if update_cache:
                x = F.scaled_dot_product_attention(q,k,v,attn_mask=mask).permute(0,2,1,3).reshape(B, N, C)
            else:
                x = F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
            scale=None
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            scale=scale
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 6*dim))
        self.dim=dim


    def forward(self, x: torch.Tensor, condition, mask, update_cache=True,scale_index=None) -> torch.Tensor:
        
        num_scales = condition.shape[0]//x.shape[0]
        condition = self.ada_lin(condition).view(-1, 1, 6, self.dim).chunk(num_scales, dim=0)
        condition=torch.cat([condition[i].repeat(1, 64,1,1) for i in range(num_scales)], dim=1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = condition.unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), mask,update_cache=update_cache,scale_index=scale_index).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x


class xAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 diffusion_batch_mul=4, 
                 clusters=4,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h*self.seq_w
        self.cluster_h, self.cluster_w = int(np.sqrt(clusters)), int(np.sqrt(clusters))
        self.clusters = clusters
        self.token_embed_dim = vae_embed_dim * patch_size**2
        

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000+1, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.time_embed = TimestepEmbedder(encoder_embed_dim)
        # Fake class embedding for CFG's unconditional generation
        #self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))


        # --------------------------------------------------------------------------
        # encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = RMSNorm(encoder_embed_dim, weight=True)#nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.mask_ratio_generator = stats.truncnorm((0.7 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm =  RMSNorm(encoder_embed_dim, weight=True)

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm =  RMSNorm(decoder_embed_dim, weight=True) 
        self.pred = nn.Linear(decoder_embed_dim, 16)
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        
        self.flow_loss_fn = SILoss()
        self.diffusion_batch_mul = diffusion_batch_mul

        attention_mask = []
        start=0
        total_length = self.seq_len
        for pz in range(clusters):
            start+=self.seq_len//clusters
            attention_mask.append(torch.cat([torch.ones((self.seq_len//clusters, start)),
                                             torch.zeros((self.seq_len//clusters, total_length - start))], dim=-1))
        # self.variable('constant', 'attention_mask', lambda :jnp.concatenate(attention_mask, axis=0))
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = torch.where(attention_mask == 0, -torch.inf, attention_mask)
        attention_mask = torch.where(attention_mask == 1, 0, attention_mask)
        # self.attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', attention_mask.contiguous())


    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def clusterify(self, x):
        bsz, c, h, w = x.shape
        p = h//self.cluster_h
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(bsz, h_ * w_, p ** 2, c)
        return [i.squeeze() for i in x.chunk(self.clusters, dim=1)]

    def unclusterify(self, x):
        b, n, c=x.shape
        x = x.reshape(b, 2,2, 8,8, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(b, c, 16, 16)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    def forward_mae_encoder(self, x, condition, mask, update_cache=True,scale_index=0):
        if self.training:
            encoder_pos_embed_learned=self.encoder_pos_embed_learned
        else:
            encoder_pos_embed_learned =self.encoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters]
        if (not self.training) and (mask is not None):
            mask = mask[:,:,(scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters, :]
        x = x + encoder_pos_embed_learned
        x = self.z_proj_ln(x)
        for blk in self.encoder_blocks:
            x = blk(x, condition, mask, update_cache=update_cache, scale_index=scale_index)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, condition, mask, update_cache=True, scale_index=0):
        x = self.decoder_embed(x)
        # decoder position embedding
        if self.training:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned
        else:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned[:, (scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters]
        if (not self.training) and (mask is not None):
            mask = mask[:,:,(scale_index+1)*self.seq_len//self.clusters-x.shape[1]:(scale_index+1)*self.seq_len//self.clusters, :(scale_index+1)*self.seq_len//self.clusters]
        x = x + decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, condition, mask,update_cache=update_cache,scale_index=scale_index)
        x = self.decoder_norm(x)
        x = self.pred(x)
        return x

    def mean_flat(self, x):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.mean(x, dim=list(range(1, len(x.size()))))
    
    def sample_logit_normal(self, mu=0, sigma=1, size=1):
        # Generate samples from the normal distribution
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        
        # Transform samples to be in the range (0, 1) using the logistic function
        samples = 1 / (1 + np.exp(-samples))

        # Numpy to Tensor
        samples = torch.tensor(samples, dtype=torch.float32)

        return samples
        
    def forward(self, imgs, labels):
        B,C,_,_=imgs.shape
        label_drop = torch.rand(imgs.shape[0],).cuda()<self.label_drop_prob
        fake_label = torch.ones(imgs.shape[0],).cuda()*1000
        labels = torch.where(label_drop, fake_label, labels)

        patches = self.clusterify(imgs) #B N C
        time_input = self.sample_logit_normal(size=patches[0].shape[0]*self.clusters)
        time_input = time_input.unsqueeze(-1).unsqueeze(-1)
        time_input = time_input.to(device=patches[0].device, dtype=patches[0].dtype)
        
        noises = [torch.randn_like(patches[0]) for i in range(self.clusters)]
        alpha_t = 1 - time_input
        sigma_t = time_input
        d_alpha_t = -1
        d_sigma_t =  1

        alpha_t = alpha_t.chunk(self.clusters, dim=0)
        sigma_t = sigma_t.chunk(self.clusters, dim=0)

        model_input = [alpha_t[i] * patches[i] + sigma_t[i] * noises[i] for i in range(len(patches))]
        model_target = [d_alpha_t * patches[i] + d_sigma_t * noises[i] for i in range(len(patches))]
        

        time_input = (time_input*1000).long() 
        time_input = self.time_embed(time_input).squeeze()
        class_embedding = self.class_emb(labels.long()).squeeze()
        model_input= torch.stack(model_input, dim=1).reshape(B,-1, C)
        model_input = self.z_proj(model_input)
        x = self.forward_mae_encoder(model_input, class_embedding.repeat(self.clusters,1)+time_input, self.mask)
        model_output = self.forward_mae_decoder(x, class_embedding.repeat(self.clusters,1)+time_input, self.mask)

        model_target = torch.stack(model_target, dim=1).reshape(B,-1, C)
        denoising_loss = self.mean_flat((model_output - model_target) ** 2).mean()
        return denoising_loss



    def sample_inference(self, x, label, time_input, update_cache, scale_index, mask=None):
        class_embedding = self.class_emb(label.long()).squeeze()
        time_input = (time_input*1000).long() 
        time_input = self.time_embed(time_input).squeeze()
        step =time_input.shape[0]//x.shape[0]
        condition = class_embedding.repeat(step,1)+time_input
        model_input = self.z_proj(x)
        x = self.forward_mae_encoder(model_input, condition, mask,update_cache=update_cache, scale_index=scale_index)
        model_output = self.forward_mae_decoder(x, condition, mask,update_cache=update_cache, scale_index=scale_index)
        if update_cache:
            model_output=model_output[:, -self.seq_len//self.clusters:]
        return model_output



    def sample_tokens(self, num_steps, cfg=1.0, label=None):
        
        if label is None:
            label = torch.ones_like(label).cuda()*1000
 
        indices = list(range(self.clusters))
        sequence = [64 for i in range(self.clusters)]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        prev_cond=None
        generated=None
        for blk in self.encoder_blocks:
            blk.attn.k=None
            blk.attn.v=None
        for blk in self.decoder_blocks:
            blk.attn.k=None
            blk.attn.v=None
        for step in indices:
            scaled_cfg = (cfg-1)*step/(self.clusters-1.)+1  #(cfg-1)*(step+1)/self.clusters+1 ##
            latents= torch.randn([label.shape[0], self.seq_len//self.clusters,16]).cuda()
            z_sample = sampler.euler_maruyama_sampler(self.sample_inference, latents, label, step,num_steps=num_steps, condition=prev_cond, cfg_scale=scaled_cfg, mask= self.mask[:,:,:sequence[step], :sequence[step]]).float()
            prev_cond = z_sample
            if generated is None:
                generated = z_sample
            else:
                generated = torch.cat([generated, z_sample], dim=1)
        # unpatchify
        tokens = self.unclusterify(generated)
        return tokens


def xar_base(**kwargs):
    model = xAR(
        encoder_embed_dim=768, encoder_depth=8, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xar_large(**kwargs):
    model = xAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def xar_huge(**kwargs):
    model = xAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

