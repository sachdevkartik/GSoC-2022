import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from e2cnn import gspaces
import e2cnn.nn as nn_eq
from PIL import Image



# file adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cvt.py 
# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        num_classes,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        mlp_last = 192,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        for prefix in ('s1', 's2'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, mlp_last),
            nn.BatchNorm1d(mlp_last),
            nn.ELU(inplace=True),
            nn.Linear(mlp_last, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class EqCvT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        num_classes,
        s1_emb_dim = 192,
        s1_emb_kernel = 3,
        s1_emb_stride = 2,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 3,
        s1_depth = 2,
        s1_mlp_mult = 4,
        mlp_last = 192,
        dropout = 0.,
        sym_group = 'Circular', 
        N = 4,
        image_size=224,
        e2cc_mult_1 = 3,

    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        # Dihyderal Equivariance
        if sym_group == 'Dihyderal':
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)

        # Circular Equivariance
        elif sym_group == 'Circular':
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn_eq.FieldType(self.r2_act, channels*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        out_type = nn_eq.FieldType(self.r2_act, e2cc_mult_1*[self.r2_act.regular_repr])
        self.block1 = nn_eq.SequentialModule(
            nn_eq.MaskModule(in_type, image_size, margin=1),
            nn_eq.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn_eq.InnerBatchNorm(out_type),
            nn_eq.ReLU(out_type, inplace=True),
        )

        self.pool1 = nn_eq.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn_eq.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size

        prefix = 's1'
        config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

        layers.append(nn.Sequential(
            nn.Conv2d(c, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
            LayerNorm(config['emb_dim']),
            Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
        ))

        dim = config['emb_dim']


        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, mlp_last),
            nn.BatchNorm1d(mlp_last),
            nn.ELU(inplace=True),
            nn.Linear(mlp_last, num_classes),
        )

    def forward(self, x):
        x = nn_eq.GeometricTensor(x, self.input_type)
        
        # Equviarant layers
        x = self.block1(x)
        x = self.pool1(x)
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        x = x.tensor

        # normal layers
        x = self.layers(x)

        
        return x

    
def test_equivariance(model: torch.nn.Module, x, device, labels_map,
                    resize1, resize2, pad, to_tensor, to_gray, image_size, channels):
    # evaluate the `model` on 8 rotated versions of the input image `x`
    model.eval()
    
    wrmup = model(torch.randn(1, channels, image_size, image_size).to(device))
    del wrmup

    x = resize1(pad(x))

    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            x_transformed = to_tensor(to_gray(resize2(x.rotate(r*45., Image.BILINEAR)))) #.reshape(1, 1, 29, 29)
            x_transformed = x_transformed.unsqueeze(0).to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            
            angle = r * 45
            print("{:5d} : {}".format(angle, labels_map[y]))
    print('##########################################################################################')
