# -- coding:utf-8 --
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):  # d_model = feature_size,为线性输出的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1)  # [5000, 1, d_model],so need seq-len <= 5000
        pe = pe.unsqueeze(2).to(device)
        print("pe shape = {}".format(pe.shape))
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        x = x.permute(3,0,2,1)#B,D,N,T->T,B,N,D
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], x.shape[2],1)  # 使batch中每个序列都添加一样的位置特征
#一个一个输入
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).to(device)
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(self.norm(x), **kwargs)
        # print('out shape={}'.format(out.shape))
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            #nn.Conv2d(dim, hidden_dim,kernel_size=1),
            #nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim)
            # nn.Conv2d(hidden_dim, dim,kernel_size=1),
            # nn.BatchNorm2d(dim),
        ).to(device)

    def forward(self, x):
        #out = x.permute(0,3,1,2)
        out = self.net(x)
        #out = self.net(out).permute(0,2,3,1)
        # print('out shape={}'.format(out.shape))
        return out


class Attention(nn.Module):#Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.3):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)


        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Conv2d(dim, inner_dim,kernel_size=1, bias=False).to(device)
        self.to_k = nn.Conv2d(dim, inner_dim, kernel_size=1,bias=False).to(device)
        self.to_v = nn.Conv2d(dim, inner_dim, kernel_size=1,bias=False).to(device)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim).to(device),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #print('x shape = {}'.format(x.shape))
        B,T, N, D = x.shape
        q = self.to_q(x.permute(0,3,1,2)).permute(0,3,2,1)#b,t,n,d->b,d,t,n->b,n,t,d
        k = self.to_k(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        v = self.to_v(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        q = q.reshape(B,N,T,self.heads,self.dim_head).permute(0,1,3,2,4)# B,N,h,T,d
        k = k.reshape(B,N,T,self.heads,self.dim_head).permute(0,1,3,2,4)
        v = v.reshape(B,N,T,self.heads,self.dim_head).permute(0,1,3,2,4)
        #print('q k v shape = {}'.format(q.shape))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #print('dots shape = {}'.format(dots.shape))

        attn = self.attend(dots)
        # print('attn shape = {}'.format(attn.shape))
        attn = self.dropout(attn)
        # print('attn shape = {}'.format(attn.shape))

        out = torch.matmul(attn, v)
        # print('out shape={}'.format(out.shape))
        out = rearrange(out, 'b n h t d -> b t n (h d)')
        # print('out shape={}'.format(out.shape))
        out = self.to_out(out)
        # print('out shape={}'.format(out.shape))

        return out

#self.trans = Transformer(out_channel, out_channel*2, out_channel*4, out_channel*4)
class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, hid_dim,skip_dim, dropout=0.3,n_layer=1):
        super().__init__()

        self.encoder_layer = nn.Sequential(
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,dropout=dropout)),
            PreNorm(dim, FeedForward(dim, hid_dim, dropout=dropout))
        )
        self.encoder_blocks = nn.ModuleList([
            self.encoder_layer for i in range(n_layer)
        ]
        )
        self.skip_conv = nn.ModuleList([
            nn.Conv2d(in_channels=dim,out_channels=skip_dim,kernel_size=1) for i in range(n_layer)
        ]
        )


    def forward(self, x):
        x = x.transpose(0,1)#T,B,N,D->B,T,N,D
        skip = 0
        for i,layer in enumerate(self.encoder_blocks):
            x = layer(x)
            # print('x shape = {}'.format(x.shape))
            skip += self.skip_conv[i](x.permute(0,3,1,2))#b,d,t,n

        return skip

class TransAm(nn.Module):
    def __init__(self,in_channel=1,feature_size=64,hid_dim = 128,n_heads = 3, dim_head = 8,
                 skip_dim=64,num_layers=1,len_clossness=10,len_period=1,len_trend=1,external_dim=7,
                 map_heigh=32,map_width=32,dropout=0.1,ext_flag=True,Transformer_flag=True):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.mask = None
        self.feature_size = feature_size
        self.len_clossness = len_clossness
        self.len_period = len_period
        self.len_trend = len_trend
        self.external_dim = external_dim
        self.map_heigh = map_heigh
        self.map_width = map_width
        self.ext_flag = ext_flag
        self.Transformer_flag = Transformer_flag
        self.channel = in_channel
        self.c_net = nn.ModuleList([
            # input_embedding
            nn.Conv2d(in_channel,feature_size,3,1,1),#B,D,N,T
            nn.BatchNorm2d(feature_size), #B,D,N,T
            #pos_encoder
            PositionalEncoding(feature_size),#T,B,N,D
            #transformer_encoder
            Transformer(dim=feature_size, heads=n_heads, dim_head=dim_head, hid_dim=hid_dim,
                        skip_dim=skip_dim, dropout=dropout, n_layer=num_layers),#B,T,N,D->b,d,t,n

        ])
        self.c_pred=nn.ModuleList([
            # to_predict
            nn.ReLU(skip_dim*len_clossness),
            nn.Conv2d(in_channels=len_clossness*skip_dim, out_channels=in_channel, kernel_size=1),

        ])
        if self.len_period > 0:
            self.p_net = nn.ModuleList([
                # input_embedding
                nn.Conv2d(in_channel, feature_size, 3, 1, 1),  # B,D,N,T
                nn.BatchNorm2d(feature_size),  # B,D,N,T
                # pos_encoder
                PositionalEncoding(feature_size),  # T,B,N,D
                # transformer_encoder
                Transformer(dim=feature_size, heads=n_heads, dim_head=dim_head, hid_dim=hid_dim,
                            skip_dim=skip_dim, dropout=dropout, n_layer=num_layers),  # B,T,N,D->b,d,t,n

            ])
            self.p_pred = nn.ModuleList([
                # to_predict
                nn.ReLU(skip_dim * len_period),
                nn.Conv2d(in_channels=len_period * skip_dim, out_channels=in_channel, kernel_size=1),

            ])

        if self.len_trend > 0:
            self.t_net = nn.ModuleList([
                # input_embedding
                nn.Conv2d(in_channel, feature_size, 3, 1, 1),  # B,D,N,T
                nn.BatchNorm2d(feature_size),  # B,D,N,T
                # pos_encoder
                PositionalEncoding(feature_size),  # T,B,N,D
                # transformer_encoder
                Transformer(dim=feature_size, heads=n_heads, dim_head=dim_head, hid_dim=hid_dim,
                            skip_dim=skip_dim, dropout=dropout, n_layer=num_layers),  # B,T,N,D->b,d,t,n

            ])
            self.t_pred = nn.ModuleList([
                # to_predict
                nn.ReLU(skip_dim * len_trend),
                nn.Conv2d(in_channels=len_trend * skip_dim, out_channels=in_channel, kernel_size=1),

            ])
        if self.ext_flag:
            self.ext_net = nn.Sequential(
                nn.Linear(self.external_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.map_heigh * self.map_width)
            )
        self.w_c = nn.Parameter(torch.rand((in_channel, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_p = nn.Parameter(torch.rand((in_channel, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_t = nn.Parameter(torch.rand((in_channel, self.map_heigh, self.map_width)), requires_grad=True)

        # self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.to_predict[1].bias.data.zero_() #将 m 模型的偏置项（bias）全部清零。
        self.to_dim[1].bias.data.zero_()
        self.to_predict[1].weight.data.uniform_(-initrange, initrange)
        self.to_dim[1].weight.data.uniform_(-initrange, initrange)

    def forward_branch(self, branch, x_in):
        for layer in branch:
            x_in = layer(x_in)
        return x_in

    def forward(self,xc,xp,xt,ext):
    # def forward(self, xc, xp, xt):
    #     # src with shape (input_window, batch_len, 1)
    #     if self.mask is None or self.mask.size(0) != len(x):
    #         device = x.device
    #         mask = self._generate_square_subsequent_mask(len(x)).to(device)
    #         self.mask = mask

        B, Tc,_, H, W = xc.shape
        N = H * W
        #print("x shape = {}".format(x.shape))
        xc = xc.reshape(B, self.len_clossness, self.channel, N).permute(0,2,3,1)# B,T,D,N->B,D,N,T
        # print("input shape = {}".format(input.shape))
        c_out = self.forward_branch(self.c_net, xc)
        c_out = c_out.reshape(B,-1,H,W)#b,d,t,n->
        c_out = self.forward_branch(self.c_pred,c_out) # B,D,N,T->B,in_channel,H,W
        # print("out shape = {}".format(out.shape))

        if self.len_period > 0:
            xp = xp.reshape(B, self.len_period, self.channel, N).permute(0,2,3,1)  # B,T,N,D->B,D,N,T
            p_out = self.forward_branch(self.p_net, xp)
            p_out = p_out.reshape(B, -1, H, W)  # b,d,t,n->
            p_out = self.forward_branch(self.p_pred,p_out)  # B,D,N,T->B,in_channel,H,W
        else:
            p_out = 0

        if self.len_trend > 0:
            xt = xt.reshape(B, self.len_trend, self.channel, N).permute(0,2,3,1)  # B,T,N,D->B,D,N,T
            t_out = self.forward_branch(self.t_net, xt)
            t_out = t_out.reshape(B, -1, H, W)  # b,d,t,n->
            t_out = self.forward_branch(self.t_pred,t_out)  # B,D,N,T->B,in_channel,H,W
        else:
            t_out = 0

        pre = self.w_c.unsqueeze(0) * c_out + \
          self.w_p.unsqueeze(0) * p_out + \
          self.w_t.unsqueeze(0) * t_out

        if self.ext_flag:
            ext_out = self.ext_net(ext).view([-1, 1, self.map_heigh, self.map_width])
            pre += ext_out

        return torch.tanh(pre)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) #torch.triu返回矩阵上三角部分
        #print('mask = {}'.format(mask))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

