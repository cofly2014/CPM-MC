import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange

import os
import torch
from models.base.clip_fsar import load,tokenize

class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q

class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention_qkv(dim,
                                         Attention_qkv(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte=0.05, mlp_dim=2048,
                 dropout_ffn=0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                    # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                    PreNormattention(dim, Attention(dim, heads=heads, dim_head=dim_head_k, dropout=dropout_atte)),
                    FeedForward(dim, mlp_dim, dropout=dropout_ffn),
                ]))

    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x

class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class Attention_qkv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention_v3(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b  i d, b  j d -> b  i j', q, k) * self.scale

        attn = self.attend(dots)  # [30, 8, 8, 5]

        out = einsum('b  i j, b  j d -> b  i d', attn, v)

        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
# PostNormattention似乎没有用
class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

####################################################################################
def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(
                - cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(
                - cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]

class CLIP_CPMMC_FSAR(nn.Module):

    def __init__(self, args, cfg):
        super(CLIP_CPMMC_FSAR, self).__init__()
        self.argss = args
        self.args = cfg
        if cfg.VIDEO.HEAD.BACKBONE_NAME == "RN50":
            clip_backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME  #所有得训练样本 标签名称
            self.class_real_test = cfg.TEST.CLASS_NAME    #所有得测试样本 标签名称
            self.mid_dim = 1024
        elif cfg.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
            #将clip模型加载了进来
            clip_backbone, self.preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=cfg, jit=False)  # ViT-B/16
            self.backbone = clip_backbone.visual  # model.load_state_dict(state_dict)
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            # backbone, self.preprocess = load("RN50", device="cuda", cfg=cfg, jit=False)
            # self.backbone = backbone.visual model.load_state_dict(state_dict)
            # self.backbone = CLIP
            self.mid_dim = 512
        with torch.no_grad():
            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_train[int(ii)]) for ii in
                                 range(len(self.class_real_train))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_train[int(ii)]) for ii in
                                 range(len(self.class_real_train))]
            text_templete = tokenize(text_templete).cuda()
            self.text_features_train = clip_backbone.encode_text(text_templete)  # 对文本进行编码，数量就是support 中类别数量 例如 kinetics就是分别对64个单词进行编码

            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                text_templete = [self.args.TEST.PROMPT.format(self.class_real_test[int(ii)]) for ii in
                                 range(len(self.class_real_test))]
            else:
                text_templete = ["a photo of {}".format(self.class_real_test[int(ii)]) for ii in
                                 range(len(self.class_real_test))]
            text_templete = tokenize(text_templete).cuda()

            self.text_features_test = clip_backbone.encode_text(text_templete)

        self.mid_layer = nn.Sequential()
        self.classification_layer = nn.Sequential()
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)

        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads=8, dim_head_k=self.mid_dim // 8, dropout_atte=0.2)
        # set_trace()
        self.factor=1
        self.motion_conv1 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1,groups=1)
        self.motion_conv2 = nn.Conv1d(self.mid_dim // self.factor, self.mid_dim // self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_coeff = self.argss.motion_coeff
        self.normal_coeff = self.argss.normal_coeff
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))

        #self.token_linear1 = nn.Linear(self.mid_dim // self.factor, self.mid_dim // self.factor)
        #self.token_linear2 = nn.Linear(self.mid_dim // self.factor, self.mid_dim // self.factor)


    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CLIP visual encoder features.
        """
        if self.training:
            support_features = self.backbone(support_images).squeeze()  # self.backbone 为 clip的视觉分支visual ModifeidResnet 输出维度 [(way*shot*frames_number),1024]
            # os.system("nvidia-smi")
            target_features = self.backbone(target_images).squeeze()
            # os.system("nvidia-smi")
            dim = int(support_features.shape[1])
            support_features = support_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            support_features_text = None
        else:
            support_features = self.backbone(support_images).squeeze()
            target_features = self.backbone(target_images).squeeze()
            dim = int(target_features.shape[1])

            support_features = support_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            target_features = target_features.reshape(-1, self.argss.seq_len, dim)  # 输出维度 [(way*shot), frame_numbers,1024]
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]  # self.text_features_test 为clip的 文本分支 backbone.encode_text

        return support_features, target_features, support_features_text

    def get_motion_feats(self, support_features, target_features):

        support_features_conv = self.motion_conv1(support_features.permute(0,2,1))
        target_features_conv = self.motion_conv1(target_features.permute(0, 2, 1))
        support_features_conv = self.motion_conv2(support_features_conv)
        target_features_conv = self.motion_conv2(target_features_conv)

        support_features_motion_f = support_features_conv[:,:, 1:] - support_features.permute(0,2,1)[:,:, :-1]
        target_features_motion_f = target_features_conv[:,:, 1:] - target_features.permute(0,2,1)[:,:, :-1]

        support_features_motion_b = support_features_conv[:,:, :-1] - support_features.permute(0,2,1)[:,:, 1:]
        target_features_motion_b = target_features_conv[:,:, :-1] - target_features.permute(0,2,1)[:,:, 1:]

        support_features_motion = 0.5*(support_features_motion_f + support_features_motion_b)
        target_features_motion = 0.5*(target_features_motion_f + target_features_motion_b)

        return support_features_motion.permute(0,2,1), target_features_motion.permute(0,2,1)

    def forward(self, inputs):  # 获得support support labels, query, support real class
        support_images, support_labels, target_images, support_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs[ 'real_support_labels']
        target_real_class = inputs['real_target_labels']
        # set_trace()
        #一个episode中所有的样本对应的文本语句特征
        if self.training:  # 取5个support样本类型对应的样本标签编码，输出维度为5 1 1024
            self.context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
            self.target_context_support = self.text_features_train[target_real_class.long()].unsqueeze(1)
        else:
            self.context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            self.target_context_support = self.text_features_test[target_real_class.long()].unsqueeze(1)

        support_features, target_features, text_features = self.get_feats(support_images, target_images, support_real_class)
        support_features_motion, target_features_motion = self.get_motion_feats(support_features, target_features)

        class_text_logits = self.video2imagetext_adapter_mean(support_features, target_features)

        class_dists_g, class_dists_l, consist_distance = self.train_and_test(self.context_support, support_features, support_features_motion, support_labels, target_features, target_features_motion)

        class_dists_g = rearrange(class_dists_g, 'c q -> q c')

        return_dict = {"class_logits": class_text_logits,  #图文匹配
                       "logits_local": - class_dists_l,    #帧匹配
                       "logits_global": -class_dists_g,    #全局token匹配
                       "target_consist_distance": consist_distance  #一致性
                       }  # [5， 5] , [10 64]
        return return_dict

    def train_and_test(self, context_support, support_features, support_features_motion, support_labels, target_features, target_features_motion):

        support_features_motion, target_features_motion, support_features_motion_pro, support_features_contra_motion, target_features_contra_motion = self.text_eh_temporal_transformer(context_support, support_features_motion,
                                                                                                                                                        target_features_motion, support_labels, self.class_token_motion)
        support_features, target_features, support_features_pro, support_features_contra, target_features_contra = self.text_eh_temporal_transformer(context_support, support_features, target_features, support_labels, self.class_token)

        #part 1, 在测试中这个距离是不用的
        ################################motion 一致性损失 start#######################################################
        target_diff_motion = target_features_motion - target_features_contra_motion
        target_norm_sq_motion = torch.norm(target_diff_motion, dim=[-2, -1]) ** 2
        # target_consist_distance_motion = torch.div(target_norm_sq_motion, target_features_motion.shape[0])
        target_consist_distance_motion = torch.mean(target_norm_sq_motion)
        support_diff_motion = support_features_motion - support_features_contra_motion
        support_norm_sq_motion = torch.norm(support_diff_motion, dim=[-2, -1]) ** 2
        # support_consist_distance_motion = torch.div(support_norm_sq_motion, target_features_motion.shape[0])
        support_consist_distance_motion = torch.mean(support_norm_sq_motion)
        consist_distance_motion = support_consist_distance_motion + target_consist_distance_motion
        ################################motion 一致性损失 end#######################################################

        ################################normal 一致性损失 start#######################################################
        target_diff = target_features - target_features_contra
        target_norm_sq = torch.norm(target_diff, dim=[-2, -1]) ** 2
        # target_consist_distance = torch.div(target_norm_sq, target_features.shape[0])
        target_consist_distance = torch.mean(target_norm_sq)
        support_diff = support_features - support_features_contra
        support_norm_sq = torch.norm(support_diff, dim=[-2, -1]) ** 2
        # support_consist_distance = torch.div(support_norm_sq, support_features.shape[0])
        support_consist_distance = torch.mean(support_norm_sq)
        consist_distance = support_consist_distance + target_consist_distance
        print("support_consist_distance " + str(support_consist_distance))
        print("target_consist_distance " + str(target_consist_distance))
        ################################normal 一致性损失 end#######################################################
        #motion和normal总体一致性损失
        consist_distance = self.normal_coeff * consist_distance + self.motion_coeff * consist_distance_motion

        # part 2, 文章中part2 的系数为零，也就这部分代码其实不生效，仅我们实验中参考一下
        ################################motion 和 normal global 距离 start#########################################
        support_features_motion_g = support_features_motion[:, 0, :]
        class_dists_q2s_motion = self.global_distance(support_features_motion_g, support_labels, target_features_motion)

        support_features_g = support_features[:, 0, :]
        class_dists_q2s = self.global_distance(support_features_g, support_labels, target_features)

        class_dists_g = self.normal_coeff * class_dists_q2s + self.motion_coeff * class_dists_q2s_motion
        ################################motion 和 normal global 距离 end###########################################

        # part 3, 可以按需要需要换下面的这个self.otam_distance的方法就可以了
        ################################motion和normal的局部帧对齐距离，以及其总体距离 start#########################################
        cum_dists = self.otam_distance(support_features_pro[:, 1:, :], target_features[:, 1:, :])
        cum_dists_motion = self.otam_distance(support_features_motion_pro[:, 1:, :], target_features_motion[:, 1:, :])


        class_dists_l = self.normal_coeff * cum_dists + self.motion_coeff * cum_dists_motion
        ################################motion和normal的局部帧对齐距离，以及其总体距离 end#########################################

        return class_dists_g, class_dists_l, consist_distance
    def otam_distance(self, support_features, target_features):
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]
        support_features = rearrange(support_features, 'b s d -> (b s) d')  # 5 8 1024-->40  1024
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        frame_sim = cos_sim(target_features, support_features)  # 类别数量*每个类的样本数量， 类别数量
        frame_dists = 1 - frame_sim
        # dists维度为 query样本数量， support类别数量，帧数，帧数
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query  双向匹配还是单向匹配
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            cum_dists = OTAM_cum_dist_v2(dists)
        else:
            cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        return cum_dists
    def global_distance(self, support_features_g, support_labels, target_features):
        unique_labels = torch.unique(support_labels)
        # query to support
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s
        class_dists_q2s = [torch.sum(torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(support_labels, c)), dim=2), dim=1) for c in unique_labels]  # 每一帧的余弦相似度相加起来
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze()
        if hasattr(self.args.TRAIN, "USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')
        return class_dists_q2s


    def text_eh_temporal_transformer(self, context_support, support_features,target_features, support_labels, token):
        target_bs = target_features.shape[0]
        support_bs = support_features.shape[0]

        #一致性真实target token, 只有训练的时候用,元测试阶段其实可以没有这段代码，放在这里为了debug的时候，可以输出真实对比
        #沿着frame方向扩真，然后叠加到帧特征上
        s_number = support_features.shape[1]
        target_context_support_repeat = self.target_context_support.repeat(1, s_number, 1)
        target_features = target_features + 0.1 * target_context_support_repeat
        target_features_contra = torch.cat([self.target_context_support, target_features], dim=1)
        target_features_contra = self.context2(target_features_contra, target_features_contra, target_features_contra)

        #一致性的fake support token
        support_token = token.expand(support_bs, -1, -1)
        #沿着frame方向扩真，然后叠加到帧特征上
        s_number = support_features.shape[1]
        support_token_repeat = support_token.repeat(1, s_number, 1)
        support_features = support_features + 0.1 * support_token_repeat
        support_features_contra = torch.cat([support_token, support_features], dim=1)
        support_features_contra = self.context2(support_features_contra, support_features_contra, support_features_contra)

        #对比比较的fake target
        target_token= token.expand(target_bs, -1, -1)
        #沿着frame方向扩真，然后叠加到帧特征上
        s_number = support_features.shape[1]
        target_token_repeat = target_token.repeat(1, s_number, 1)
        target_features = target_features + 0.1 * target_token_repeat
        target_features = torch.cat([target_token, target_features], dim=1)
        target_features = self.context2(target_features, target_features, target_features)

        context_support = self.mid_layer(context_support)
        '''这里的MERGE_BEFORE参数是说，是先按类别聚合之后，再进行原型调制增强，还是先对样本进行调整增强之后再进行聚合 '''
        if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
            unique_labels = torch.unique(support_labels)
            support_features = [
                torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for
                c in unique_labels]
            support_features = torch.stack(support_features)
            context_support = [
                torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for
                c in unique_labels]
            context_support = torch.stack(context_support)
        #沿着frame方向扩真，然后叠加到帧特征上 增加context叠加

        #对比比较的real support
        s_number = support_features.shape[1]
        context_support_repeat = context_support.repeat(1, s_number, 1)
        support_features = support_features + 0.1 * context_support_repeat
        #support实际第一位用的是实际文本
        support_features = torch.cat([context_support,support_features], dim=1)  # 对应于R^txC 并上R^C--> R^(t+1)xC, 我们将文本特征沿着时间维度堆叠到相应的视频 Prototype modulation
        support_features = self.context2(support_features, support_features, support_features) # 对Prototype modulation之后的support进行temporal transformer操作

        if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
            pass
        else:
            unique_labels = torch.unique(support_labels)
            support_features_pro = [
                torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0)
                for c in unique_labels]
            support_features_pro = torch.stack(support_features_pro)
        return support_features, target_features, support_features_pro, support_features_contra, target_features_contra

    def video2imagetext_adapter_mean(self, support_features, target_features):
        if self.training:
            text_features = self.text_features_train
        else:
            text_features = self.text_features_test
        # 是否使用分类标签 也就是文本分支
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            feature_classification_in = torch.cat([support_features, target_features],  dim=0)  # 第一维度是support query一共的样本数量
            feature_classification = self.classification_layer(feature_classification_in).mean(1)  # 在第二位 帧的维度进行平均  维度为 [样本数量，1024] 公式2中的GAP操作
            class_text_logits = cos_sim(feature_classification, text_features) * self.scale  # 10 64， 10是10个视频样本，64是64个标签，对应于公式2
        else:
            class_text_logits = None
        return class_text_logits

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        gpus_use_number = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.argss.num_gpus > 1:
            self.backbone.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(gpus_use_number)])