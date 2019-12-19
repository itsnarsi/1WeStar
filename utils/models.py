# @Author: Narsi Reddy <narsi>
# @Date:   2019-12-18T20:16:34-06:00
# @Last modified by:   narsi
# @Last modified time: 2019-12-18T21:46:55-06:00
import torch
torch.manual_seed(29)
from torch import nn
import torch.nn.functional as F

def pixel_unshuffle(x, factor = 2):
    b, c, h, w = x.size()
    new_c = c * factor**2
    new_h = h // factor
    new_w = w // factor
    x = x.contiguous().view(b, c, new_h, factor, new_w, factor)
    x = x.permute(0,1,3,5,2,4).contiguous().view(b, new_c, new_h, new_w)
    return x


class binTanH(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        c = input.clamp(min=-1, max =1)
        c[c > 0.0] = 1
        c[c < 0.0] = -1
        return c
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0
        return grad_input, None

class BinTANH(torch.nn.Module):
    def __init__(self):
        super(BinTANH, self).__init__()
        self.quantclip = binTanH

    def forward(self, input):
        return self.quantclip.apply(input)

WN_ = lambda x: nn.utils.weight_norm(x)

class BLOCK_1x1ReLU3x3(nn.Module):
    def __init__(
        self, in_ch, out_ch, ker, act = nn.ReLU(True)
        ):
        super(BLOCK_1x1ReLU3x3, self).__init__()

        feat = []
        feat.append(
            WN_(nn.Conv2d(in_ch, out_ch, 1)),
            act,
            nn.ReflectionPad2d(ker//2),
            WN_(nn.Conv2d(out_ch, out_ch, ker))
        )
        self.feat = nn.Sequential(*feat)

    def forward(self, x):
        x = torch.cat((x, self.feat(x)), dim = 1)
        return x

class BLOCK_3x3(nn.Module):
    def __init__(
        self, in_ch, out_ch, ker
        ):
        super(BLOCK_3x3, self).__init__()

        feat = []
        feat.append(
            nn.ReflectionPad2d(ker//2),
            WN_(nn.Conv2d(out_ch, out_ch, ker))
        )
        self.feat = nn.Sequential(*feat)

    def forward(self, x):
        x = self.feat(x)
        return x


class MODEL1(nn.Module):
    def __init__(
        self, feat = 48, nb_blocks = 5, shuffle = 4
        ):
        self.factor = shuffle

        self.ps_in   = nn.PixelShuffle(1/shuffle)
        self.en_in   = BLOCK_3x3(3 * shuffle**2, feat, ker = 3)
        self.en_feat = nn.Sequential(BLOCK_1x1ReLU3x3(feat, feat),
                                     *[BLOCK_1x1ReLU3x3(feat*2, feat) for i in range(nb_blocks-1)])
        self.en_out  = BLOCK_3x3(feat*2, 8, ker = 3)
        self.en_bin  = BinTANH()


        self.de_in   = BLOCK_3x3(8, feat, ker = 3)
        self.de_feat = nn.Sequential(BLOCK_1x1ReLU3x3(feat, feat),
                                     *[BLOCK_1x1ReLU3x3(feat*2, feat) for i in range(nb_blocks-1)])
        self.en_in   = BLOCK_3x3(feat*2, 3 * shuffle**2, ker = 3)

    def forward(self, x):
        # Encode Input
        en_x = pixel_unshuffle(x, self.factor)
        en_x = self.en_bin(self.en_out(self.en_feat(self.en_in(en_x))))
        # Decode Input
        de_x = self.de_out(self.de_feat(self.de_in(en_x)))
        de_x = F.pixel_shuffle(de_x, self.factor)
        return de_x
