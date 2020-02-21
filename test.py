# @Author: Narsi Reddy <narsi>
# @Date:   2020-02-14T19:34:50-06:00
# @Email:  sdhy7@mail.umkc.edu
# @Last modified by:   narsi
# @Last modified time: 2020-02-19T21:22:01-06:00
import numpy as np
from matplotlib import pyplot as plt


def compress(I_org, model):

    e_ = 512
    c_ = 16
    d_ = e_ // c_
    pad_ = 4

    w, h = I_org.size

    comp_w_new = np.ceil(w/c_)
    comp_h_new = np.ceil(h/c_)

    new_w = int(e_ * np.ceil(w/e_))
    new_h = int(e_ * np.ceil(h/e_))

    com_w = new_w // c_
    com_h = new_h // c_

    I = np.uint8(I_org).copy()
    I = np.pad(I, ((0, int(new_h - h)),
                   (0, int(new_w - w)),
                   (0, 0)), mode = "reflect")
    I = Image.fromarray(I)


    I1 = np.float32(I)/255.0
    I1 = np.transpose(I1, [2, 0, 1])

    Enout = np.zeros((3, com_h, com_w))
    Enout_w = np.zeros((3, com_h, com_w))
    for i in list(np.arange(0, new_h, e_)):
        for j in list(np.arange(0, new_w, e_)):
            if i == 0:
                x1 = int(i)
                x2 = int((i + e_) + (pad_*2*c_))
            else:
                x1 = int(i - (pad_*c_))
                x2 = int((i + e_) + (pad_*c_))

            if j == 0:
                y1 = int(j)
                y2 = int((j + e_) + (pad_*2*c_))
            else:
                y1 = int(j - (pad_*c_))
                y2 = int((j + e_) + (pad_*c_))
            It = torch.from_numpy(np.expand_dims(I1[:, x1:x2, y1:y2], 0))
            Xe = model(It.cuda())
            Xe = (Xe + 1.0)/2.0
            print(Xe.size())
            print([ x1//c_,x2//c_, y1//c_, y2//c_])
            Enout[:, x1//c_:x2//c_, y1//c_:y2//c_] += Xe.data.squeeze().cpu().numpy()
            Enout_w[:, x1//c_:x2//c_, y1//c_:y2//c_] += 1.0

    Enout = Enout/Enout_w
    Enout = np.uint8(255 * Enout.transpose([1, 2, 0]))

    Enout = Image.fromarray(Enout).crop((0, 0, comp_w_new, comp_h_new))

    return Enout


def decompress(EnIn, model):

    e_ = 512
    c_ = 16
    d_ = e_ // c_
    pad_ = 4

    w, h = int(EnIn.text['w']), int(EnIn.text['h'])

    comp_w_new = np.ceil(w/c_)
    comp_h_new = np.ceil(h/c_)

    new_w = int(e_ * np.ceil(w/e_))
    new_h = int(e_ * np.ceil(h/e_))

    com_w = new_w // c_
    com_h = new_h // c_


    I = np.zeros((3,new_h,new_w), dtype = np.float32)
    I_w = np.zeros((3,new_h,new_w), dtype = np.float32)

    EnIn = np.uint8(EnIn).copy()
    EnIn = np.pad(EnIn, ((0, int(new_h - EnIn.shape[0])),
                         (0, int(new_w - EnIn.shape[1])),
                         (0, 0)), mode = "reflect")


    EnIn = np.float32(EnIn)/255.0
    EnIn = np.transpose(EnIn, [2, 0, 1])
    for i in list(np.arange(0, com_h, d_)):
        for j in list(np.arange(0, com_w, d_)):

            if i == 0:
                x1 = int(i)
                x2 = int((i + d_) + pad_*2)
            else:
                x1 = int(i - pad_)
                x2 = int((i + d_) + pad_)

            if j == 0:
                y1 = int(j)
                y2 = int((j + d_) + pad_*2)
            else:
                y1 = int(j - pad_)
                y2 = int((j + d_) + pad_)

            It = torch.from_numpy(np.expand_dims(EnIn[:, x1:x2, y1:y2], 0))
            It = It * 2.0 - 1.0
            Xe = model(It.cuda())
            I[:, x1*c_:x2*c_, y1*c_:y2*c_] += np.clip(Xe.data.squeeze().cpu().numpy(), 0, 1)
            I_w[:, x1*c_:x2*c_, y1*c_:y2*c_] += 1.0

    I = I/I_w

    I = np.uint8(255 * I.transpose([1, 2, 0]))
    I = Image.fromarray(I).crop((0, 0, w, h))

    return I










w, h = (2048, 1363)

e_ = 512
c_ = 16
d_ = e_ // c_
pad_ = 4

new_w = int(e_ * np.ceil(w/e_))
new_h = int(e_ * np.ceil(h/e_))

com_w = new_w // c_
com_h = new_h // c_

En = np.zeros((1, com_w, com_h))
for i in list(np.arange(0, new_w, e_)):
    for j in list(np.arange(0, new_h, e_)):

        if i == 0:
            x1 = int(i)
            x2 = int((i + e_) + pad_*2*c_)
        else:
            x1 = int(i - pad_*c_)
            x2 = int((i + e_) + pad_*c_)

        if j == 0:
            y1 = int(j)
            y2 = int((j + e_) + pad_*2*c_)
        else:
            y1 = int(j - pad_)
            y2 = int((j + e_) + pad_*c_)

        En[:, x1//c_:x2//c_, y1//c_:y2//c_] += 1

plt.imshow(En[0, ...])

Iout = np.zeros((1, new_w, new_h))

for i in list(np.arange(0, com_w, d_)):
    for j in list(np.arange(0, com_h, d_)):

        if i == 0:
            x1 = int(i)
            x2 = int((i + d_) + pad_*2)
        else:
            x1 = int(i - pad_)
            x2 = int((i + d_) + pad_)

        if j == 0:
            y1 = int(j)
            y2 = int((j + d_) + pad_*2)
        else:
            y1 = int(j - pad_)
            y2 = int((j + d_) + pad_)

        Iout[:, x1*c_:x2*c_, y1*c_:y2*c_] += 1

plt.imshow(np.uint8(255 * (Iout[0, ...]/4)), cmap = 'gray')

np.unique(En[0, ...])
