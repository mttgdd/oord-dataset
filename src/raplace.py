import numpy as np
from skimage.transform import radon
from skimage.transform import rescale
import torch
from tqdm import tqdm

class SinoFFT(object):
    def __init__(self) -> None:
        self.scale = 0.25
    
    def call_impl(self, x):
        theta = np.arange(0,180)
        R = radon(x, theta=theta)
        R /= np.max(R)
        R = R.astype(np.float64)
        R = rescale(R, scale=self.scale) # original cv2.resize
        R = np.abs(np.fft.fft(R, axis=0))
        R = R[:int(R.shape[0]/2),:]
        R = (R-np.mean(R))/np.std(R)
        return R
    
    def __call__(self, x):
        device = x.device
        R = []
        for i in tqdm(range(x.shape[0])):
            x_elem = x[i].squeeze(0).detach().cpu().numpy()
            R.append(self.call_impl(x_elem))
        R = np.array(R)
        R = torch.Tensor(R).to(torch.float).to(device)
        return R

class RaPlace(object):
    def __init__(self) -> None:
        pass

    def call_impl(self, Mq, Mi):
        Fq = np.fft.fft(Mq, axis=0)
        Fn = np.fft.fft(Mi, axis=0)
        corrmap_2d = np.fft.ifft(Fq*np.conj(Fn), axis=0)
        corrmap = np.sum(corrmap_2d,axis=-1)
        maxval = np.max(corrmap)
        return maxval

    def __call__(self, Mq, Mi):
        mCauto = self.call_impl(Mq, Mq)
        mCqi = self.call_impl(Mq, Mi)
        return np.abs(mCauto-mCqi)