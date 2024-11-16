import cv2
import math
import numpy as np


class Dehaze:
    def __init__(self, im):
        self.im = im.astype('float64') / 255

    def DarkChannel(self, sz):
        b, g, r = cv2.split(self.im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def AtmLight(self, dark):
        [h, w] = self.im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz)
        imvec = self.im.reshape(imsz, 3)

        indices = darkvec.argsort()
        indices = indices[imsz - numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx

        return A

    def TransmissionEstimate(self, A, sz):
        omega = 0.95
        im3 = np.empty(self.im.shape, self.im.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = self.im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.DarkChannel(sz)
        return transmission

    def Guidedfilter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * im + mean_b
        return q

    def TransmissionRefine(self, im, et):
        im = (im * 255).astype(np.uint8)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 60
        eps = 0.0001
        t = self.Guidedfilter(gray, et, r, eps)
        return t


    def dehaze(self):
        dark = self.DarkChannel(15)
        A = self.AtmLight(dark)
        te = self.TransmissionEstimate(A, 15)
        t = self.TransmissionRefine(self.im, te)
        return t
