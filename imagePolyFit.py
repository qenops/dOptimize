import cv2, math
import numpy as np
from matplotlib import pyplot as plt

deg = 6
window = (0,-1)  # position of polynomial in image
width = 50  # size of surface in mm
baseImg = 'IMG_0369.tif'
img = 'focus_far.tif'
img = 'majorAxis/TIFF/focus_far_clean.png'
baseImg = 'majorAxis/TIFF/IMG_0419.tif'

def parseImg(img, baseImg=None, deg=12, width=57.25, window=(25,-25) ):
    baseImg = cv2.imread(baseImg,-1)
    img = cv2.imread(img,-1)
    diff = img if baseImg is None else abs(baseImg - img)
    median = cv2.medianBlur(diff, 21)
    edges = cv2.Canny(median,400,200)
    #
    # get signal in pixel space
    points = np.nonzero(edges.T[window[0]:window[1]])
    dif = points[0]-np.roll(points[0], 1)
    dif = np.nonzero(dif)[0]
    signal = np.zeros(img.shape[1]) + img.shape[0]
    x = points[0][dif]+window[0]
    y = points[1][dif]
    signal[x] = y
    t = np.arange(img.shape[1])
    #
    #plt.figure(1, figsize=(8, 16))
    #ax1 = plt.subplot(211)
    #ax1.imshow(edges,cmap = 'gray')
    #ax2 = plt.subplot(212)
    #ax2.axis([0, edges.shape[1], edges.shape[0], 0])
    #ax2.plot(signal)
    #plt.show()
    #
    # transition to real world space
    zero = math.floor(img.shape[1]/2)
    scale = (img.shape[1]-window[0]+window[1])/width
    x = (x - zero)/scale
    y = y/scale
    t = (t - zero)/scale
    #
    z = np.polyfit(x, y, deg)
    f = np.poly1d(z)
    z = np.array(z)
    z[range(1,deg,2)] = 0
    f_e = np.poly1d(z)
    #
    #t = np.arange(-zero/scale, zero/scale, 1/scale)
    plt.figure(2, figsize=(8, 16))
    ax1 = plt.subplot(211)
    ax1.imshow(edges,cmap = 'gray')
    ax2 = plt.subplot(212)
    ax2.axis([-zero/scale, zero/scale, edges.shape[0]/scale, 0])
    ax2.plot(t, f(t))
    ax2.plot(t, f_e(t))
    ax2.plot(t, signal/scale)
    plt.show()
    #
    return f, f_e
