import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_to_rgb(path, plot=False):
    """
    read image from path
    convert BGR to RGB
    """
    img = cv2.imread(path)
    # cv2 reads image as BGR, but matplotlib.pyplot shows image as RGB
    # in order to show image correctly, we need to convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if plot:
        plt.imshow(img)
        plt.show()
    return img

def save_from_rgb(img, path):
    """
    save image to path
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def hist_equalize(img, plot=False):
    """
    histogram equalization method (baseline)
    
    copied from courseware PPT
    """
    (r, g, b) = cv2.split(img)
    rh = cv2.equalizeHist(r)
    gh = cv2.equalizeHist(g)
    bh = cv2.equalizeHist(b)
    img = cv2.merge((rh, gh, bh))
    if plot:
        plt.imshow(img)
        plt.show()
    return img

def get_dark_channel(img, plot=False):
    """
    dark channel prior
    """
    r, g, b = cv2.split(img)
    dark_chn = cv2.min(cv2.min(r, g), b) / 255
    if plot:
        plt.imshow(dark_chn, cmap='gray')
        plt.show()
    return dark_chn


def rect_erode(dark_chn, size=10, plot=False):
    """
    erode image with rectangle kernel

    designed for dark channel method
    to choose the darkest pixel in a rectangle neighborhood window
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    # print(kernel)
    dark_chn_eroded = cv2.erode(dark_chn, kernel)
    if plot:
        plt.imshow(dark_chn_eroded, cmap='gray')
        plt.show()
    return dark_chn_eroded

def median_filter(img, size=5, plot=False):
    """
    median filter method
    """
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.medianBlur(img, size)
    # img = cv2.medianBlur(img, size)
    img = img / 255
    if plot:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img

def gaussian_filter(img, size=5, plot=False):
    """
    gaussian filter method
    """
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.GaussianBlur(img, (size, size), 0)
    img = img / 255
    if plot:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img

def bilateral_filter(img, size=5, plot=False):
    """
    bilateral filter method
    """
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.bilateralFilter(img, size, 135, 75)
    img = img / 255
    if plot:
        plt.imshow(img, cmap='gray')
        plt.show()
    return img

def get_transmission(dark_chn, omega=0.95, plot=False):
    """
    estimate transmission map from dark channel prior

    omega: a constant to control the degree of haze removal
    """
    transmission = 1 - omega * dark_chn
    if plot:
        plt.imshow(transmission, cmap='gray')
        plt.show()
    return transmission

def get_A(img, dark_chn, br_ratio=0.01, plot=False):
    """
    estimate atmospheric light A

    A is the pixel with maximum intensity in the original image
    """
    if br_ratio is not None:
        # use the top brightest pixels to estimate A
        top_num = int(br_ratio * img.shape[0] * img.shape[1])
        indices = np.argsort(dark_chn, axis=None)
        r = np.take_along_axis(img[:,:,0], indices[-top_num:], axis=None).mean()
        g = np.take_along_axis(img[:,:,1], indices[-top_num:], axis=None).mean()
        b = np.take_along_axis(img[:,:,2], indices[-top_num:], axis=None).mean()
        A = np.array([r, g, b])
    else:
        # use the top 1 brightest pixel to estimate A
        indices = np.where(dark_chn == np.max(dark_chn))
        i = indices[0][0]
        j = indices[1][0]
        # print('i: ', i)
        # print('j: ', j)
        A = img[i, j]
    
    if plot:
        print('A =', A)
    return A

def transmission_map(x, a=10, b=0.5):
    """
    cast a map function to x
    """
    # return np.minimum(1 / (1 + np.exp(-a * (x - b))), x)
    # return (np.exp(x) - 1) / (np.e - 1)
    # return np.log(x+1) / np.log(2)
    # return np.maximum(x, 0.1)
    # return (x - np.mean(x)) * 0.5 + np.mean(x)
    return x
    

def get_dehazed(img, transmission, A, t0=0.1, plot=False):
    """
    dehaze image

    t0: a constant to avoid division by zero
    """
    dehazed = np.zeros(img.shape)
    # print('img: ', img)
    # print('transmission: ', transmission)

    # draw map func
    # x = np.linspace(0, 1, 100)
    # y = transmission_map(x)
    # plt.plot(x, y)
    # plt.show()

    # plot map result
    # plt.imshow(transmission_map(np.maximum(transmission, t0)), cmap="gray")
    # plt.show()

    dehazed = (img.astype(np.int32) - A.astype(np.int32)) / \
        transmission_map(np.maximum(transmission, t0))[:, :, np.newaxis] + \
        A.astype(np.int32)
    dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
    # print('dehazed: ', dehazed)
    if plot:
        plt.imshow(dehazed)
        plt.show()
    return dehazed

def remove_haze(img, kernel_size=5, omega=0.98, br_ratio=0.01, t0=0.05, plot=False):
    if plot:
        print("暗通道图：")
    dark_chn = get_dark_channel(img, plot=plot)
    if plot:
        print('before dehaze, mean: ', np.mean(dark_chn))

    # dark_chn = rect_erode(dark_chn, size=kernel_size, plot=plot)
    # dark_chn = median_filter(dark_chn, size=kernel_size, plot=plot)
    if plot:
        print("双边滤波：")
    dark_chn = bilateral_filter(dark_chn, size=kernel_size, plot=plot)

    if plot:
        print("透射率：")
    transmission = get_transmission(dark_chn, omega=omega, plot=plot)

    if plot:
        print("计算模糊暗通道图以求解环境光强度...")
    dark_chn_blur = dark_chn
    if plot:
        print("取领域内最小值：")
    dark_chn_blur = rect_erode(dark_chn, size=25, plot=plot)
    if plot:
        print("高斯滤波：")
    dark_chn_blur = gaussian_filter(dark_chn_blur, size=25, plot=plot)
    if plot:
        print("环境光强度：")
    A = get_A(img, dark_chn_blur, br_ratio=br_ratio, plot=plot)

    if plot:
        print("去雾后：")
    dehazed = get_dehazed(img, transmission, A, t0=t0, plot=plot)

    dark_chn = get_dark_channel(dehazed, plot=False)
    if plot:
        print('after dehaze, mean: ', np.mean(dark_chn))

    if np.mean(dark_chn) > 0.35:
        if plot:
            print("去雾不完全，进一步去雾...")
        return remove_haze(dehazed, kernel_size=kernel_size, omega=omega, br_ratio=br_ratio, t0=t0, plot=plot)
    else:
        if plot:
            print("去雾结束")
    return dehazed