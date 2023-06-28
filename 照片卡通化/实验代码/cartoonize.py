import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_to_rgb(path, plot=False):
    """
    read image from path
    convert BGR to RGB
    """
    img = cv2.imread(path)
    # cv2 reads image as BGR, but matplotlib.pyplot shows image as RGB
    # inorder to show image correctly, we need to convert BGR to RGB
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

def color_quantization(img, k=8, plot=False):
    """
    color quantization

    reduce the number of colors in an image
    """
    shape = img.shape
    # reshape image to (num_pixels, 3)
    img = img.reshape((-1, 3))
    # convert to float
    img = np.float32(img)
    # define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # apply kmeans
    ret, label, center = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to uint8
    center = np.uint8(center)
    # flatten the labels array
    label = label.flatten()
    # reconstruct the image
    res = center[label]
    # reshape the image
    res = res.reshape(shape)
    if plot:
        plt.imshow(res)
        plt.show()
    return res

def get_edge(img, blur_size=5, block_size=9, C=9, plot=False):
    """
    edge detection
    """
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur the image
    blur = cv2.medianBlur(gray, blur_size)
    # detect edges
    edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    if plot:
        plt.imshow(edge, cmap='gray')
        plt.show()
    return edge

def cartoonize(img, k=16, plot=True):
    edge1 = get_edge(img, 7, 45, 8, plot=plot)
    edge2 = get_edge(img, 7, 15, 5, plot=plot)
    edge2 = cv2.bitwise_or(edge1, edge2)
    if plot:
        plt.imshow(edge2, cmap='gray')
        plt.show()

    img = color_quantization(img, k=k, plot=plot)
    img = cv2.medianBlur(img, 7)
    if plot:
        plt.imshow(img)
        plt.show()

    edge1 = cv2.bitwise_and(img, img, mask=edge1)
    edge2 = cv2.bitwise_and(img, img, mask=edge2)

    img = cv2.addWeighted(img, 0.5, edge1, 0.5, 0)
    if plot:
        plt.imshow(img)
        plt.show()
    img = cv2.addWeighted(img, 0.5, edge2, 0.5, 0)
    if plot:
        plt.imshow(img)
        plt.show()

    return img