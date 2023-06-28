import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

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

def load_from_pkl(path):
    """
    load data from pkl file
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_to_pkl(data, path):
    """
    save data to pkl file
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pkl_cache(func, path):
    """
    cache data to pkl file
    """
    def wrapper(*args, **kwargs):
        if os.path.exists(path):
            print("load data from cache: ", path)
            data = load_from_pkl(path)
        else:
            data = func(*args, **kwargs)
            print("save data to cache: ", path)
            save_to_pkl(data, path)
        return data
    return wrapper

# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
import copyreg
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
def _pickle_dmatch(match):
    return cv2.DMatch, (match.queryIdx, match.trainIdx, match.distance)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
copyreg.pickle(cv2.DMatch().__class__, _pickle_dmatch)

def sift_kp(img):
    """
    use SIFT to find keypoints
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # 参考网页：OpenCV - SIFT 参数及计算返回结果说明
    # https://www.aiuai.cn/aifarm1639.html
    return kp, des

def match_kp(des1, des2):
    """
    match keypoints
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    return matches

def get_good_matches(matches, ratio=0.75):
    """
    get good keypoint matches
    """
    good = []
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    return good

def stitch_img(left, right, save_suffix, save_path="output", cache_path="cache", reversed=False):
    """
    stitch two images

    if reversed is True, then stitch left image to right image 
    otherwise, stitch right image to left image
    """
    if reversed:
        save_suffix += "_r"
        left, right = right[:,::-1,:], left[:,::-1,:]

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    
    # find keypoints
    # finding keypoints is time-consuming, so we cache the result
    left_kp, left_des = pkl_cache(sift_kp, os.path.join(cache_path, "left_kp"+save_suffix+".pkl"))(left)
    # print(left_kp)
    # print(left_des)
    right_kp, right_des = pkl_cache(sift_kp, os.path.join(cache_path, "right_kp"+save_suffix+".pkl"))(right)
    # print(right_kp)
    # print(right_des)
    print("left keypoint number: ", len(left_kp))
    print("left descriptor shape: ", left_des.shape)
    print("right keypoint number: ", len(right_kp))
    print("right descriptor shape: ", right_des.shape)

    # plot keypoints
    left_kp_img = cv2.drawKeypoints(left, left_kp, None)
    right_kp_img = cv2.drawKeypoints(right, right_kp, None)
    plt.imshow(left_kp_img)
    plt.show()
    plt.imshow(right_kp_img)
    plt.show()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_from_rgb(left_kp_img, os.path.join(save_path, "left"+save_suffix+"_kp.jpg"))
    save_from_rgb(right_kp_img, os.path.join(save_path, "right"+save_suffix+"_kp.jpg"))

    # match keypoints
    # matching keypoints is time-consuming, so we cache the result
    matches = pkl_cache(match_kp, os.path.join(cache_path, "matches"+save_suffix+".pkl"))(left_des, right_des)
    # matches = match_kp(left_des, right_des)
    print("matches number: ", len(matches))
    good = get_good_matches(matches, ratio=0.5)
    print("good matches number: ", len(good))
    img_match = cv2.drawMatches(left, left_kp, right, right_kp, good, None, flags=2)
    plt.imshow(img_match)
    plt.show()
    save_from_rgb(img_match, os.path.join(save_path, "match"+save_suffix+".jpg"))

    # calculate homography matrix
    left_good_kp = np.float32([left_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    right_good_kp = np.float32([right_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    print("left good keypoint number: ", len(left_good_kp))
    print("right good keypoint number: ", len(right_good_kp))
    print("left good keypoint shape: ", left_good_kp.shape)
    print("right good keypoint shape: ", right_good_kp.shape)
    # 参考网页：findHomography()函数详解
    # https://blog.csdn.net/fengyeer20120/article/details/87798638
    M, mask = cv2.findHomography(right_good_kp, left_good_kp, cv2.RANSAC, 5.0)
    print("M: ", M)
    print("mask shape: ", mask.shape)

    # warp perspective
    # get size
    h, w = right.shape[:2]
    # calc size after warp
    right_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    left_corners = cv2.perspectiveTransform(right_corners, M)
    print("left_corners: ", left_corners)
    h_max = max(left_corners[0][0][1], left_corners[1][0][1], left_corners[2][0][1], left_corners[3][0][1])
    w_max = max(left_corners[0][0][0], left_corners[1][0][0], left_corners[2][0][0], left_corners[3][0][0])
    h_min = min(left_corners[0][0][1], left_corners[1][0][1], left_corners[2][0][1], left_corners[3][0][1])
    w_min = min(left_corners[0][0][0], left_corners[1][0][0], left_corners[2][0][0], left_corners[3][0][0])
    h_min = max(h_min, 0)
    w_min = max(w_min, 0)
    print("h_max: ", h_max)
    print("w_max: ", w_max)
    print("left shape: ", left.shape)
    print("right shape: ", right.shape)
    h_bigger = max(left.shape[0], h_max)
    w_bigger = max(left.shape[1], w_max)
    print("h_bigger: ", h_bigger)
    print("w_bigger: ", w_bigger)
    img_out = cv2.warpPerspective(right, M, (int(w_bigger), int(h_bigger)))
    # img_out = cv2.warpPerspective(right, M, (left.shape[1]+right.shape[1], left.shape[0]))
    plt.imshow(img_out)
    plt.show()

    # sitch images
    
    # get mask
    mask = cv2.threshold(cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY)[1] / 255
    mask[left.shape[0]:, :] = 0
    mask[:, left.shape[1]:] = 0
    print("mask shape: ", mask.shape)
    plt.imshow(mask, cmap="gray")
    plt.show()

    # get weight
    def exp_map(x, a=1):
        return (np.exp(a * x)-1)/(np.exp(a)-1)
    def log_map(x, a=1):
        return np.log(a*x+1)/np.log(a+1)
    
    def long_dark_map(x, a=1):
        return (np.exp(np.log(a+1)*x)-1)/a
    def short_dark_map(x, a=1):
        return np.log(a*x+1)/np.log(a+1)
    def long_light_map(x, a=1):
        return 1-long_dark_map(1-x, a=a)
    def short_light_map(x, a=1):
        return 1-short_dark_map(1-x, a=a)

    # get light region bounds of mask
    top_bound = np.logical_or.accumulate(mask>0, axis=1)[:, -1].argmax()
    bottom_bound = mask.shape[0] - np.logical_or.accumulate(mask>0, axis=1)[::-1, -1].argmax()
    left_bound = np.logical_or.accumulate(mask>0, axis=0)[-1, :].argmax()
    right_bound = mask.shape[1] - np.logical_or.accumulate(mask>0, axis=0)[-1, ::-1].argmax()
    print("left_bound: ", left_bound)
    print("right_bound: ", right_bound)
    print("top_bound: ", top_bound)
    print("bottom_bound: ", bottom_bound)

    lb = left_bound
    rb = right_bound
    lr_lin = (np.arange(0, rb-lb)/(rb-lb)).reshape(1, -1)
    tb = top_bound
    bb = bottom_bound
    tb_lin = (np.arange(0, bb-tb)/(bb-tb)).reshape(-1, 1)
    
    
    weight_top = np.zeros((bb-tb, rb-lb))
    weight_bottom = np.zeros((bb-tb, rb-lb))
    weight_left = np.zeros((bb-tb, rb-lb))
    weight_right = np.zeros((bb-tb, rb-lb))
    a=100
    # top is dark
    weight_top = long_dark_map(tb_lin, a) * short_light_map(lr_lin, a)
    # left is dark
    weight_left = short_light_map(tb_lin, a) * long_dark_map(lr_lin, a)
    # right is light
    weight_right = short_dark_map(tb_lin, a) * long_light_map(lr_lin, a)
    if left_corners[1][0][1] > left.shape[0]:
        # bottom is light
        weight_bottom = long_light_map(tb_lin, a) * short_dark_map(lr_lin, a)
    else:
        # bottom is dark
        weight_bottom = weight_top[::-1, :] - 1
        weight_left = weight_left * short_light_map(tb_lin[::-1, :], a)
        weight_right = weight_right * short_dark_map(tb_lin[::-1, :], a)
    weight_top -= 1
    weight_left -= 1
    
    plt.imshow(weight_top, cmap="gray")
    plt.show()
    plt.imshow(weight_bottom, cmap="gray")
    plt.show()
    plt.imshow(weight_left, cmap="gray")
    plt.show()
    plt.imshow(weight_right, cmap="gray")
    plt.show()
    
    weight_top *= 1-tb_lin * 1.25
    weight_left *= 1-lr_lin * 1.5
    weight_right *= lr_lin * 1.5
    weight_bottom *= tb_lin

    weight_balanced \
        = weight_top \
        + weight_bottom \
        + weight_left \
        + weight_right
    plt.imshow(weight_top, cmap="gray")
    plt.show()
    plt.imshow(weight_bottom, cmap="gray")
    plt.show()
    plt.imshow(weight_left, cmap="gray")
    plt.show()
    plt.imshow(weight_right, cmap="gray")
    plt.show()
    plt.imshow(weight_balanced, cmap="gray")
    plt.show()

    ideal_mu = 1
    ideal_sigma = 1.25
    mu = np.mean(weight_balanced)
    sigma = np.std(weight_balanced)
    print("mu: ", mu)
    print("sigma: ", sigma)
    weight_balanced = (weight_balanced - mu) / sigma * ideal_sigma + ideal_mu
    weight_balanced = np.minimum(weight_balanced, 1)
    weight_balanced = np.maximum(weight_balanced, 0)
    plt.imshow(weight_balanced, cmap="gray")
    plt.show()
    weight_out = np.zeros((mask.shape[0], mask.shape[1]))
    weight_out[tb:bb, lb:rb] = weight_balanced

    # get left-right linear weight
    # weight_lr = np.zeros((mask.shape[0], mask.shape[1]))
    # weight_lr[tb:bb, lb:rb] = np.array([np.arange(0, rb-lb)/(rb-lb)])
    # weight_lr_exp = exp_map(weight_lr, a=10)

    # add mask to weight
    weight = (mask * weight_out).reshape(mask.shape[0], mask.shape[1], 1)
    # weight = (mask * weight_lr).reshape(mask.shape[0], mask.shape[1], 1)
    # weight = (mask * weight_lr_exp).reshape(mask.shape[0], mask.shape[1], 1)
    # weight = weight * 0
    # weight = (mask * np.ones((mask.shape[0], mask.shape[1]))).reshape(mask.shape[0], mask.shape[1], 1)
    # weight = (mask * 0.5 * np.ones((mask.shape[0], mask.shape[1]))).reshape(mask.shape[0], mask.shape[1], 1)
    print("weight shape: ", weight.shape)
    plt.imshow(weight, cmap="gray")
    plt.show()
    
    # stitch
    weight = weight[0:left.shape[0], 0:left.shape[1], :]
    img_out[0:left.shape[0], 0:left.shape[1]] = \
        left * (1 - weight) + img_out[0:left.shape[0], 0:left.shape[1]] * weight
    
    if reversed:
        img_out = img_out[:, ::-1, :]

    save_from_rgb(img_out, os.path.join(save_path, "stitch"+save_suffix+".jpg"))
    plt.imshow(img_out)
    plt.show()

    return img_out