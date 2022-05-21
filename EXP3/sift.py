import warnings
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR, INTER_NEAREST, cvtColor, COLOR_BGR2GRAY
warnings.filterwarnings("ignore")  # 忽略警告


def generateBaseImage(image, sigma_0, camara_sigma):
    # 上采样得到基础图像
    # image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)  # 节省时间不采用上采样
    sigma_diff = np.sqrt(max((sigma_0 ** 2) - ((2 * camara_sigma) ** 2), 0.01))  # sigma_0为摄像头默认0.5高斯模糊
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def generateSigmas(sigma, n, s):
    k = 2 ** (1. / n)
    sigmas = np.zeros(s)
    sigmas[0] = sigma
    
    for image_index in range(1, s):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        sigmas[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return sigmas


def convolve(kernel, img, padding, strides):
    '''
    :param kernel:  输入的核函数
    :param img:    输入的图片
    :param padding:  需要填充的位置
    :param strides:   高斯核移动的步长
    :return:   返回卷积的结果
    '''
    result = None
    kernel_size = kernel.shape
    img_size = img.shape
    if len(img_size) == 3:  # 三通道图片就对每通道分别卷积  dstack和并
        channel = []
        for i in range(img_size[-1]):
            pad_img = np.pad(img[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            temp = []
            for j in range(0, img_size[0], strides[1]):
                temp.append([])
                for k in range(0, img_size[1], strides[0]):
                    val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                    k * strides[0]:k * strides[0] + kernel_size[1]]).sum()
                    temp[-1].append(val)
            channel.append(np.array(temp))
        
        channel = tuple(channel)
        result = np.dstack(channel)
    elif len(img_size) == 2:
        channel = []
        pad_img = np.pad(img, ((padding[0], padding[1]), (padding[2], padding[3])),
                         'constant')  # pad是填充函数 边界处卷积需要对边界外根据高斯核大小填0
        for j in range(0, img_size[0], strides[1]):  # 第j列 strides 是步长 本例步长为1 相当于遍历
            channel.append([])
            for k in range(0, img_size[1], strides[0]):  # 第i行
                val = (kernel * pad_img[j * strides[1]:j * strides[1] + kernel_size[0],
                                k * strides[0]:k * strides[0] + kernel_size[1]]).sum()  # 卷积的定义 相当于用高斯核做加权和
                channel[-1].append(val)
        
        result = np.array(channel)
    
    return result


# 产生高斯核
def GuassianKernel(sigma, dim):
    '''
    :param sigma: 标准差
    :param dim: 高斯核模板大小
    '''
    temp = [t - (dim // 2) for t in range(dim)]  # 生成二维高斯的x与y
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)  # 二维高斯公式
    return result


# 得到高斯金字塔和高斯差分金字塔
def getDoG(image, n, sigma0, s=None, O=None):
    if s is None:
        s = n + 3
    if O is None:
        O = int(np.log2(min(image.shape[0], image.shape[1]))) - 1
    
    sigmas = generateSigmas(sigma0, n, s)
    
    GuassianPyramid = []
    for octave_index in range(O):
        gaussian_images_in_octave = [image]
        for gaussian_kernel in sigmas[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        GuassianPyramid.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                       interpolation=INTER_NEAREST)  # 降采样
    GuassianPyramid = np.array(GuassianPyramid, dtype=object)
    DoG = []
    for gaussian_images_in_octave in GuassianPyramid:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))
        DoG.append(dog_images_in_octave)
    return DoG, GuassianPyramid, O


# 通过泰勒展开精调位置精调位置
def adjustLocalExtrema(DoG, o, s, x, y, contrastThreshold, edgeThreshold, sigma, n, SIFT_FIXPT_SCALE):
    SIFT_MAX_INTERP_STEPS = 5
    SIFT_IMG_BORDER = 5
    
    point = []
    
    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25
    
    img = DoG[o][s]
    i = 0
    while i < SIFT_MAX_INTERP_STEPS:
        if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
                img.shape[0] - SIFT_IMG_BORDER:
            return None, None, None, None
        
        img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]
        
        dD = [(img[x, y + 1] - img[x, y - 1]) * deriv_scale,
              (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
              (next[x, y] - prev[x, y]) * deriv_scale]
        
        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
        dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale
        
        H = [[dxx, dxy, dxs],
             [dxy, dyy, dys],
             [dxs, dys, dss]]
        
        X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))
        
        xi = -X[2]
        xr = -X[1]
        xc = -X[0]
        
        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break
        
        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))
        
        i += 1
    
    if i >= SIFT_MAX_INTERP_STEPS:
        return None, x, y, s
    if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
            img.shape[0] - SIFT_IMG_BORDER:
        return None, None, None, None
    
    t = (np.array(dD)).dot(np.array([xc, xr, xi]))
    
    contr = img[x, y] * img_scale + t * 0.5
    # 舍去低对比度的点
    if np.abs(contr) * n < contrastThreshold:
        return None, x, y, s
    
    # 边缘效应的去除。 利用Hessian矩阵的迹和行列式计算主曲率的比值
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None, x, y, s
    
    point.append((x + xr) * (1 << o))
    point.append((y + xc) * (1 << o))
    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
    point.append(sigma * np.power(2.0, (s + xi) / n) * (1 << o) * 2)
    
    return point, x, y, s


def GetMainDirection(img, r, c, radius, sigma, BinNum):
    expf_scale = -1.0 / (2.0 * sigma * sigma)
    
    X = []
    Y = []
    W = []
    temphist = []
    
    for i in range(BinNum):
        temphist.append(0.0)
    
    # 图像梯度直方图统计的像素范围
    k = 0
    for i in range(-radius, radius + 1):
        y = r + i
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c + j
            if x <= 0 or x >= img.shape[1] - 1:
                continue
            
            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])
            
            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1
    
    length = k
    
    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    
    # 计算直方图的每个bin
    for k in range(length):
        bin = int(np.round((BinNum / 360.0) * Ori[k]))
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]
    
    # smooth the histogram
    # 高斯平滑
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
    temphist.insert(0, temp[0])
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])
    temphist.insert(len(temphist), temp[3])  # padding
    
    hist = []
    for i in range(BinNum):
        hist.append(
            (temphist[i] + temphist[i + 4]) * (1.0 / 16.0) + (temphist[i + 1] + temphist[i + 3]) * (4.0 / 16.0) +
            temphist[i + 2] * (6.0 / 16.0))
    
    # 得到主方向
    maxval = max(hist)
    
    return maxval, hist


# 确定关键点
def LocateKeyPoint(DoG, sigma, GuassianPyramid, n, BinNum=36, contrastThreshold=0.04, edgeThreshold=10.0):
    SIFT_ORI_SIG_FCTR = 1.52
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
    SIFT_ORI_PEAK_RATIO = 0.8
    
    SIFT_INT_DESCR_FCTR = 512.0
    # SIFT_FIXPT_SCALE = 48
    SIFT_FIXPT_SCALE = 1
    
    KeyPoints = []
    O = len(DoG)
    S = len(DoG[0])
    for o in range(O):
        for s in range(1, S - 1):
            # 第一步：设定阈值
            threshold = 0.5 * contrastThreshold / (n * 255 * SIFT_FIXPT_SCALE)  # 用于阈值化，去噪
            img_prev = DoG[o][s - 1]
            img = DoG[o][s]
            img_next = DoG[o][s + 1]
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    val = img[i, j]
                    eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]),
                                            max(0, j - 1):min(j + 2, img_prev.shape[1])]
                    eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]),
                                       max(0, j - 1):min(j + 2, img.shape[1])]
                    eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]),
                                            max(0, j - 1):min(j + 2, img_next.shape[1])]
                    # 第二步：阈值化，在高斯差分金字塔中找极值
                    if np.abs(val) > threshold and \
                            ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (
                                    val >= eight_neiborhood_next).all())
                             or (val < 0 and (val <= eight_neiborhood_prev).all() and (
                                            val <= eight_neiborhood).all() and (
                                         val <= eight_neiborhood_next).all())):  # 如果某点大于阈值，并且 比周围8个点、上下2*9个点共26个点都大或都小，则认为是关键点
                        # 第三步：精调位置，通过函数2.1.1 adjustLocalExtrema：实现
                        point, x, y, layer = adjustLocalExtrema(DoG, o, s, i, j, contrastThreshold, edgeThreshold,
                                                                sigma, n, SIFT_FIXPT_SCALE)
                        if point == None:
                            continue
                        scl_octv = point[-1] * 0.5 / (1 << o)
                        # GetMainDirection：（确定极值点的位置以后就）求主方向
                        omax, hist = GetMainDirection(GuassianPyramid[o][layer], x, y,
                                                      int(np.round(SIFT_ORI_RADIUS * scl_octv)),
                                                      SIFT_ORI_SIG_FCTR * scl_octv, BinNum)
                        mag_thr = omax * SIFT_ORI_PEAK_RATIO
                        for k in range(BinNum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = BinNum - 1
                            if k < BinNum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[k] + hist[r2])
                                if bin < 0:
                                    bin = BinNum + bin
                                else:
                                    if bin >= BinNum:
                                        bin = bin - BinNum
                                temp = point[:]
                                temp.append((360.0 / BinNum) * bin)
                                KeyPoints.append(temp)
    
    return KeyPoints


# calcSIFTDescriptor：更小的计算描述符函数
def calcSIFTDescriptor(img, ptf, ori, scl, d, n, SIFT_DESCR_SCL_FCTR=3.0, SIFT_DESCR_MAG_THR=0.2,
                       SIFT_INT_DESCR_FCTR=512.0, FLT_EPSILON=1.19209290E-07):
    dst = []
    pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))]  # 坐标点取整
    # 旋转到主方向
    cos_t = np.cos(ori * (np.pi / 180))  # 余弦值
    sin_t = np.sin(ori * (np.pi / 180))  # 正弦值
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    # radius： 统计区域边长的一半
    radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
    cos_t /= hist_width
    sin_t /= hist_width
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))
    X = []
    Y = []
    RBin = []
    CBin = []
    W = []
    
    k = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            
            c_rot = j * cos_t - i * sin_t
            r_rot = j * sin_t + i * cos_t
            rbin = r_rot + d // 2 - 0.5
            cbin = c_rot + d // 2 - 0.5
            r = pt[1] + i
            c = pt[0] + j
            
            if rbin > -1 and rbin < d and cbin > -1 and cbin < d and r > 0 and r < rows - 1 and c > 0 and c < cols - 1:
                dx = (img[r, c + 1] - img[r, c - 1])
                dy = (img[r - 1, c] - img[r + 1, c])
                X.append(dx)
                Y.append(dy)
                RBin.append(rbin)
                CBin.append(cbin)
                W.append((c_rot * c_rot + r_rot * r_rot) * exp_scale)
                k += 1
    
    length = k
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    W = np.exp(np.array(W))
    
    for k in range(length):
        rbin = RBin[k]
        cbin = CBin[k]
        obin = (Ori[k] - ori) * bins_per_rad
        mag = Mag[k] * W[k]
        
        r0 = int(rbin)
        c0 = int(cbin)
        o0 = int(obin)
        rbin -= r0
        cbin -= c0
        obin -= o0
        
        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n
        
        # histogram update using tri-linear interpolation
        v_r1 = mag * rbin
        v_r0 = mag - v_r1
        
        v_rc11 = v_r1 * cbin
        v_rc10 = v_r1 - v_rc11
        
        v_rc01 = v_r0 * cbin
        v_rc00 = v_r0 - v_rc01
        
        v_rco111 = v_rc11 * obin
        v_rco110 = v_rc11 - v_rco111
        
        v_rco101 = v_rc10 * obin
        v_rco100 = v_rc10 - v_rco101
        
        v_rco011 = v_rc01 * obin
        v_rco010 = v_rc01 - v_rco011
        
        v_rco001 = v_rc00 * obin
        v_rco000 = v_rc00 - v_rco001
        
        idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
        hist[idx] += v_rco000
        hist[idx + 1] += v_rco001
        hist[idx + (n + 2)] += v_rco010
        hist[idx + (n + 3)] += v_rco011
        hist[idx + (d + 2) * (n + 2)] += v_rco100
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
        hist[idx + (d + 3) * (n + 2)] += v_rco110
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111
    
    # finalize histogram, since the orientation histograms are circular
    for i in range(d):
        for j in range(d):
            idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
            hist[idx] += hist[idx + n]
            hist[idx + 1] += hist[idx + n + 1]
            for k in range(n):
                dst.append(hist[idx + k])
    
    # copy histogram to the descriptor,
    # apply hysteresis thresholding
    # and scale the result, so that it can be easily converted
    # to byte array
    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += dst[k] * dst[k]
    thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR
    
    nrm2 = 0
    for i in range(length):
        val = min(dst[i], thr)
        dst[i] = val
        nrm2 += val * val
    nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)  # 归一化
    for k in range(length):
        dst[k] = min(max(dst[k] * nrm2, 0), 255)
    
    return dst


# calcDescriptors：计算描述符
def calcDescriptors(gpyr, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
    # SIFT_DESCR_WIDTH = 4，描述直方图的宽度
    # SIFT_DESCR_HIST_BINS = 8
    d = SIFT_DESCR_WIDTH
    n = SIFT_DESCR_HIST_BINS
    descriptors = []
    
    # keypoints(x,y,低8位组数次8位层数，尺度，主方向）
    for i in range(len(keypoints)):
        kpt = keypoints[i]
        o = kpt[2] & 255  # 组序号
        s = (kpt[2] >> 8) & 255  # 该特征点所在的层序号
        scale = 1.0 / (1 << o)  # 缩放倍数
        size = kpt[3] * scale  # 该特征点所在组的图像尺寸
        ptf = [kpt[1] * scale, kpt[0] * scale]  # 该特征点在金字塔组中的坐标
        img = gpyr[o][s]  # 该点所在的金字塔图像
        
        descriptors.append(calcSIFTDescriptor(img, ptf, kpt[-1], size * 0.5, d, n))  # calcSIFTDescriptor：更小的计算描述符函数
    return descriptors


def SIFT(img, showDoGimgs=False):
    # 若为三通道，转灰度图
    
    # 1. 建立高斯差分金字塔，
    SIFT_SIGMA = 1.6
    CAMERA_SIGMA = 0.5  # 假设的摄像头的尺度
    n = 3
    img = generateBaseImage(img, SIFT_SIGMA, CAMERA_SIGMA)
    DoG, GuassianPyramid, O = getDoG(img, n, SIFT_SIGMA)  # 得到高斯金字塔和高斯差分金字塔
    # return DoG, GuassianPyramid, O
    
    if showDoGimgs:
        # for dubugging
        plt.figure(1)
        for i in range(O):
            for j in range(n + 3):
                array = np.array(GuassianPyramid[i][j], dtype=np.float32)
                plt.subplot(O, n + 3, j + (i) * (n + 3) + 1)
                plt.imshow(array.astype(np.uint8), cmap='gray')
                plt.axis('off')
        plt.show()
        
        plt.figure(2)
        for i in range(O):
            for j in range(n + 2):
                array = np.array(DoG[i][j], dtype=np.float32)
                plt.subplot(O, n + 2, j + (i) * (n + 2) + 1)
                plt.imshow(array.astype(np.uint8), cmap='gray')
                plt.axis('off')
        plt.show()
    
    KeyPoints = LocateKeyPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)  # 确定关键点
    
    discriptors = calcDescriptors(GuassianPyramid, KeyPoints)  # 生成描述符
    
    return KeyPoints, discriptors


def getClusterCentures(features, num_words, target_path):
    des_matrix = np.zeros((1, 128))
    target_id = target_path.split("/")[-1].split(".")[0]
    assert target_id in features.keys()
    for k, v in features.items():
        if k != target_id:
            des_matrix = np.row_stack((des_matrix, np.array(v)))
    des_matrix = des_matrix[1:, :]
    
    # 计算聚类中心  构造视觉单词词典
    kmeans = KMeans(n_clusters=num_words, random_state=33)
    kmeans.fit(des_matrix)
    centres = kmeans.cluster_centers_  # 视觉聚类中心
    
    return centres, des_matrix


# 将特征描述转换为特征向量
def des2feature(des, num_words, centers):
    '''
    des: 单幅图像的SIFT特征描述
    num_words: 视觉单词数/聚类中心数
    centers: 聚类中心坐标   num_words*128
    return: feature vector 1*num_words
    '''
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(len(des)):
        feature_k_rows = np.ones((num_words, 128), 'float32')
        feature = des[i]
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centers) ** 2, 1)
        index = np.argmax(feature_k_rows)
        img_feature_vec[0][index] += 1
    return img_feature_vec


def get_all_features(features, num_words, centers):
    # 获取所有图片的特征向量
    allvec = np.zeros((len(features.keys()), num_words), 'float32')
    for k, v in features.items():
        allvec[int(k)] = des2feature(centers=centers, des=v, num_words=num_words)
    return allvec


def getNearestImg(feature, allvec, num_close):
    features = np.ones((allvec.shape[0], len(feature)), 'float32')
    features = features * feature
    dist = np.sum((features - allvec) ** 2, 1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]


def showImg(target_path, index, file_paths):
    paths = []
    for i in index:
        paths.append(file_paths[i])
    
    plt.figure(figsize=(20, 40))  # figsize 用来设置图片大小
    plt.subplot(6, 2, 1), plt.imshow(plt.imread(target_path)), plt.title('target_image')
    
    for i in range(len(index)):
        plt.subplot(6, 2, i + 3), plt.imshow(plt.imread(paths[i]))
    plt.show()


# 暴力搜索
def retrieval_img(target_path, features, centers, allvec, file_paths):
    num_close = 10 + 1
    num_words = 3
    target_id = target_path.split("/")[-1].split(".")[0]
    feature = des2feature(des=features[target_id], centers=centers, num_words=num_words)
    sorted_index = getNearestImg(feature, allvec, num_close).tolist()
    sorted_index.remove(int(target_id))
    showImg(target_path, sorted_index, file_paths)


def Image_Retrieval(target_path, features, file_paths, num_words=3):
    centers, des_matrix = getClusterCentures(features, num_words=num_words, target_path=target_path)
    allvec = get_all_features(features, num_words=3, centers=centers)
    retrieval_img(target_path, features, centers, allvec, file_paths)
