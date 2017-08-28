import numpy as np
import cv2
import time

def myPCA(img):
    y, x = np.nonzero(img)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x - x_mean
    y = y - y_mean
    x[np.abs(x)>40] = 0
    y[np.abs(y)>40] = 0
    
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    # x_v2, y_v2 = evec2
    scale = 30
    x1 = int(-x_v1*scale*2+x_mean)
    y1 = int(-y_v1*scale*2+y_mean)
    x2 = int(x_v1*scale*2+x_mean)
    y2 = int(y_v1*scale*2+y_mean)
    # plt.plot([x_v1*-scale*2+x_mean, x_v1*scale*2+x_mean],
             # [y_v1*-scale*2+y_mean, y_v1*scale*2+y_mean], color='red')
    # plt.plot([x_v2*-scale, x_v2*scale],
             # [y_v2*-scale, y_v2*scale], color='blue')
    return x1, y1, x2, y2


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies**i_order * y_indices**j_order).sum()

def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov

def moments(img):
    y, x = np.nonzero(img)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = moments_cov(img)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evec2
    scale = 30
    x1 = int(-x_v1*scale*2+x_mean)
    y1 = int(-y_v1*scale*2+y_mean)
    x2 = int(x_v1*scale*2+x_mean)
    y2 = int(y_v1*scale*2+y_mean)
    # plt.plot([x_v1*-scale*2+x_mean, x_v1*scale*2+x_mean],
             # [y_v1*-scale*2+y_mean, y_v1*scale*2+y_mean], color='red')
    # plt.plot([x_v2*-scale, x_v2*scale],
             # [y_v2*-scale, y_v2*scale], color='blue')
    return x1, y1, x2, y2


def lin_reg(img):
    y, x = np.nonzero(img)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x - x_mean
    y = y - y_mean
    x[np.abs(x)>40] = 0
    y[np.abs(y)>40] = 0
    a,b = np.polyfit(x,y,1)
    theta = np.tan(a)
    scale = 80
    x1 = scale*np.sin(theta) + x_mean
    y1 = scale*np.cos(theta) + y_mean
    x2 = -scale*np.sin(theta) + x_mean
    y2 = -scale*np.cos(theta) + y_mean
    return int(x1),int(y1),int(x2),int(y2)

cap = cv2.VideoCapture('330 minute 1.mp4')

detector = cv2.SimpleBlobDetector_create()
prev_frame = np.zeros((480,852))
count = 0
playVideo = True
while(cap.isOpened()):
    # if playVideo:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print (type(gray[0,0]))
    out_frame = (128 + (gray/2)) - (prev_frame/2)
    thres = 140
    out_frame[out_frame > thres] = 255
    out_frame[out_frame <= thres] = 0
    # out_frame[out_frame > 140] = 255
    # out_frame[out_frame < 120] = 255
    # out_frame[out_frame != 255] = 0
    # keypoints = detector.detect(out_frame.astype(np.uint8))
    # print(keypoints)
    # x1, y1, x2, y2 = moments(out_frame)
    # cv2.line(out_frame, (x1,y1), (x2,y2), 255)
    
    # im2,contours,hierarchy = cv2.findContours(out_frame, 1, 2)
    # cnt = contours[0]
    # print(cnt)
    y, x = np.nonzero(out_frame)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x1,y1,x2,y2 = lin_reg(out_frame)
    cv2.line(out_frame, (x1,y1), (x2,y2), 255)
    x1, y1, x2, y2 = myPCA(out_frame)
    cv2.line(out_frame, (x1,y1), (x2,y2), 255)
    
    cv2.circle(out_frame, (int(x_mean), int(y_mean)), 40, 255)
    cv2.imshow('frame',out_frame.astype(np.uint8))
    prev_frame = gray
    # print(np.max(out_frame), np.min(out_frame), np.mean(out_frame))
    count+=1
    # time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        playVideo = not playVideo

cap.release()
cv2.destroyAllWindows()

