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
    x_diff = x.max() - x.min()
    y_diff = y.max() - y.min()

    transposed = False
    if (y_diff > x_diff):
        transposed = True
        temp = x
        x = y
        y = temp

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    scale = 80
    a,b = np.polyfit(x,y,1)
    x1 = x_mean - scale
    y1 = x1 * a + b
    x2 = x_mean + scale
    y2 = x2 * a + b

    if (transposed):
        temp = x1
        x1 = y1
        y1 = temp
        temp = x2
        x2 = y2
        y2 = temp
    return int(x1),int(y1),int(x2),int(y2)

def diff_gray(image, prev_image):
    result = (128 + (image / 2)) - (prev_image / 2)
    delta = 16
    result[result > 128 + delta] = 255
    result[result <= 128 - delta] = 255
    result[result != 255] = 0
    return result

def extract_abc(box):
    C = np.mean(box, axis=0)
    point1 = box[0]
    closest_index = -1
    closest_distance = 99999
    for i in range(1, 4):
        distance = np.linalg.norm(point1 - box[i])
        if distance < closest_distance:
            closest_distance = distance
            closest_index = i

    point2 = box[closest_index]
    A = np.mean([point1, point2], axis=0)

    box = np.delete(box, (closest_index), axis=0)
    box = np.delete(box, (0), axis=0)
    B = np.mean([box[0], box[1]], axis=0)
    return A,B,C

box_corner1 = (125, 16)
box_corner2 = (728, 452)
box_center = (433, 236)
box_center_radius = 72

def wall_distances(point):
    dist1 = point[0] - box_corner1[0]
    dist2 = point[1] - box_corner1[1]
    dist3 = box_corner2[0] - point[0]
    dist4 = box_corner2[1] - point[1]
    dist5 = np.linalg.norm(point - box_center) - box_center_radius
    return [dist1, dist2, dist3, dist4, dist5]


mask = np.zeros((480,852))
# mask.fill(0)
# cv2.rectangle(mask, box_corner1, box_corner2, 255, -1)
# empty[mask == 0] = (0, 0, 0)
# empty_gray[mask == 0] = 0

base_filename = "330 minute 1"
cap = cv2.VideoCapture("videos/"+base_filename+".mp4")

detector = cv2.SimpleBlobDetector_create()
prev_frame = np.zeros((480,852))
mask = np.zeros(prev_frame.shape)
count = 0
playVideo = True
step = True

ksize = (5, 5)
sigmaX = 1
threshold = 200

bg = cv2.imread("background.png")

features_filename = open("features/"+base_filename+'_features.csv', 'w')
features_filename.write("frame number,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y,"\
                        "A_x,A_y,B_x,B_y,C_x,C_y,dist_A_left,dist_A_top,dist_A_right,dist_A_bottom,dist_A_center,"\
                        "dist_B_left,dist_B_top,dist_B_right,dist_B_bottom,dist_B_center\n")
debug = False

prev_A = [0, 0]
prev_B = [0, 0]

ret, frame = cap.read()
count += 1
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):
    if playVideo or step:
        step = False
        ret, frame = cap.read()

        # Udacity image processing. produces good bug shape but with alot of reflective noise
        img = frame.astype(np.float)
        img = cv2.GaussianBlur(img, ksize, sigmaX)
        img = cv2.normalize(img - bg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # prepare mask which is diff between current and previous frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_frame = np.copy(gray)
        mask = diff_gray(gray, prev_frame)

        # mask out anything not the bug (prev / current) frame
        out_frame[mask == 0] = 0
        # clean "prev" frame location
        # out_frame[out_frame > 170] = 0
        # out_frame[out_frame != 0] = 255
        y, x = np.nonzero(out_frame)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        mask[:] = 0
        cv2.circle(mask, (int(x_mean), int(y_mean)), 45, 255, thickness=-1)
        out_frame[mask == 0] = 0

        # AND our implementation and udacity result to get best of both
        out_frame = np.bitwise_and(out_frame,img)
        out_frame[out_frame != 0] = 255
        out_frame = cv2.erode(out_frame, (10,10))
        out_frame = cv2.erode(out_frame, (10,10))
        # x1,y1,x2,y2 = lin_reg(out_frame)
        # cv2.line(out_frame, (x1,y1), (x2,y2), 255)

        # convex Hull magic to get bounding box of the bug
        ret, thresh = cv2.threshold(out_frame, 127, 255, 0)
        im2, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        y, x = np.nonzero(out_frame)
        coords = np.vstack([x, y])
        try:
            nonZero = cv2.findNonZero(out_frame)

            hull = cv2.convexHull(nonZero)

            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        except:
            print("error in frame {}".format(count))

        if debug:
            cv2.drawContours(frame, [box], 0, 255, 2)

        params = [count]
        for point in box:
            params.append(point[0])
            params.append(point[1])

        A,B,C = extract_abc(box)

        distance_A = np.linalg.norm(A - prev_A)
        distance_B = np.linalg.norm(A - prev_B)
        if (distance_A > distance_B):
            A,B = B,A

        distances_A = wall_distances(A)
        distances_B = wall_distances(B)

        params = params + [A[0], A[1], B[0], B[1], C[0], C[1]]
        params = params + distances_A + distances_B
        params = [str(x) for x in params]
        line = ",".join(params)

        # print(line)
        features_filename.write(line+"\n")
        cv2.circle(frame, (int(C[0]), int(C[1])), 5, (255, 0, 0), thickness=-1)
        cv2.circle(frame, (int(A[0]), int(A[1])), 5, (0, 255, 0), thickness=-1)
        cv2.circle(frame, (int(B[0]), int(B[1])), 5, (0, 0, 255), thickness=-1)

        # cv2.rectangle(frame, box_corner1, box_corner2, (255, 255, 0))
        # cv2.circle(frame, box_center, box_center_radius, (255, 255, 0))

        if debug:
            cv2.imshow('frame', frame.astype(np.uint8))
        else:
            if count % 100 == 0:
                print(count)

        prev_frame = gray
        prev_A = A
        prev_B = B
        count+=1

    char = cv2.waitKey(1)
    if char == ord('q'):
        break
    elif char == ord('n'):
        step = True
    elif char == ord('p'):
        playVideo = not playVideo

cap.release()
cv2.destroyAllWindows()

features_filename.close()