import cv2
import numpy as np
from constants import (
    MIN_SIDE_LENGTH_THRESHOLD,
    MAX_SIDE_LENGTH_THRESHOLD,
    SATURATION_INCREASE_VALUE,
    CUBE_SIDE,
    COLORS,
    CLOSE_THRESHOLD,
    CIEDE2000,
    BGR2LAB,
)


def find_color(input_color):
    diffs = [
        np.sum(np.array([(input_color[i] - COLORS[color][i]) ** 2 for i in range(3)]))
        for color in COLORS
    ]
    min_idx = np.argmin(np.array(diffs))
    return {i: key for i, key in enumerate(COLORS)}[min_idx]


def find_color_ciede2000(lab):
    dists = []
    for color in COLORS:
        r, g, b = COLORS[color]
        lab2 = BGR2LAB((b, g, r))
        dists.append((color, CIEDE2000(lab, lab2)))
    name, _ = min(dists, key=lambda item: item[1])
    return name


def preprocess_img(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur to remove noise, kernel size set to large because
    # don't care about details, we just want to extract the color.
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # perform edge detection with canny edge detector
    canny = cv2.Canny(blurred, 20, 40)
    # make lines thicker by dilating
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    return dilated


def is_close_to(cnt1, cnt2):
    corner1 = (
        np.sqrt(
            (cnt1[0][0][0] - cnt2[0][0][0]) ** 2 + (cnt1[0][0][1] - cnt2[0][0][1]) ** 2
        )
        <= CLOSE_THRESHOLD
    )
    corner2 = (
        np.sqrt(
            (cnt1[2][0][0] - cnt2[2][0][0]) ** 2 + (cnt1[2][0][1] - cnt2[2][0][1]) ** 2
        )
        <= CLOSE_THRESHOLD
    )
    return corner1 and corner2


def squares_from_contours(dilated):
    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    approxs = [
        cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), closed=True)
        for contour in contours
    ]

    # filter out polygons that do not have 4 sides
    # filter out polygons that do not match the side length thresholds
    approxs = list(
        filter(
            lambda approx: len(approx) == 4
            and np.sqrt(cv2.contourArea(approx)) > MIN_SIDE_LENGTH_THRESHOLD
            and np.sqrt(cv2.contourArea(approx)) < MAX_SIDE_LENGTH_THRESHOLD,
            approxs,
        )
    )

    # results = []

    # # filtering overlapping contours by checking two opposite corners
    # flagged_approxs = set()

    # for i in range(len(approxs)):
    #     a1 = approxs[i]
    #     if i in flagged_approxs:
    #         continue
    #     for j in range(len(approxs)):
    #         if i == j:
    #             continue
    #         a2 = approxs[j]
    #         if is_close_to(a1, a2):
    #             flagged_approxs.add(j)
    #             break
    #     results.append(a1)

    return approxs


def increase_saturation(img, value=SATURATION_INCREASE_VALUE):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # TODO: does not work well with very saturated colors
    s += value
    final_hsv = cv2.merge((h, s, v))
    img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img2


def sort_contours_by_pos(contours):
    # find top left square
    contours.sort(key=lambda item: item[0][0][0] + item[0][0][1])
    first = contours[0]

    # sort by rows
    contours.sort(key=lambda item: item[0][0][1] - first[0][0][1])
    sorted_contours = []
    for i in range(0, len(contours), CUBE_SIDE):
        temp = contours[i : i + CUBE_SIDE]
        temp.sort(key=lambda item: item[0][0][0])
        sorted_contours.extend(temp)
    return sorted_contours


def compute_color_locs(img, name):
    dilated = preprocess_img(img)

    contours = squares_from_contours(dilated)

    # img = increase_saturation(img)

    sorted_contours = sort_contours_by_pos(contours)

    color_locs = []

    # extract color
    for i in range(len(sorted_contours)):
        approx = sorted_contours[i]
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 10)
        mid_x = 0
        mid_y = 0
        for point in approx:
            mid_x += point[0][0]
            mid_y += point[0][1]
        mid_x /= 4
        mid_y /= 4
        b, g, r = img[int(mid_y)][int(mid_x)]

        lab = BGR2LAB((b, g, r))

        r, g, b = COLORS[find_color_ciede2000(lab)]
        if i % 3 == 0:
            color_locs.append([find_color_ciede2000(lab)])
        else:
            color_locs[-1].append(find_color_ciede2000(lab))
        cv2.fillPoly(img, [approx], (b, g, r))

    cv2.imwrite("output/%s.png" % name, img)
    return color_locs


if __name__ == "__main__":
    img = cv2.imread("data/4.jpeg")
    print(compute_color_locs(img, "img4"))
