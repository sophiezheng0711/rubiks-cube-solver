import cv2
import numpy as np
from constants import (
    MIN_SIDE_LENGTH_THRESHOLD,
    MAX_SIDE_LENGTH_THRESHOLD,
    SATURATION_INCREASE_VALUE,
    CUBE_SIDE,
    COLORS,
)


def find_color(input_color):
    diffs = [
        np.sum(np.array([(input_color[i] - COLORS[color][i]) ** 2 for i in range(3)]))
        for color in COLORS
    ]
    min_idx = np.argmin(np.array(diffs))
    return {i: key for i, key in enumerate(COLORS)}[min_idx]


def preprocess_img(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur to remove noise, kernel size set to large because
    # don't care about details, we just want to extract the color.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # perform edge detection with canny edge detector
    canny = cv2.Canny(blurred, 20, 40)
    # make lines thicker by dilating
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    return dilated


def squares_from_contours(dilated):
    contours, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    approxs = []

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        area = cv2.contourArea(approx)
        side_length = np.sqrt(area)
        # render if satisfy the following conditions:
        # 1. 4 corners
        # 2. [side_length], or square root of [area] is
        # within the defined threshold (requires some tuning)
        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
            and side_length > MIN_SIDE_LENGTH_THRESHOLD
            and side_length < MAX_SIDE_LENGTH_THRESHOLD
        ):
            approxs.append(approx)

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

    img = increase_saturation(img)

    sorted_contours = sort_contours_by_pos(contours)

    color_locs = []

    # extract color
    for i in range(len(sorted_contours)):
        approx = sorted_contours[i]
        cv2.drawContours(img, [approx], -1, (0, 255, 255), 5)
        mid_x = 0
        mid_y = 0
        for point in approx:
            mid_x += point[0][0]
            mid_y += point[0][1]
        mid_x /= 4
        mid_y /= 4
        b, g, r = img[int(mid_y)][int(mid_x)]
        r, g, b = COLORS[find_color((r, g, b))]
        if i % 3 == 0:
            color_locs.append([find_color((r, g, b))])
        else:
            color_locs[-1].append(find_color((r, g, b)))
        cv2.fillPoly(img, [approx], (b, g, r))

    cv2.imwrite("output/%s.png" % name, img)
    return color_locs


if __name__ == "__main__":
    img = cv2.imread("data/1.jpeg")
    print(compute_color_locs(img, "img1"))
