import cv2
import numpy as np
from constants import (
    MIN_SIDE_LENGTH_THRESHOLD,
    MAX_SIDE_LENGTH_THRESHOLD,
    SATURATION_INCREASE_VALUE,
    CUBE_SIDE,
    COLORS,
    COLORS2CODE,
    CIEDE2000,
    BGR2LAB,
)
import kociemba


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
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # perform edge detection with canny edge detector
    canny = cv2.Canny(blurred, 20, 40)
    # make lines thicker by dilating
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    return dilated


def squares_from_contours(dilated):
    contours, hierarchy = cv2.findContours(
        dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # we set items in the format of [approx, hierarchy, index]
    items = [
        (
            cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), closed=True),
            rel,
            i,
        )
        for i, (contour, rel) in enumerate(zip(contours, hierarchy[0]))
    ]

    # filter out polygons that do not have 4 sides
    # filter out polygons that do not match the side length thresholds
    # filter out polygons that are not convex
    items = list(
        filter(
            lambda item: len(item[0]) == 4
            and np.sqrt(cv2.contourArea(item[0])) > MIN_SIDE_LENGTH_THRESHOLD
            and np.sqrt(cv2.contourArea(item[0])) < MAX_SIDE_LENGTH_THRESHOLD
            and cv2.isContourConvex(item[0]),
            items,
        )
    )

    # find all existing indicies of contours after the first filter
    contour_idx_set = set([item[2] for item in items])

    # filter all contours that have children also in the filtered contours (overlap)
    # hierarchy is in format [Next, Previous, First_child, Parent]
    items = list(filter(lambda item: not item[1][2] in contour_idx_set, items))

    return [item[0] for item in items]


def increase_brightness(img, value=SATURATION_INCREASE_VALUE):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
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

    img = increase_brightness(img)

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

    if name != None:
        cv2.imwrite("output/%s.png" % name, img)
    return color_locs


def convert_face_to_string(color_locs):
    rows = ["".join([COLORS2CODE[color] for color in row]) for row in color_locs]
    return "".join(rows)


def image_example():
    colors1 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/1.jpeg"), None)
    )
    colors2 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/2.jpeg"), None)
    )
    colors3 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/3.jpeg"), None)
    )
    colors4 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/4.jpeg"), None)
    )
    colors5 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/5.jpeg"), None)
    )
    colors6 = convert_face_to_string(
        compute_color_locs(cv2.imread("data/6.jpeg"), None)
    )
    res = ""
    # In the order of U->R->F->D->L->B
    res += colors1 + colors4 + colors3 + colors6 + colors2 + colors5
    return res


if __name__ == "__main__":
    # img = cv2.imread("data/6.jpeg")
    # color_locs = compute_color_locs(img, "img6")
    # print(convert_face_to_string(color_locs))

    example_str = image_example()
    print(example_str)
    print(kociemba.solve(example_str))
