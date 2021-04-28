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


if __name__ == "__main__":
    img = cv2.imread("data/1.jpeg")

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

    # contours?
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
        # 2. all four lines must be roughly the same length
        # 3. all four corners must be roughly 90 degrees
        if (
            len(approx) == 4
            and cv2.isContourConvex(approx)
            and side_length > MIN_SIDE_LENGTH_THRESHOLD
            and side_length < MAX_SIDE_LENGTH_THRESHOLD
        ):
            approxs.append(approx)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s += SATURATION_INCREASE_VALUE
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # find top left square
    approxs.sort(key=lambda item: item[0][0][0] + item[0][0][1])
    first = approxs[0]

    # sort by rows
    approxs.sort(key=lambda item: item[0][0][1] - first[0][0][1])
    rows = []
    for i in range(0, len(approxs), CUBE_SIDE):
        temp = approxs[i : i + CUBE_SIDE]
        temp.sort(key=lambda item: item[0][0][0])
        rows.extend(temp)

    rgbs = []

    # extract color
    for i in range(len(rows)):
        approx = rows[i]
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
        print(find_color((r, g, b)))
        cv2.fillPoly(img, [approx], (b, g, r))

    cv2.imwrite("output/target.png", img)
