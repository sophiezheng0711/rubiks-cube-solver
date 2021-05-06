import cv2
import kociemba
from edge_color_detector import (
    convert_face_to_string,
    preprocess_img,
    squares_from_contours,
    increase_brightness,
    sort_contours_by_pos,
    color_locs_from_contours,
)
from constants import COLORS
import numpy as np


def draw_face_structure(x, y, frame, color_locs, w=30):
    for i in range(3):
        for j in range(3):
            if color_locs != None:
                r, g, b = COLORS[color_locs[j][i]]
                cv2.rectangle(
                    frame,
                    (x + i * w, y + j * w),
                    (x + (i + 1) * w, y + (j + 1) * w),
                    (b, g, r),
                    -1,
                )
            cv2.rectangle(
                frame,
                (x + i * w, y + j * w),
                (x + (i + 1) * w, y + (j + 1) * w),
                (0, 255, 0),
                2,
            )


def draw_arrows(key, img, center_locs):
    letter = key[0]
    if letter == "F":
        if len(key) == 2 and key[1] == "'":
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[1][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[1][0][1]) ** 2
                )
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_x = int((center_locs[2][1][0] - center_locs[2][0][0]) / 2)

            cv2.arrowedLine(
                img,
                (center_locs[2][0][0] + half_x, center_locs[2][0][1]),
                center_locs[2][1],
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
        elif len(key) == 2 and key[1] == "2":
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[1][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[1][0][1]) ** 2
                )
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_y = int((center_locs[1][2][1] - center_locs[0][2][1]) / 2)

            cv2.arrowedLine(
                img,
                (center_locs[0][2][0], center_locs[0][2][1] + half_y),
                center_locs[1][2],
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[1][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[1][0][1]) ** 2
                )
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_y = int((center_locs[1][2][1] - center_locs[0][2][1]) / 2)

            cv2.arrowedLine(
                img,
                (center_locs[0][2][0], center_locs[0][2][1] + half_y),
                center_locs[1][2],
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
    elif letter == "R":
        if len(key) == 2 and key[1] == "'":
            cv2.arrowedLine(img, center_locs[0][2], center_locs[2][2], (0, 255, 0), 9)
        elif len(key) == 2 and key[1] == "2":
            cv2.arrowedLine(img, center_locs[2][2], center_locs[0][2], (0, 255, 0), 9)
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.arrowedLine(img, center_locs[2][2], center_locs[0][2], (0, 255, 0), 9)
    elif letter == "U":
        if len(key) == 2 and key[1] == "'":
            cv2.arrowedLine(img, center_locs[0][0], center_locs[0][2], (0, 255, 0), 9)
        elif len(key) == 2 and key[1] == "2":
            cv2.arrowedLine(img, center_locs[0][2], center_locs[0][0], (0, 255, 0), 9)
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.arrowedLine(img, center_locs[0][2], center_locs[0][0], (0, 255, 0), 9)
    elif letter == "L":
        if len(key) == 2 and key[1] == "'":
            cv2.arrowedLine(img, center_locs[2][0], center_locs[0][0], (0, 255, 0), 9)
        elif len(key) == 2 and key[1] == "2":
            cv2.arrowedLine(img, center_locs[0][0], center_locs[2][0], (0, 255, 0), 9)
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.arrowedLine(img, center_locs[0][0], center_locs[2][0], (0, 255, 0), 9)
    elif letter == "B":
        if len(key) == 2 and key[1] == "'":
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[2][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[2][0][1]) ** 2
                )
                * 1.5
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_r = int(radius / 2)

            cv2.arrowedLine(
                img,
                (center_locs[0][2][0] + half_r, center_locs[0][2][1] + half_r),
                (center_locs[1][2][0] + half_r, center_locs[1][2][1] + half_r),
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
        elif len(key) == 2 and key[1] == "2":
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[2][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[2][0][1]) ** 2
                )
                * 1.5
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_r = int(radius / 4)

            cv2.arrowedLine(
                img,
                (center_locs[2][0][0] - half_r, center_locs[2][0][1] + half_r),
                (center_locs[2][1][0] - half_r, center_locs[2][1][1] + half_r),
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            radius = int(
                np.sqrt(
                    (center_locs[1][1][0] - center_locs[2][0][0]) ** 2
                    + (center_locs[1][1][1] - center_locs[2][0][1]) ** 2
                )
                * 1.5
            )
            cv2.ellipse(
                img,
                center_locs[1][1],
                (radius, radius),
                0,
                135,
                360,
                (0, 255, 0),
                9,
            )

            half_r = int(radius / 4)

            cv2.arrowedLine(
                img,
                (center_locs[2][0][0] - half_r, center_locs[2][0][1] + half_r),
                (center_locs[2][1][0] - half_r, center_locs[2][1][1] + half_r),
                (0, 255, 0),
                9,
                tipLength=0.5,
            )
    elif letter == "D":
        if len(key) == 2 and key[1] == "'":
            cv2.arrowedLine(img, center_locs[2][2], center_locs[2][0], (0, 255, 0), 9)
        elif len(key) == 2 and key[1] == "2":
            cv2.arrowedLine(img, center_locs[2][0], center_locs[2][2], (0, 255, 0), 9)
            cv2.putText(
                img,
                "2",
                center_locs[1][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.arrowedLine(img, center_locs[2][0], center_locs[2][2], (0, 255, 0), 9)


class Stream:
    def __init__(self, solution):
        self.camera = cv2.VideoCapture(1)
        self.cube = []
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.isSolved = False if solution == None else True
        self.solution = [] if solution == None else solution
        self.solution_idx = 0

    def run(self):
        while True:
            _, frame = self.camera.read()

            frame1 = frame.copy()

            key = cv2.waitKey(10) & 0xFF

            if key == 27:
                break

            draw_face_structure(
                150, 20, frame, None if len(self.cube) <= 0 else self.cube[0]
            )
            draw_face_structure(
                60, 110, frame, None if len(self.cube) <= 1 else self.cube[1]
            )
            draw_face_structure(
                150, 110, frame, None if len(self.cube) <= 2 else self.cube[2]
            )
            draw_face_structure(
                240, 110, frame, None if len(self.cube) <= 3 else self.cube[3]
            )
            draw_face_structure(
                330, 110, frame, None if len(self.cube) <= 4 else self.cube[4]
            )
            draw_face_structure(
                150, 200, frame, None if len(self.cube) <= 5 else self.cube[5]
            )

            if not self.isSolved:
                if key == 127 and len(self.cube) > 0:
                    del self.cube[-1]

                if key == 13:
                    if len(self.cube) == 6:
                        color_2_code = {}
                        color_2_code[self.cube[0][1][1]] = "U"
                        color_2_code[self.cube[1][1][1]] = "L"
                        color_2_code[self.cube[2][1][1]] = "F"
                        color_2_code[self.cube[3][1][1]] = "R"
                        color_2_code[self.cube[4][1][1]] = "B"
                        color_2_code[self.cube[5][1][1]] = "D"
                        u = convert_face_to_string(self.cube[0], color_2_code)
                        l = convert_face_to_string(self.cube[1], color_2_code)
                        f = convert_face_to_string(self.cube[2], color_2_code)
                        r = convert_face_to_string(self.cube[3], color_2_code)
                        b = convert_face_to_string(self.cube[4], color_2_code)
                        d = convert_face_to_string(self.cube[5], color_2_code)
                        self.solution = kociemba.solve(u + r + f + d + l + b).split(" ")
                        self.isSolved = True
                        print(self.solution)

                dilated = preprocess_img(frame1)
                contours = squares_from_contours(dilated)
                if len(contours) == 9:
                    frame1 = increase_brightness(frame1)
                    sorted_contours = sort_contours_by_pos(contours)

                    color_locs, _ = color_locs_from_contours(
                        sorted_contours, frame1, frame
                    )

                    if key == 32 and len(self.cube) < 6:
                        self.cube.append(color_locs)

            else:
                dilated = preprocess_img(frame1)
                contours = squares_from_contours(dilated)

                if key == 3 and self.solution_idx + 1 < len(self.solution):
                    self.solution_idx += 1

                if key == 2 and self.solution_idx - 1 >= 0:
                    self.solution_idx -= 1

                if len(contours) == 9:
                    frame1 = increase_brightness(frame1)
                    sorted_contours = sort_contours_by_pos(contours)

                    _, center_locs = color_locs_from_contours(
                        sorted_contours, frame1, frame1
                    )
                    draw_arrows(self.solution[self.solution_idx], frame, center_locs)

            cv2.imshow("Rubik's cube solver", frame)
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream = Stream(None)
    stream.run()
