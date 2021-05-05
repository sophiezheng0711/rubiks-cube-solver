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


class Stream:
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.cube = []
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def run(self):
        while True:
            _, frame = self.camera.read()

            key = cv2.waitKey(10) & 0xFF

            if key == 27:
                break

            if key == 127:
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
                    solution = kociemba.solve(u + r + f + d + l + b)
                    print(solution)

            frame1 = frame.copy()

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

            dilated = preprocess_img(frame1)
            contours = squares_from_contours(dilated)
            if len(contours) == 9:
                frame1 = increase_brightness(frame1)
                sorted_contours = sort_contours_by_pos(contours)

                color_locs = color_locs_from_contours(sorted_contours, frame1, frame)
                if key == 32:
                    self.cube.append(color_locs)

            cv2.imshow("Rubik's cube solver", frame)
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream = Stream()
    stream.run()
