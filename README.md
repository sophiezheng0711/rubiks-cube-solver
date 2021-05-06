# rubiks-cube-solver

A Computer Vision-based 3x3x3 Rubik's Cube Solver that uses edge and color detection and the Kociemba algorithm to solve the cube.

## User Manual

Run `python3 stream.py` to start the live stream version of the solver. You can see an empty structure of the cube on the top left corner.
Adjust your cube's positioning until the colors are identified and are outlined in the app. Then, press **SPACE** to add it to the structure. Do so in the order of the top side, the left side, the front side, the right side, the back side, and the bottom side. After all sides are added and confirmed, press **ENTER** to start the solving phase. If you make mistakes of adding faces, you can always delete them by pressing the **DELETE** button. You can press **ESC** to exit at any time. After the cube is solved, the app will give you a step by step tutorial on how to solve the cube. Hold the front face of the cube towards the camera, and arrows on the app will appear to help you (see [this manual](https://ruwix.com/the-rubiks-cube/notation/) for reference). You can go back to a step by pressing the **LEFT ARROW** key, or skip to a future step by pressing the **RIGHT ARROW** key.

![Alt Text](scan_demo.gif)

To see a full demo, go [here](https://drive.google.com/file/d/1wYjh0bRD8KbtMLeGF6P8vV11lJxACVbp/view?usp=sharing).

## Edge Detection

We use a OpenCV-based approach of applying a gaussian filter, a canny edge detector, and computing contours of the resulting image. Then, we approximate the polygon version of the contours, and find the ones that satisfy the following criterions:

- the contour has 4 edges
- the contour is convex
- the contour has side length within a fine-tuned threshold
- the contour is not nested with another contour

If we obtain 9 squares after the processing, this step is done.

## Color Detection

Initially, we tried the Cartesian distance between the RGB values, and that did not work well. Thus, we used the [CIEDE2000](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000) method. This method requires a conversion from RGB to [LAB](https://en.wikipedia.org/wiki/CIELAB_color_space#CIELAB) colorspace. After some research, this method proved to work well with our given case.

## Kociemba Algorithm

We solve the Rubik's cube using the Kociemba Algorithm. Initially, we also looked at Korf's Algorithm, which involved a pattern database with a heuristic search algorithm, but that simply required too much unnecessary space and time. We used the python `kociemba` package. See [Wikipedia reference](https://en.wikipedia.org/wiki/Optimal_solutions_for_Rubik%27s_Cube#Kociemba's_algorithm).
