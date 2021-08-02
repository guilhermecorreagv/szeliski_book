import numpy as np
import cv2
from numpy.core.defchararray import center
from numpy.core.fromnumeric import mean

possible_transforms = [
    'trans', 'rigid', 'sim', 'affinex', 'affiney', 'affineskew', 'projx',
    'projy', 'save', 'affine', 'proj'
]
center_point = [600, 500]  # Center point of the axis


def init_canvas(next_color):
    canvas = 255 * np.ones((900, 1200, 3)).astype('uint8')
    # Menu bar
    global bar_height, center_point
    border_thick = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas[:bar_height, :, :] = 127
    canvas[bar_height:bar_height + border_thick + 1, :, :] = 0
    canvas[:border_thick + 1, :, :] = 0
    canvas[:bar_height, :border_thick, :] = 0
    canvas[:bar_height, -border_thick:, :] = 0

    # Transformation options
    cv2.putText(canvas, 'Translation', (10, 65), font, 1, (0, 0, 0), 2)
    canvas[:bar_height, 200:200 + border_thick, :] = 0
    cv2.putText(canvas, 'Rigid', (220, 65), font, 1, (0, 0, 0), 2)
    canvas[:bar_height, 310:310 + border_thick, :] = 0
    cv2.putText(canvas, 'Similarity', (330, 65), font, 1, (0, 0, 0), 2)
    canvas[:bar_height, 490:490 + border_thick, :] = 0
    cv2.putText(canvas, 'Affine', (510, 65), font, 1, (0, 0, 0), 2)
    canvas[:bar_height, 610:610 + border_thick, :] = 0
    cv2.putText(canvas, 'Perspective', (630, 65), font, 1, (0, 0, 0), 2)
    canvas[:bar_height, 820:820 + border_thick, :] = 0

    # Add rectangle option
    canvas[40:bar_height - 30, 850:1000] = list(next_color)
    canvas[:bar_height, 1030:1030 + border_thick, :] = 0

    # Save option
    pts = np.array([[1080, 15], [1135, 15], [1150, 30], [1150, 75], [1080,
                                                                     75]])
    cv2.fillPoly(canvas, pts=[pts], color=(255, 128, 0), lineType=cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                 (0, 0, 0), 2)
    canvas[17:35, 1100:1130] = 96
    canvas[20:30, 1120:1125] = (255, 128, 0)
    canvas[45:74, 1090:1140] = 96

    # X and Y axis
    for i in range(0, 1180, 20):
        cv2.line(canvas, (i, 500), (i + 10, 500), (0, 0, 0), 4)
    cv2.line(canvas, (1170, 480), (1190, 500), (0, 0, 0), 4)
    cv2.line(canvas, (1170, 520), (1190, 500), (0, 0, 0), 4)
    cv2.putText(canvas, 'x', (1170, 540), font, 1, (0, 0, 0), 2)

    for i in range(120, 900, 20):
        cv2.line(canvas, (600, i), (600, i + 10), (0, 0, 0), 4)
    cv2.line(canvas, (580, 140), (600, 110), (0, 0, 0), 4)
    cv2.line(canvas, (620, 140), (600, 110), (0, 0, 0), 4)
    cv2.putText(canvas, 'y', (630, 140), font, 1, (0, 0, 0), 2)
    return canvas


def get_current_function(x):
    if x < 200:
        return 'trans'
    elif x < 310:
        return 'rigid'
    elif x < 490:
        return 'sim'
    elif x < 610:
        return 'affine'
    elif x < 820:
        return 'proj'
    elif x < 1030:
        return 'add'
    else:
        return 'save'


class Square():
    def __init__(self, pt_ini, pt_end, color):
        self.pts = [
            pt_ini, [pt_end[0], pt_ini[1]], pt_end, [pt_ini[0], pt_end[1]]
        ]
        self.color = color
        self.click_position = []
        self.grabbed = False
        self.mesh = None
        self.grab_point = []
        self.transform_matrix = np.identity(3, dtype='float')
        self.transform = ''
        self.s = 1.0
        self.sx = 1.0
        self.sy = 1.0
        self.t = 0

    def update_matrix(self, cur_pos):
        c_curr_point = cur_pos.copy()
        c_curr_point[0] = c_curr_point[0] - center_point[0]
        c_curr_point[1] = -c_curr_point[1] + center_point[1]

        c_grab_point = self.grab_point.copy()
        c_grab_point[0] = c_grab_point[0] - center_point[0]
        c_grab_point[1] = -c_grab_point[1] + center_point[1]
        mean_pt = np.mean(self.pts, 0).astype('int')
        if self.transform == 'trans':

            self.transform_matrix[:2, 2] = np.array(c_curr_point) - mean_pt
        elif self.transform == 'rigid':
            theta_ini = 2 * np.pi if c_grab_point[0] == 0 else 2 * np.arctan(
                c_grab_point[1] / c_grab_point[0])
            theta_end = 2 * np.pi if c_curr_point[0] == 0 else 2 * np.arctan(
                c_curr_point[1] / c_curr_point[0])

            delta_theta = theta_end - theta_ini
            rot = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                            [np.sin(delta_theta),
                             np.cos(delta_theta)]],
                           dtype='float')
            self.transform_matrix[:2, :2] = rot * self.s
            self.transform_matrix[0, 0] *= self.sx
            self.transform_matrix[1, 1] *= self.sy
            self.transform_matrix[0, 1] += self.t

        elif self.transform == 'sim':
            new_s = np.exp(0.001 * np.linalg.norm(c_curr_point - c_grab_point))
            if np.linalg.norm(mean_pt -
                              c_curr_point) < np.linalg.norm(mean_pt -
                                                             c_grab_point):
                new_s = 1 / new_s
            self.transform_matrix[:2, :2] /= self.s
            self.transform_matrix[:2, :2] *= new_s
            self.s = new_s
        elif self.transform == 'affinex':
            new_sx = np.exp(0.001 *
                            np.linalg.norm(c_curr_point - c_grab_point))
            if np.linalg.norm(mean_pt -
                              c_curr_point) < np.linalg.norm(mean_pt -
                                                             c_grab_point):
                new_sx = 1 / new_sx
            self.transform_matrix[0, 0] /= self.sx
            self.transform_matrix[0, 0] *= new_sx
            self.sx = new_sx

        elif self.transform == 'affiney':
            new_sy = np.exp(0.001 *
                            np.linalg.norm(c_curr_point - c_grab_point))
            if np.linalg.norm(mean_pt -
                              c_curr_point) < np.linalg.norm(mean_pt -
                                                             c_grab_point):
                new_sy = 1 / new_sy
            self.transform_matrix[1, 1] /= self.sy
            self.transform_matrix[1, 1] *= new_sy
            self.sy = new_sy

        elif self.transform == 'affineskew':
            new_t = (c_curr_point[0] - c_grab_point[0]) / 100
            self.transform_matrix[0, 1] -= self.t
            self.transform_matrix[0, 1] += new_t
            self.t = new_t

        elif self.transform == 'projx':
            px = (c_curr_point[0] - c_grab_point[0]) / 10000
            self.transform_matrix[2, 0] = px

        elif self.transform == 'projy':
            py = (c_curr_point[1] - c_grab_point[1]) / 10000
            self.transform_matrix[2, 1] = py


def mark_x(img, x, y, line_delta, color, thickness):
    img = cv2.line(img, (x - line_delta, y - line_delta),
                   (x + line_delta, y + line_delta), color, thickness)
    img = cv2.line(img, (x - line_delta, y + line_delta),
                   (x + line_delta, y - line_delta), color, thickness)


def highlight(mouse_point, target_point, selection_threshold):
    if np.linalg.norm(target_point -
                      np.array(mouse_point)) > selection_threshold:
        return (127, 127, 127), 2
    else:
        return (0, 0, 0), 6


def transform_point(point, square):
    point = point + [1]
    point = square.transform_matrix @ point
    point = np.divide(point, point[2])
    point[0] = point[0] + center_point[0]
    point[1] = point[1] * -1 + center_point[1]
    return point[:2].astype('int')


def draw(square, display_img, state, cur_pos, selection_threshold):
    if square.grabbed:
        square.update_matrix(cur_pos)

    if square.mesh is not None:
        np_pts = np.array(square.pts)
        minc, maxc = np.amin(np_pts[:, 0]), np.amax(np_pts[:, 0])
        minr, maxr = np.amin(np_pts[:, 1]), np.amax(np_pts[:, 1])
        for i in range(minr, maxr):
            for j in range(minc, maxc):
                trans_pt = [j, i]
                trans_pt = transform_point(trans_pt, square)
                display_img[trans_pt[1],
                            trans_pt[0]] = square.mesh[maxr - i + minr,
                                                       j - minc]

    else:
        transformed_coords = []
        for pt in square.pts:
            trans_pt = transform_point(pt, square)
            transformed_coords.append(trans_pt)
        transformed_coords = np.array(transformed_coords, dtype='int')
        cv2.fillPoly(display_img,
                     pts=[transformed_coords],
                     color=square.color,
                     lineType=cv2.LINE_AA)

    if state not in possible_transforms:
        square.click_position = []
    else:
        # Translation grab point
        mean_pt = np.mean(square.pts, 0).astype('int').tolist()
        mean_pt = transform_point(mean_pt, square)
        square.click_position = [mean_pt]
        x_color, x_thickness = highlight(cur_pos, mean_pt, selection_threshold)
        mark_x(display_img, mean_pt[0], mean_pt[1], line_delta, x_color,
               x_thickness)
        if state == 'trans' or state == 'save':
            return

        # Rotation grab point
        up_right_corner = transform_point(square.pts[1], square)
        v0 = up_right_corner - mean_pt
        v0 = np.divide(v0, np.linalg.norm(v0))
        v0 *= 25
        v0 = v0.astype('int')
        draw_center = up_right_corner + v0
        square.click_position.append(draw_center)
        # Rotation arrow in display
        x_color, x_thickness = highlight(cur_pos, draw_center,
                                         selection_threshold)
        bot_right_corner = transform_point(square.pts[2], square)
        up_left_corner = transform_point(square.pts[0], square)
        bot_left_corner = transform_point(square.pts[3], square)
        v1 = bot_right_corner - up_right_corner
        v1 = np.divide(v1, np.linalg.norm(v1))
        v1 *= 25
        v1 = v1.astype('int')
        pt1 = tuple(draw_center.tolist())
        pt2 = tuple((draw_center + v1).tolist())
        cv2.line(display_img, pt1, pt2, x_color, x_thickness)
        v2 = up_left_corner - up_right_corner
        v2 = np.divide(v2, np.linalg.norm(v2))
        v2 *= 25
        v2 = v2.astype('int')
        pt3 = tuple((draw_center + v2).tolist())
        cv2.line(display_img, pt1, pt3, x_color, x_thickness)
        rot30 = np.array([[np.sqrt(3) / 2, -1 / 2], [1 / 2, np.sqrt(3) / 2]])
        v3 = rot30 @ v2.T
        v3 = v3.T
        v3 *= 0.75
        v3 = v3.astype('int')
        pt4 = tuple((draw_center + v2 - v3).tolist())
        cv2.line(display_img, pt3, pt4, x_color, x_thickness)
        v4 = rot30.T @ v2.T
        v4 = v4.T
        v4 *= 0.75
        v4 = v4.astype('int')
        pt5 = tuple((draw_center + v2 - v4).tolist())
        cv2.line(display_img, pt3, pt5, x_color, x_thickness)
        if state == 'rigid':
            return

        # Special case
        if state == 'sim':
            # Scaling arrow in display
            v0 = up_left_corner - mean_pt
            v0 = np.divide(v0, np.linalg.norm(v0))
            v0 *= 15
            v0 = v0.astype('int')
            draw_center = up_left_corner + 1.5 * v0
            draw_center = draw_center.astype("int")
            x_color, x_thickness = highlight(cur_pos, draw_center,
                                             selection_threshold)
            pt1 = tuple((draw_center - v0).tolist())
            pt2 = tuple((draw_center + v0).tolist())
            cv2.line(display_img, pt1, pt2, x_color, x_thickness)

            square.click_position.append(draw_center)
            v1 = rot30 @ v0.T
            v1 = v1.T
            v1 *= 0.75
            v1 = v1.astype('int')
            pt3 = tuple((draw_center + v0 - v1).tolist())
            cv2.line(display_img, pt2, pt3, x_color, x_thickness)
            v4 = rot30.T @ v0.T
            v4 = v4.T
            v4 *= 0.75
            v4 = v4.astype('int')
            pt4 = tuple((draw_center + v0 - v4).tolist())
            cv2.line(display_img, pt2, pt4, x_color, x_thickness)
            return

        square.click_position.append([-1000, -1000])  # Dummy point

        x_scale_pt = mean_pt.copy()
        x_scale_pt[0] -= 60
        square.click_position.append(x_scale_pt)
        x_color, x_thickness = highlight(cur_pos, x_scale_pt,
                                         selection_threshold)
        pt1 = (x_scale_pt[0] + 15, x_scale_pt[1])
        pt2 = (x_scale_pt[0] - 15, x_scale_pt[1])
        cv2.line(display_img, pt1, pt2, x_color, x_thickness)
        cv2.line(display_img, pt2, (pt2[0] + 15, pt2[1] - 10), x_color,
                 x_thickness)
        cv2.line(display_img, pt2, (pt2[0] + 15, pt2[1] + 10), x_color,
                 x_thickness)

        y_scale_pt = mean_pt.copy()
        y_scale_pt[1] += 60
        square.click_position.append(y_scale_pt)
        x_color, x_thickness = highlight(cur_pos, y_scale_pt,
                                         selection_threshold)
        pt1 = (y_scale_pt[0], y_scale_pt[1] + 15)
        pt2 = (y_scale_pt[0], y_scale_pt[1] - 15)
        cv2.line(display_img, pt1, pt2, x_color, x_thickness)
        cv2.line(display_img, pt1, (pt1[0] + 10, pt1[1] - 15), x_color,
                 x_thickness)
        cv2.line(display_img, pt1, (pt1[0] - 10, pt1[1] - 15), x_color,
                 x_thickness)

        mean_top_pt = np.mean(square.pts[:2], 0).astype('int').tolist()
        mean_top_pt = transform_point(mean_top_pt, square)
        mean_top_pt[1] -= 20
        square.click_position.append(mean_top_pt)
        x_color, x_thickness = highlight(cur_pos, mean_top_pt,
                                         selection_threshold)
        pt1 = mean_top_pt.copy()
        cv2.line(display_img, (pt1[0] - 40, pt1[1]), (pt1[0] + 40, pt1[1]),
                 x_color, x_thickness)

        if state.startswith('affine'):
            return

        # Projective X in display
        v0 = bot_left_corner - mean_pt
        v0 = np.divide(v0, np.linalg.norm(v0))
        v0 *= 15
        v0 = v0.astype('int')
        draw_center = bot_left_corner + 1.5 * v0
        draw_center = draw_center.astype("int")
        x_color, x_thickness = highlight(cur_pos, draw_center,
                                         selection_threshold)
        mark_x(display_img, draw_center[0], draw_center[1], line_delta,
               x_color, x_thickness)
        square.click_position.append(draw_center)

        # Projective Y in display
        v0 = bot_right_corner - mean_pt
        v0 = np.divide(v0, np.linalg.norm(v0))
        v0 *= 15
        v0 = v0.astype('int')
        draw_center = bot_right_corner + 1.5 * v0
        draw_center = draw_center.astype("int")
        x_color, x_thickness = highlight(cur_pos, draw_center,
                                         selection_threshold)
        mark_x(display_img, draw_center[0], draw_center[1], line_delta,
               x_color, x_thickness)
        cv2.circle(display_img, tuple(draw_center), 20, x_color, x_thickness)
        square.click_position.append(draw_center)


# mouse callback function
def interface(event, x, y, flags, param):
    global display_img, img, bar_height, color, line_delta, thickness, state, list_squares, pt_buffer, next_color, selection_threshold, grab_idx

    display_img = np.array(img)
    for square in list_squares:
        draw(square, display_img, state, [x, y], selection_threshold)
    if pt_buffer is not None:
        pt_end = tuple([x, y])
        cv2.rectangle(display_img, tuple(pt_buffer), pt_end, next_color, 4)
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < bar_height:
            state = get_current_function(x)
        else:
            if state == 'add':
                if pt_buffer is None:
                    pt_buffer = [x, y]
                else:
                    pt_buffer[0] = pt_buffer[0] - center_point[0]
                    pt_buffer[1] = -pt_buffer[1] + center_point[1]
                    pt_ini = center_point - np.array([x, y])
                    pt_ini[0] *= -1
                    list_squares.append(
                        Square(pt_buffer, pt_ini.tolist(), next_color))
                    next_color = tuple(
                        np.random.randint(0, 256, 3, dtype='uint8').tolist())
                    img[40:bar_height - 30, 850:1000] = list(next_color)
                    pt_buffer = None
                    state = 'idle'
            elif state == 'grabbing':
                state = 'idle'
                list_squares[grab_idx].grabbed = False
            else:  # Case where we want to grab part of a square
                for i in range(len(list_squares) - 1, -1, -1):
                    for j in range(len(list_squares[i].click_position)):
                        click_pos = list_squares[i].click_position[j]
                        if np.linalg.norm(click_pos - np.array([x, y])
                                          ) < selection_threshold:
                            if state == 'save':
                                my_img = cv2.imread(my_img_path)
                                new_shape = np.array(
                                    list_squares[i].pts[2]) - np.array(
                                        list_squares[i].pts[0])
                                new_shape = tuple(np.abs(new_shape.T).tolist())
                                my_img = cv2.resize(my_img, new_shape)
                                list_squares[i].mesh = my_img
                                state = 'idle'
                            else:
                                list_squares[
                                    i].transform = possible_transforms[j]
                                state = 'grabbing'
                                grab_idx = i
                                list_squares[i].grabbed = True
                                list_squares[i].grab_point = click_pos
                            break


if __name__ == '__main__':
    thickness = 2  # Thickness of the drawn x
    line_delta = 10  # Lenght of the line in the drawn x
    bar_height = 100  # Height of the menu bar
    selection_threshold = 15  # Distance in pixels to select a point

    my_img_path = './images/lena.jpeg'
    next_color = tuple(np.random.randint(0, 256, 3, dtype='uint8').tolist())
    state = 'idle'
    list_squares = []
    pt_buffer = None
    grab_idx = None

    cv2.namedWindow('canvas')
    img = init_canvas(next_color)

    display_img = np.array(img)
    cv2.setMouseCallback('canvas', interface)
    while (1):
        cv2.imshow('canvas', display_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('./saved.png', display_img)
    cv2.destroyAllWindows()