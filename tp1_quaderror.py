#!/bin/python
import glob
import threading
import time

import matplotlib.pyplot as plt

from rotate import *


norm_factor = 100.00


def show_image(img):
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.show()


def image_hist(img):
    colors_hist = np.zeros((3, 256))
    for col in img:
        for cel in col:
            b_value = cel[0]
            g_value = cel[1]
            r_value = cel[2]
            colors_hist[0][b_value] += 1
            colors_hist[1][g_value] += 1
            colors_hist[2][r_value] += 1

    max_b = colors_hist[0].max()
    max_g = colors_hist[1].max()
    max_r = colors_hist[2].max()

    colors_hist[0] *= norm_factor / max_b
    colors_hist[1] *= norm_factor / max_g
    colors_hist[2] *= norm_factor / max_r
    return colors_hist


class SlideWindow:
    height = 0.0
    width = 0.0
    stride = 2

    def __init__(self, height, width, stride):
        self.height = height
        self.width = width
        self.stride = stride


class Difference:
    def __init__(self, histogram=-1.0, pixel=-1.0):
        self.pixel = pixel
        self.histogram = histogram
        self.value = (pixel * 0.3) + (histogram * 0.7)

    def greater(self, target):

        if (self.value > target.get_value()) or (self.pixel == -1):
            return 1
        else:
            return 0

    def get_value(self):
        return self.value

    def __str__(self):
        return "Difference (value= " + str(self.value) + ", pixel= " + str(self.pixel) + ", histogram=" + str(
            self.histogram) + ")"


class WindowSlider(threading.Thread):
    def __init__(self, slide_window, thread_id, query_bw, query_hist, target_color, target_bw, column, best):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.query_bw = query_bw
        self.query_hist = query_hist
        self.target_color = target_color
        self.target_bw = target_bw
        self.column = column
        self.best = best
        self.crop = None
        self.slide_window = slide_window

    def run(self):
        y = self.column
        x = self.slide_window.width
        target_height_bw, target_width_bw = self.target_bw.shape[:2]

        while x < target_width_bw:
            calc_y = y - self.slide_window.height
            calc_x = x - self.slide_window.width
            crop_bw = self.target_bw[calc_y: y, calc_x: x]
            crop_color = self.target_color[calc_y: y, calc_x: x]
            hist = image_hist(crop_color)

            hist_diff = dist_bin(self.query_hist, hist)
            diff = Difference(hist_diff, 0.0)

            pixel_diff = get_square_diff(self.query_bw, crop_bw)
            diff.pixel = pixel_diff

            if self.best.greater(diff):
                self.crop = crop_bw
                self.best = diff
                print self.best

            x += self.slide_window.stride

            if x > target_width_bw:
                x = target_width_bw


def dist_bin(query, target):
    diff = np.zeros((3, 256))
    sums = 0.0
    for row in range(256):
        diff[0][row] = (query[0][row] - target[0][row]) ** 2
        diff[1][row] = (query[1][row] - target[1][row]) ** 2
        diff[2][row] = (query[2][row] - target[2][row]) ** 2
        sums += diff[0][row] + diff[1][row] + diff[2][row]
    return sums / (256 * 3)


def get_square_diff(query, target):
    (m, n) = query.shape[:2]
    sums = np.power(np.array(query)-np.array(target), 2).sum()
    return sums / (m * n)


def remove_noise(img): 
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    return dst

class Finder:
    norm_factor = 1

    def __init__(self):
        # used to reduce the amount of data treated
        base_scale = 0.4
        # step to slide
        stride = 5
        max_threads = 4
        query_index = 2
        rotation_factor = 360
        angle_range = 360 / rotation_factor

        dataset_query = '/home/gorigan/datasets/icv/tp1/imagens/query/'
        dataset_target_sem_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/sem_ruido/'
        # dataset_target_com_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/com_ruido/'

        query_list = ['001_apple_obj.png', '002_dumbphone_obj.png', '003_japan_obj.png', '004_yen_obj.png',
                      '005_bottle_obj.png', '006_shoe_obj.png', '007_kay_obj.png', '008_starbucks_obj.png',
                      '009_coca_obj.png']
        rotate_diff = Difference()
        rotate_crop = None
        best_query = None
        best_angle = 0
        starting = time.time()
        for angle in range(angle_range):

            print dataset_target_sem_ruido + '00' + str(query_index) + '*.png'
            for target_image_path in glob.glob(dataset_target_sem_ruido + '00' + str(query_index) + '*.png'):

                target_color = cv2.imread(target_image_path)
                target_color = cv2.resize(target_color, None, fx=base_scale, fy=base_scale)
                target_height, target_width = target_color.shape[:2]
                target_bw = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)
                last_local_diff = rotate_diff
                last_local_crop = rotate_crop

                for factor_index in range(4, 0, -1):
                    # Readjusting the query size and the slide window to match X% of the targeted image
                    factor = 0.1 * factor_index
                    print "Factor: " + str(factor)
                    query_color = cv2.imread(dataset_query + query_list[query_index - 1])
                    query_bw = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
                    query_color = rotate_image(query_color, angle * rotation_factor)
                    query_height, query_width = query_color.shape[:2]
                    query_base_scale = (target_width * factor) / query_width

                    query_color = cv2.resize(query_color, None, fx=query_base_scale, fy=query_base_scale)
                    query_bw = cv2.resize(query_bw, None, fx=query_base_scale, fy=query_base_scale)

                    query_height_w, query_width_w = query_bw.shape[:2]

                    query_color_hist = image_hist(query_color)

                    slide_window = SlideWindow(query_height_w, query_width_w, stride)

                    print "Target Dimensions"
                    print target_height
                    print target_width
                    print "Angle " + str(angle * rotation_factor)
                    # showImage(target_color)

                    print "Slide Window Dimensions"
                    print slide_window.height
                    print slide_window.width
                    # showImage(query_color)

                    y = slide_window.height
                    local_diff = last_local_diff
                    local_crop = last_local_crop

                    while y < target_height:
                        threads = []
                        for t_index in range(max_threads):
                            l_thread = WindowSlider(slide_window, t_index, query_bw, query_color_hist, target_color,
                                                    target_bw, y, local_diff)
                            l_thread.start()
                            threads.append(l_thread)
                            y += stride
                            print "Line :" + str(y)
                            if y > target_height:
                                y = target_height

                        for t in threads:
                            t.join()

                        for t in threads:
                            if local_diff.greater(t.best):
                                if t.crop is not None:
                                    print "Thread: " + str(t.threadID) + " Value: " + str(t.best)
                                    local_diff = t.best
                                    local_crop = t.crop
                                    # show_image(local_crop)
                    if last_local_diff.greater(local_diff):
                        last_local_crop = local_crop
                        last_local_diff = local_diff
                        best_query = query_color
                        print "INTERMEDIATE TIME: " + str(time.time() - starting) + " SECS - FACTOR: " + str(factor) + \
                              " - ANGLE: " + str(angle * rotation_factor)
                        print "DIFF: " + str(local_diff)
                        # showImage(local_crop)
                        # showImage(best_query)
                    else:
                        break

                if rotate_diff.greater(last_local_diff):
                    rotate_crop = last_local_crop
                    rotate_diff = last_local_diff
                    best_angle = angle * rotation_factor
                    print "INTERMEDIATE TIME: " + str(time.time() - starting) + " SECS - ANGLE: " + str(
                        angle * rotation_factor)
                    print "DIFF: " + str(rotate_diff)
                    # showImage(rotate_crop)
                    # showImage(best_query)

        print "TIME: " + str(time.time() - starting) + " SECS "
        print "DIFF: " + str(rotate_diff)
        print "ANGLE: " + str(best_angle)
        show_image(rotate_crop)
        print "Query "
        show_image(best_query)


if __name__ == "__main__":
    Finder()