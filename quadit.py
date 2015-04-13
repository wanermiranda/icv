#!/bin/python
MAXIMUM_SIZE = 600.00
norm_factor = 100.00
max_threads = 4
query_index = 6
rotation_factor = 204
angle_range = 360 / rotation_factor

__author__ = 'gorigan'
import glob
import threading
import time

import matplotlib.pyplot as plt

from rotate import *




dataset_query = '/home/gorigan/datasets/icv/tp1/imagens/query/'
dataset_target_sem_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/sem_ruido/'
dataset_target_com_ruido = '/home/gorigan/datasets/icv/tp1/imagens/target/com_ruido/'

query_list = ['001_apple_obj.png', '002_dumbphone_obj.png', '003_japan_obj.png', '004_yen_obj.png',
              '005_bottle_obj.png', '006_shoe_obj.png', '007_kay_obj.png', '008_starbucks_obj.png',
              '009_coca_obj.png']


class SlideWindow:

    def __init__(self, height, width, stride=5):
        self.height = height
        self.width = width
        self.stride = stride


class Difference:
    def __init__(self, histogram=-1.0, pixel=-1.0):
        self.clear = pixel == -1
        self._pixel = pixel 
        self._histogram = histogram * 2000

    def greater(self, target):

        if (self.get_value() > target.get_value()) or (self.clear):
            return 1
        else:
            return 0

    def get_value(self):
        return self._pixel  + self._histogram 

    def __str__(self):
        return "Difference (value= " + str(self.get_value()) + ", pixel= " + str(self._pixel) + ", histogram=" + str(
            self._histogram) + ")"


class WindowSlider(threading.Thread):
    def __init__(self, slide_window, thread_id, query_color, target_color, column, best, full=True):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.query_color = query_color
        self.query_hist = image_hist(query_color)
        self.target_color = target_color
        self.column = column
        self.best = best
        self.crop = None
        self.full = full
        self.slide_window = slide_window

    def run(self):
        y = self.column
        x = self.slide_window.width
        target_height, target_width = self.target_color.shape[:2]

        while x < target_width:
            calc_y = y - self.slide_window.height
            calc_x = x - self.slide_window.width
            crop_color = self.target_color[calc_y: y, calc_x: x]

            hist_diff = get_mse(image_hist(crop_color), self.query_hist)
            pixel_diff = get_mse(self.query_color, crop_color)

            diff = Difference(hist_diff, pixel_diff)

            if self.best.greater(diff):
                self.crop = crop_color
                self.best = diff
                print self.best

            x += self.slide_window.stride

            if x > target_width:
                x = target_width


def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def image_hist(img):
    colors_hist = np.zeros((3, 256))
    for i in range(0, 3):
        colors_hist[i, :256] = cv2.calcHist([img], [i], None, [256], [0, 256])[:256, 0]
        max_bin = colors_hist[i, :256].max()
        colors_hist[i, :256] /= max_bin
    return colors_hist


def get_mse(query, target):
    (m, n) = query.shape[:2]
    if (m == 3) and (n == 256):
        query = query[0:3, 0:256]
        target = target[0:3, 0:256]
    # (m, n) = query.shape[:2]
    # print str(m) + ", " + str(n)
    sums = np.power(np.array(query)-np.array(target), 2).sum()
    return sums / (m * n)


def get_size_factor(query, target, factor):
    (target_h, target_w) = target.shape[:2]
    (query_h, query_w) = query.shape[:2]
    if query_h > query_w:
        factor = (factor * target_h) / query_h
    else:
        factor = (factor * target_w) / query_w
    return factor


def get_base_scale(img, maximum):
    (query_h, query_w) = img.shape[:2]
    if query_h > query_w:
        factor = maximum / query_h
    else:
        factor = maximum / query_w
    return factor


def remove_noise(img): 
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)
    return dst

class Finder:
    norm_factor = 1

    def __init__(self):
        # used to reduce the amount of data treated

        rotate_diff = Difference()
        rotate_crop = None
        best_query = None
        best_angle = 0
        starting = time.time()

        for target_image_path in glob.glob(dataset_target_com_ruido + '00' + str(query_index) + '*.png'):
            print target_image_path
            target_color = cv2.imread(target_image_path)
            base_scale = get_base_scale(target_color, MAXIMUM_SIZE)
            target_color = cv2.resize(target_color, None, fx=base_scale, fy=base_scale)
            target_color = remove_noise(target_color)
            target_height, target_width = target_color.shape[:2]

            for angle in range(1,2):
                    # target_bw = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)
                    last_local_diff = rotate_diff
                    last_local_crop = rotate_crop

                    for factor_index in range(7, 1, -1):
                        # Readjusting the query size and the slide window to match X% of the targeted image
                        factor = 0.1 * factor_index

                        print "Factor: " + str(factor)
                        query_color = cv2.imread(dataset_query + query_list[query_index - 1])
                        query_color = rotate_image(query_color, angle * rotation_factor)

                        query_base_scale = get_size_factor(query_color, target_color, factor)

                        query_color = cv2.resize(query_color, None, fx=query_base_scale, fy=query_base_scale)
                        query_color= remove_noise(query_color)

                        query_height_w, query_width_w = query_color.shape[:2]

                        # step to slide
                        slide_window = SlideWindow(query_height_w, query_width_w)

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
                                l_thread = WindowSlider(slide_window, t_index, query_color, target_color,
                                                        y, local_diff)
                                l_thread.start()
                                threads.append(l_thread)
                                y += slide_window.stride
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
                        # else:
                            # break

                    if rotate_diff.greater(last_local_diff):
                        rotate_crop = last_local_crop
                        rotate_diff = last_local_diff
                        best_angle = angle * rotation_factor
                        print "INTERMEDIATE TIME: " + str(time.time() - starting) + " SECS - ANGLE: " + str(
                            angle * rotation_factor)
                        print "DIFF: " + str(rotate_diff)
                        # showImage(rotate_crop)
                        # showImage(best_query)

            print target_image_path
            print "TIME: " + str(time.time() - starting) + " SECS "
            print "DIFF: " + str(rotate_diff)
            print "ANGLE: " + str(best_angle)
            show_image(rotate_crop)
            print "Query "
            show_image(best_query)


if __name__ == "__main__":
    Finder()