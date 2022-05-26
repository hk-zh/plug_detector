import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyrealsense2 as rs
import os
from PIL import Image, ImageDraw

# camera D435
CAMERA_INTRINSICS_MAT = np.array(
    [[615.9142456054688, 0.0, 327.5135498046875], [0.0, 616.4674682617188, 242.85317993164062], [0, 0, 1]], dtype=np.float32)
CAMERA_DISTORTION_COEFF_MAT = np.array([0, 0, 0, 0, 0], dtype=np.float32)
ARUCO_NAME = cv2.aruco.DICT_4X4_50
MARKER_SIDE_LENGTH_MM = 0.093
MARKER_SEPRATION = MARKER_SIDE_LENGTH_MM/4


class CameraHandler:
    """
    1. generate markers and print them out
    2. save image and poses list
    """
    def __init__(self, image_size, camera_intrinsics_mat, camera_distortion_coeff_mat):
        self.img_size = image_size
        self.cam_intr_mat = camera_intrinsics_mat
        self.cam_dist_coeff_mat = camera_distortion_coeff_mat
        self.marker_side_length_mm = MARKER_SIDE_LENGTH_MM
        self.marker_separation = MARKER_SEPRATION
        self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_NAME)
        self.detector_parameters = cv2.aruco.DetectorParameters_create()

    def draw_single_marker(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        img = cv2.aruco.drawMarker(self.aruco_dict, 0, 700)
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")

        plt.savefig("single_marker.pdf")
        plt.show()

    def draw_markers(self):
        fig = plt.figure()
        nx = 4
        ny = 3
        for i in range(1, nx * ny + 1):
            ax = fig.add_subplot(ny, nx, i)
            img = cv2.aruco.drawMarker(self.aruco_dict, i, 700)
            plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
            ax.axis("off")

        plt.savefig("markers.pdf")
        plt.show()

    def draw_marker_board(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        board = cv2.aruco.GridBoard_create(3, 3, self.marker_side_length_mm, self.marker_separation,
                                           self.aruco_dict)
        board_img = board.draw((640, 480))
        plt.imshow(board_img, cmap=mpl.cm.gray, interpolation="nearest")
        ax.axis("off")

        plt.savefig("board_new.pdf")
        plt.show()

    def detect_aruco_corners(self, image):
        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.detector_parameters)
        return marker_corners, marker_ids

    def save_camera_images(self, dir):
        # path for saving
        path = os.path.dirname(os.path.realpath(__file__))
        # /home/kejia/wiping/Cognitive-Wiping-Robot/Hand-Eye Calibration
        file_path = path + dir
        print("save images to", file_path)
        index = 1

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        # # initalize roi
        # roi = None

        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                # # select region of interest
                # if roi is None:
                #     roi = cv2.selectROI(windowName="roi", img=color_image, showCrosshair=True, fromCenter=False)
                #     x, y, w, h = roi
                #
                # cv2.rectangle(img=color_image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
                # color_image = color_image[y:y + h, x:x + w]

                # marker_corners, marker_ids = self.detect_aruco_corners(color_image)
                # marker_image = color_image.copy()
                # if marker_ids is not None:
                #     marker_image = cv2.aruco.drawDetectedMarkers(marker_image, marker_corners, marker_ids)
                #     # marker_image = cv2.aruco.drawAxis(marker_image, marker_corners, marker_ids)

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)


                k = cv2.waitKey(1)

                if k == ord('s'):
                    print("saving an image")
                    file_name = file_path + "calib_img_" + str(index) + ".png"
                    cv2.imwrite(file_name, color_image)
                    index = index + 1
                elif k == ord('c'):
                    break

                # option = input("enter the image name, e.g. a number")
                #
                # if option is not None:
                #     # /home/kejia/wiping/Cognitive-Wiping-Robot/Hand-Eye Calibration
                #     file_name = file_path + "calib_img_" + str(option) + ".png"
                #     cv2.imwrite(file_name, color_image)

        finally:
            # Stop streaming
            pipeline.stop()


if __name__ == '__main__':
    handler = CameraHandler((640, 480), CAMERA_INTRINSICS_MAT, CAMERA_DISTORTION_COEFF_MAT)
    # handler.draw_marker_board()

    # A minimum of 2 motions with non parallel rotation axes are necessary to determine the hand-eye transformation.
    # So at least 3 different poses are required, but it is strongly recommended to use many more poses.

    handler.save_camera_images(dir='/')

    # path = os.path.dirname(os.path.realpath(__file__))
    # file = path + "/test.txt"
    #
    # a = 0
    # test_list = []
    # while True:
    #     a = a + 1
    #     x = input()
    #     if x == "p":
    #         test_list.append(a)
    #     if x == "s":
    #         with open(file, 'w') as f:
    #             for item in test_list:
    #                 f.write("%s\n" % item)


    print("done")