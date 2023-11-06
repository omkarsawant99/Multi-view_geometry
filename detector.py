import cv2
import cv2.aruco as aruco
import numpy as np


markerSize = 6 
totalMarkers=250


def detect_aruco_tag(img, markerSize, totalMarkers, dictionary_type=aruco.DICT_6X6_250):
    arucoDict = aruco.getPredefinedDictionary(dictionary_type)
    arucoParams = aruco.DetectorParameters()
    (corners, ids, rejected) = aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    return corners, ids


def prepare_corner_points(corners, ids):
    centers = []
    for corner in corners:
        center = corner[0].mean(axis=0)
        centers.append(center)
    return np.array(centers, dtype=np.float32), ids


def process_image_for_aruco(img, markerSize, totalMarkers):
    corners, ids = detect_aruco_tag(img, markerSize, totalMarkers)
    if ids is not None:
        pts, ids = prepare_corner_points(corners, ids)
        return pts, ids
    else:
        return None, None

