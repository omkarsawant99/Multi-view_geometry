import cv2
import numpy as np

def detect_aruco_tag(img, marker_size, total_markers):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners_aruco, ids, rejected = cv2.aruco.detectMarkers(grey, arucoDict, parameters = arucoParam)
    return corners_aruco, ids