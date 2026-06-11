import time
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QMessageBox
, QFileDialog, QApplication)
from PyQt5.QtGui import QPixmap
import joblib
# from model import nnNet_withclion
from sklearn.calibration import CalibratedClassifierCV
import nibabel as nib
import os
import vtkmodules.all as vtk
import sys
import pandas as pd
# from radiomics import featureextractor
import six
from win import Ui_MainWindow
import SimpleITK as itk 
from qimage2ndarray import array2qimage
import skimage.transform as st
import torch
from torch import overrides
# import torchvision
import sklearn
import pydicom
import tifffile
import dicom2nifti
import argparse
import pickle
import nnunet
import cv2
from cv2 import applyColorMap
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
import math
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免与PyQt冲突
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import io
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, accuracy_score, roc_curve
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOImage import vtkMetaImageReader, vtkNIFTIImageReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from radiomics import featureextractor
gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T1 = time.time()

def _check_opengl_available():
    """检测OpenGL是否可用（Mesa swrast驱动是否存在）"""
    import ctypes
    import ctypes.util
    try:
        gl_lib = ctypes.util.find_library('GL')
        if gl_lib is None:
            return False
        lib = ctypes.CDLL(gl_lib)
        return True
    except Exception:
        return False

_OPENGL_OK = _check_opengl_available()
if not _OPENGL_OK:
    print("[WARNING] OpenGL not available, 3D volume rendering will be disabled.")

class MyWindow1(QMainWindow, Ui_MainWindow):
    vtk_available = _OPENGL_OK  # 3D渲染是否可用

    def __init__(self):
        super(MyWindow1, self).__init__()
        self.setupUi(self)
        self.img = None
        self.showmask = None
        self.prinimg = None
        self.printlivermask = None
        self.mask = None
        self.clion = None
        self.clion_CT = None
        self.clion_radiomics = None
        self.space = None
        # self.net = nnNet_withclion(channel=1, numclass=1, numword=60, f=9, lian=2).to(device)
        # self.load_GPUS(self.net)
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.livermask = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.face_flage = 0
        # pixmap = QPixmap('stand3.png').scaled(80, 895)
        # self.index.setPixmap(pixmap)
        # with open('onehot.pkl', 'rb') as intp:
        # 3D-plus 11特征 LogisticRegression 管道模型
        with open('3dplus_model_pipeline.pickle', 'rb') as f:
            self.model = pickle.load(f)
        with open('3dplus_feature_names.pickle', 'rb') as f:
            self.feature_names = pickle.load(f)
        self.youden_threshold = 0.556
        # torch.cuda.empty_cache()
        self.view1.setMouseTracking(True)
        self.view2.setMouseTracking(True)
        self.view3.setMouseTracking(True)
        self.view1.installEventFilter(self)
        self.view2.installEventFilter(self)
        self.view3.installEventFilter(self)
        self.leng_img = -100
        self.width_img = -100
        self.high_img = -100

        # os.system('pip install -e FLARE22-main/. -i https://mirrors.aliyun.com/pypi/simple/')  # 已预安装，不需要每次启动安装
        os.environ['RESULTS_FOLDER'] = 'FLARE22-main/RESULTS_FOLDER'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'


        self.right_press_flag = False
        self.left_press_flag = False

        self.face_w = 500
        self.face_h = 420

        self.filename = ''
        self.volumes = {}
        self.volume_path = ''
        self.volume_old = None
        self.mask_path = None
        self.livermask_path = None
        self.scale_ratio = 1

        self.scene1.mouseDoubleClickEvent = self.pointselect1
        self.scene2.mouseDoubleClickEvent = self.pointselect2
        self.scene3.mouseDoubleClickEvent = self.pointselect3
        self.pen = QtGui.QPen(QtCore.Qt.green)
        self.pen2 = QtGui.QPen(QtCore.Qt.red, 4)
        self.pen3 = QtGui.QPen(QtCore.Qt.red)
        self.x_line1 = QtWidgets.QGraphicsLineItem()
        self.x_line2 = QtWidgets.QGraphicsLineItem()
        self.x_line1.setPen(self.pen)
        self.x_line2.setPen(self.pen)
        self.y_line1 = QtWidgets.QGraphicsLineItem()
        self.y_line2 = QtWidgets.QGraphicsLineItem()
        self.y_line1.setPen(self.pen)
        self.y_line2.setPen(self.pen)
        self.z_line1 = QtWidgets.QGraphicsLineItem()
        self.z_line2 = QtWidgets.QGraphicsLineItem()
        self.z_line1.setPen(self.pen)
        self.z_line2.setPen(self.pen)

        self.x_point1 = QtWidgets.QGraphicsEllipseItem()
        self.x_point2 = QtWidgets.QGraphicsEllipseItem()
        self.x_point1.setPen(self.pen2)
        self.x_point2.setPen(self.pen2)
        self.y_point1 = QtWidgets.QGraphicsEllipseItem()
        self.y_point2 = QtWidgets.QGraphicsEllipseItem()
        self.y_point1.setPen(self.pen2)
        self.y_point2.setPen(self.pen2)
        self.z_point1 = QtWidgets.QGraphicsEllipseItem()
        self.z_point2 = QtWidgets.QGraphicsEllipseItem()
        self.z_point1.setPen(self.pen2)
        self.z_point2.setPen(self.pen2)
        self.x_point_flag = 1
        self.y_point_flag = 1
        self.z_point_flag = 1
        self.x_point2line = QtWidgets.QGraphicsLineItem()
        self.x_point2line.setPen(self.pen3)
        self.y_point2line = QtWidgets.QGraphicsLineItem()
        self.y_point2line.setPen(self.pen3)
        self.z_point2line = QtWidgets.QGraphicsLineItem()
        self.z_point2line.setPen(self.pen3)

        self.x_x = None
        self.x_y = None
        self.y_x = None
        self.y_y = None
        self.z_x = None
        self.z_y = None

        self.factor1 = None
        self.factor2 = None
        self.factor3 = None
        self.factor4 = None
        self.factor5 = None
        self.factor6 = None
        self.factor7 = None
        self.factor8 = None
        self.factor9 = None
        self.factor10 = None
        self.factor11 = None
        self.factor12 = None
        self.factor13 = None
        self.factor14 = None
        self.factor15 = None
        # self.factor16 = None
        # self.factor17 = None
        # self.factor18 = None
        # self.factor19 = None
        # self.factor20 = None
        # self.factor21 = None
        # self.factor22 = None
        self.pixmapItem1 = None
        self.pixmapItem2 = None
        self.pixmapItem3 = None

    def pointselect1(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = event.scenePos().y()
            self.y_x = event.scenePos().x()
            self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.z_x = int(round(self.face_w * (event.scenePos().y() / self.face_h), 0))
            self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.width_img = int(round((event.scenePos().y() / self.face_h) * self.width_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.x_point_flag == 1:
                self.draw_point_x(self.x_point1, event.scenePos().x(), event.scenePos().y())
                self.point_x1_x = event.scenePos().x()
                self.point_x1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.x_point_flag = 2
            elif self.x_point_flag == 2:
                self.draw_point_x(self.x_point2, event.scenePos().x(), event.scenePos().y())
                self.point_x2_x = event.scenePos().x()
                self.point_x2_y = event.scenePos().y()
                self.drawline(self.scene1, self.x_point2line, self.point_x1_x, self.point_x1_y,
                              self.point_x2_x, self.point_x2_y)

                self.x_distance_x = abs(self.point_x1_x - self.point_x2_x) / 500 * self.high_max * self.space[0]
                self.x_distance_y = abs(self.point_x1_y - self.point_x2_y) / 420 * self.width_max * self.space[1]
                self.x_distance = math.sqrt(self.x_distance_y ** 2 + self.x_distance_x ** 2)
                self.distance.setText(f"{self.x_distance:>.8f}")
                self.x_point_flag = 1

    def pointselect2(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = event.scenePos().x()
            self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
            self.y_x = event.scenePos().x()
            self.y_y = event.scenePos().y()
            self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.high_img = int(round((event.scenePos().x() / self.face_w) * self.high_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.y_point_flag == 1:
                self.draw_point_y(self.y_point1, event.scenePos().x(), event.scenePos().y())
                self.point_y1_x = event.scenePos().x()
                self.point_y1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point1)
                self.scene3.removeItem(self.z_point2)
                self.y_point_flag = 2
            elif self.y_point_flag == 2:
                self.draw_point_y(self.y_point2, event.scenePos().x(), event.scenePos().y())
                self.point_y2_x = event.scenePos().x()
                self.point_y2_y = event.scenePos().y()
                self.drawline(self.scene2, self.y_point2line, self.point_y1_x, self.point_y1_y,
                              self.point_y2_x, self.point_y2_y)

                self.y_distance_x = abs(self.point_y1_x - self.point_y2_x) / 500 * self.high_max * self.space[0]
                self.y_distance_y = abs(self.point_y1_y - self.point_y2_y) / 420 * self.leng_max * self.space[2]
                self.y_distance = math.sqrt(self.y_distance_y ** 2 + self.y_distance_x ** 2)
                self.distance.setText(f"{self.y_distance:>.8f}")
                self.y_point_flag = 1

    def pointselect3(self, event):
        if self.prinimg is not None and event.button() == Qt.LeftButton:
            self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.x_y = int(round(self.face_h * (event.scenePos().x() / self.face_w), 0))
            self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.y_y = event.scenePos().y()
            self.z_x = event.scenePos().x()
            self.z_y = event.scenePos().y()
            self.leng_img = int(round((event.scenePos().y() / self.face_h) * self.leng_max, 0))
            self.width_img = int(round((event.scenePos().x() / self.face_w) * self.width_max, 0))
            self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)
        elif self.prinimg is not None and event.button() == Qt.RightButton:
            if self.z_point_flag == 1:
                self.draw_point_z(self.z_point1, event.scenePos().x(), event.scenePos().y())
                self.point_z1_x = event.scenePos().x()
                self.point_z1_y = event.scenePos().y()
                self.scene1.removeItem(self.x_point2line)
                self.scene2.removeItem(self.y_point2line)
                self.scene3.removeItem(self.z_point2line)
                self.scene1.removeItem(self.x_point1)
                self.scene1.removeItem(self.x_point2)
                self.scene2.removeItem(self.y_point1)
                self.scene2.removeItem(self.y_point2)
                self.scene3.removeItem(self.z_point2)
                self.z_point_flag = 2
            elif self.z_point_flag == 2:
                self.draw_point_z(self.z_point2, event.scenePos().x(), event.scenePos().y())
                self.point_z2_x = event.scenePos().x()
                self.point_z2_y = event.scenePos().y()
                self.drawline(self.scene3, self.z_point2line, self.point_z1_x, self.point_z1_y,
                              self.point_z2_x, self.point_z2_y)
                self.z_distance_x = abs(self.point_z1_x - self.point_z2_x) / 500 * self.width_max * self.space[1]
                self.z_distance_y = abs(self.point_z1_y - self.point_z2_y) / 420 * self.leng_max * self.space[2]
                self.z_distance = math.sqrt(self.z_distance_y ** 2 + self.z_distance_x ** 2)
                self.distance.setText(f"{self.z_distance:>.8f}")
                self.z_point_flag = 1

    def draw_point_x(self, item, x, y):
        self.scene1.removeItem(item)
        self.scene1.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene1.addItem(item)
        self.scene1.addItem(item)

    def drawline(self, scene, item, x1, y1, x2, y2):
        item.setLine(QtCore.QLineF(QtCore.QPointF(x1, y1),
                                           QtCore.QPointF(x2, y2)))
        scene.addItem(item)

    def draw_point_y(self, item, x, y):
        self.scene2.removeItem(item)
        self.scene2.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene2.addItem(item)
        self.scene2.addItem(item)

    def draw_point_z(self, item, x, y):
        self.scene3.removeItem(item)
        self.scene3.removeItem(item)
        item.setRect(x-2, y-2, 4, 4)
        self.scene3.addItem(item)
        self.scene3.addItem(item)

    def draw_line_x(self, x, y):
        self.scene1.removeItem(self.x_line1)
        self.scene1.removeItem(self.x_line2)
        self.x_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene1.height())))
        self.x_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene1.width(), int(y))))
        self.scene1.addItem(self.x_line1)
        self.scene1.addItem(self.x_line2)

    def draw_line_y(self, x, y):
        self.scene2.removeItem(self.y_line1)
        self.scene2.removeItem(self.y_line2)
        self.y_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene2.height())))
        self.y_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene2.width(), int(y))))
        self.scene2.addItem(self.y_line1)
        self.scene2.addItem(self.y_line2)

    def draw_line_z(self, x, y):
        self.scene3.removeItem(self.z_line1)
        self.scene3.removeItem(self.z_line2)
        self.z_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x), 0),
                                           QtCore.QPointF(int(x), self.scene3.height())))
        self.z_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y)),
                                           QtCore.QPointF(self.scene3.width(), int(y))))
        self.scene3.addItem(self.z_line1)
        self.scene3.addItem(self.z_line2)

    def cleardistancef(self):
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)
        self.distance.clear()

    def clearfactorf(self):
        self.box1.clear()
        self.box2.clear()
        self.box3.clear()
        self.box4.clear()
        self.box5.clear()
        self.box6.clear()
        self.box7.clear()
        self.box8.clear()
        self.box9.clear()
        self.box10.clear()
        self.box11.clear()
        self.box12.clear()
        self.box13.clear()
        self.box14.clear()
        self.box15.clear()
        # self.box16.clear()
        # self.box17.clear()
        # self.box18.clear()
        # self.box19.clear()
        # self.box20.clear()
        # self.box21.clear()
        # self.box22.clear()
        self.clion = None
        self.clion_CT = None
        self.clion_radiomics = None


    def clearallf(self):
        # self.img = None
        # self.prinimg = None
        self.mask = None
        self.livermask = None
        self.printlivermask = None
        self.clion = None
        self.clion_CT = None
        self.clion_radiomics = None
        self.space = None
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.heatmaprongqi = None
        self.resultprint = None
        self.flag = False
        self.plotresult.clear()
        # self.plotcamminmax3d.clear()
        # self.page.clear()
        # self.page2.clear()
        # self.page3.clear()
        # if self.factor1 is not None and self.factor2 is not None and self.factor2 is not None \
        #         and self.factor3 is not None and self.factor4 is not None and self.factor5 is not None \
        #         and self.factor6 is not None and self.factor7 is not None and self.factor8 is not None \
        #         and self.factor9 is not None and self.factor10 is not None and self.factor11 is not None \
        #         and self.factor12 is not None and self.factor13 is not None and self.factor14 is not None \
        #         and self.factor15 is not None and self.factor16 is not None and self.factor17 is not None \
        #         and self.factor18 is not None and self.factor19 is not None and self.factor20 is not None \
        #         and self.factor21 is not None and self.factor22 is not None:
        self.box1.clear()
        self.box2.clear()
        self.box3.clear()
        self.box4.clear()
        self.box5.clear()
        self.box6.clear()
        self.box7.clear()
        self.box8.clear()
        self.box9.clear()
        self.box10.clear()
        self.box11.clear()
        self.box12.clear()
        self.box13.clear()
        self.box14.clear()
        self.box15.clear()
        # self.box16.clear()
        # self.box17.clear()
        # self.box18.clear()
        # self.box19.clear()
        # self.box20.clear()
        # self.box21.clear()
        # self.box22.clear()
        # self.scene1.clear()
        # self.scene2.clear()
        # self.scene3.clear()
        # self.scene1.removeItem(self.x_point2line)
        # self.scene2.removeItem(self.y_point2line)
        # self.scene3.removeItem(self.z_point2line)
        # self.scene1.removeItem(self.x_point1)
        # self.scene1.removeItem(self.x_point2)
        # self.scene2.removeItem(self.y_point1)
        # self.scene2.removeItem(self.y_point2)
        # self.scene3.removeItem(self.z_point1)
        # self.scene3.removeItem(self.z_point2)
        # self.scene1.removeItem(self.pixmapItem1)
        # self.scene2.removeItem(self.pixmapItem2)
        # self.scene3.removeItem(self.pixmapItem3)
        # self.scene1.removeItem(self.x_line1)
        # self.scene1.removeItem(self.x_line2)
        # self.scene2.removeItem(self.y_line1)
        # self.scene2.removeItem(self.y_line2)
        # self.scene3.removeItem(self.z_line1)
        # self.scene3.removeItem(self.z_line2)
        # self.distance.clear()
        # self.ren.RemoveAllViewProps()
        # camera = self.ren.GetActiveCamera()
        # self.iren.Initialize()


    def __setDragEnabled(self, isEnabled: bool):
        """ 设置拖拽是否启动 """
        self.view1.setDragMode(self.view1.ScrollHandDrag if isEnabled else self.view1.NoDrag)
        self.view2.setDragMode(self.view2.ScrollHandDrag if isEnabled else self.view2.NoDrag)
        self.view3.setDragMode(self.view3.ScrollHandDrag if isEnabled else self.view3.NoDrag)

    def __isEnableDrag(self, pixmap):
        """ 根据图片的尺寸决定是否启动拖拽功能 """
        if self.prinimg is not None:
            v = pixmap.width() > 500
            h = pixmap.height() > 420
            return v or h

    def showpic_xyz(self, x, y, z, w_size, h_size):
        if self.prinimg is not None:
            if self.prinimg.ndim == 3:
                image_axi = array2qimage(np.expand_dims(self.prinimg[x, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_axi = array2qimage(self.prinimg[x, ...])

            pixmap_axi = QPixmap.fromImage(image_axi).scaled(w_size, h_size)
            self.pixmapItem1 = QtWidgets.QGraphicsPixmapItem(pixmap_axi)
            self.scene1.addItem(self.pixmapItem1)
            self.view1.setSceneRect(QtCore.QRectF(pixmap_axi.rect()))
            self.view1.setScene(self.scene1)
            self.page.setText(str(self.leng_img + 1) + '/' + str(int(self.leng_max)))

            if self.prinimg.ndim == 3:
                image_cor = array2qimage(np.expand_dims(self.prinimg[:, y, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_cor = array2qimage(self.prinimg[:, y, ...])
            pixmap_cor = QPixmap.fromImage(image_cor).scaled(w_size, h_size)
            self.pixmapItem2 = QtWidgets.QGraphicsPixmapItem(pixmap_cor)
            self.scene2.addItem(self.pixmapItem2)
            self.view2.setSceneRect(QtCore.QRectF(pixmap_cor.rect()))
            self.view2.setScene(self.scene2)
            self.page2.setText(str(self.width_img + 1) + '/' + str(int(self.width_max)))

            if self.prinimg.ndim == 3:
                image_sag = array2qimage(np.expand_dims(self.prinimg[:, :, z, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_sag = array2qimage(self.prinimg[:, :, z, ...])
            pixmap_sag = QPixmap.fromImage(image_sag).scaled(w_size, h_size)
            self.pixmapItem3 = QtWidgets.QGraphicsPixmapItem(pixmap_sag)
            self.scene3.addItem(self.pixmapItem3)
            self.view3.setSceneRect(QtCore.QRectF(pixmap_sag.rect()))
            self.view3.setScene(self.scene3)
            self.page3.setText(str(self.high_img + 1) + '/' + str(int(self.high_max)))

            self.__setDragEnabled(self.__isEnableDrag(pixmap_axi))

    def showpic(self):
        # fname = QFileDialog.getOpenFileName(self, '加载图片', 'C:\\')
        fname = QFileDialog.getOpenFileName(self, caption='Load CT image',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        self.filename = fname[0]
        self.parent_file = os.path.abspath(os.path.join(self.filename, ".."))
        print(self.filename)
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            self.space = img.GetSpacing()
            # img = itk.GetArrayFromImage(img)
            self.img = itk.GetArrayFromImage(img)
            img = np.clip(self.img, -17.0, 201.0)
            img = np.flip(img, axis=0)
            self.prinimg = (img - 99.40078) / 39.392952
            self.ori_prinimg = self.prinimg
            self.leng_max, self.width_max, self.high_max = self.img.shape
            self.face_w = 500
            self.face_h = 420
            # self.leng_scale, self.width_scale, self.high_scale = self.img.shape
            # if self.width_scale >= 500 or self.high_scale >= 420:
            #     if self.high_scale >= 420:
            #         self.scale_ratio = 420 / self.high_scale
            #         self.high_scale = 420
            #         self.width_scale = int(self.width_scale * self.scale_ratio)
            #     if self.width_scale >= 500:
            #         self.scale_ratio = 500 / self.width_scale
            #         self.width_scale = 500
            #         self.high_scale = int(self.high_scale * self.scale_ratio)
            # if self.width_scale < 500 or self.high_scale < 420:
            #     if self.high_scale < 420:
            #         self.scale_ratio = 420 / self.high_scale
            #         self.high_scale = 420
            #         self.width_scale = int(self.width_scale * self.scale_ratio)
            #     if self.width_scale < 500:
            #         self.scale_ratio = 500 / self.width_scale
            #         self.width_scale = 500
            #         self.high_scale = int(self.high_scale * self.scale_ratio)
            self.leng_img = int(self.leng_max / 2)
            self.width_img = int(self.width_max / 2)
            self.high_img = int(self.high_max / 2)
            self.showpic_xyz(int(self.leng_max / 2), int(self.width_max / 2), int(self.high_max / 2), self.face_w, self.face_h)
            self.x_x = self.face_w // 2
            self.x_y = self.face_h // 2
            self.y_x = self.face_w // 2
            self.y_y = self.face_h // 2
            self.z_x = self.face_w // 2
            self.z_y = self.face_h // 2

            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)

            self.coord.setText(f"Path: {self.filename}")

            reader = vtkNIFTIImageReader()
            reader.SetFileName(self.filename)
            reader.Update()

            volumeMapper = vtkGPUVolumeRayCastMapper()
            volumeMapper.SetInputData(reader.GetOutput())

            volumeProperty = vtkVolumeProperty()
            volumeProperty.SetInterpolationTypeToLinear()
            volumeProperty.ShadeOn()
            volumeProperty.SetAmbient(0.4)
            volumeProperty.SetDiffuse(0.6)
            volumeProperty.SetSpecular(0.2)
            compositeOpacity = vtkPiecewiseFunction()
            compositeOpacity.AddPoint(70, 0.00)
            compositeOpacity.AddPoint(90, 0.4)
            compositeOpacity.AddPoint(180, 0.6)
            volumeProperty.SetScalarOpacity(compositeOpacity)
            volumeGradientOpacity = vtkPiecewiseFunction()
            volumeGradientOpacity.AddPoint(10, 0.0)
            volumeGradientOpacity.AddPoint(90, 0.5)
            volumeGradientOpacity.AddPoint(100, 1.0)
            color = vtkColorTransferFunction()
            color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
            color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
            color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
            volumeProperty.SetColor(color)

            volume = vtkVolume()
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProperty)

            if self.volume_old is not None:
                self.ren.RemoveViewProp(self.volume_old)
            self.ren.AddViewProp(volume)
            self.volume_old = volume
            camera = self.ren.GetActiveCamera()
            c = volume.GetCenter()
            camera.SetViewUp(0, 0, 1)
            camera.SetPosition(c[0], c[1] - 800, c[2]-200)
            camera.SetFocalPoint(c[0], c[1], c[2])
            camera.Azimuth(30.0)
            camera.Elevation(30.0)
            self.iren.Initialize()


    def radiomics_link(self):
        self.statusbar.showMessage("Calculate the features of GLCM/GLRLM")
        self.show_message_radiomics()
        if self.livermask is None and self.mask is None:
            if os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == False and self.livermask is None:
                self.get_livermask()
            elif os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == True:
                livermask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_liver.nii.gz')
                self.livermask = itk.GetArrayFromImage(livermask)
                self.livermask = np.where(self.livermask != 0, 1, 0)
            if os.path.exists(self.filename.split('.nii.gz')[0] + '_spleen.nii.gz') == False and self.mask is None:
                self.get_tumormask()
            elif os.path.exists(self.filename.split('.nii.gz')[0] + '_spleen.nii.gz') == True:
                tumormask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_spleen.nii.gz')
                self.mask = itk.GetArrayFromImage(tumormask)
                self.mask = np.where(self.mask != 0, 1, 0)

        img = itk.ReadImage(self.filename)
        image_space = img.GetSpacing()
        image_direction = img.GetDirection()
        image_origin = img.GetOrigin()

        position_1 = np.where(self.livermask!=0)
        position_2 = np.where(self.mask!=0)
        new_img = np.zeros_like(itk.GetArrayFromImage(img))
        new_img[position_1] = 1
        new_img[position_2] = 1
        src_new = itk.GetImageFromArray(new_img)
        src_new.SetSpacing(image_space)
        src_new.SetOrigin(image_origin)
        src_new.SetDirection(image_direction)
        itk.WriteImage(src_new, self.filename.split('.nii.gz')[0] + '_cat.nii.gz')
        save_curdata, name = self.catch_features(self.filename, self.filename.split('.nii.gz')[0] + '_cat.nii.gz')
        pos = np.where((name == 'wavelet-LHH_glrlm_RunVariance') | (name == 'wavelet-HLH_glcm_MaximumProbability')
                       |(name=='wavelet-LmeiHH_glrlm_LongRunEmphasis') | (name=='wavelet-LHL_glcm_DifferenceEntropy'))
        out = []
        for kk in pos[0]:
            out.append(save_curdata[kk])
        self.clion_radiomics=out
        s = 0
        self.statusbar.showMessage("The features of GLCM/GLRLM have been extracted")
        # os.remove(self.filename.split('.nii.gz')[0] + '_cat.nii.gz')


    def catch_features(self, imagePath, maskPath):
        if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
            raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
        settings = {}
        settings['binWidth'] = 25  # 5
        settings['sigma'] = [1, 3, 5]
        settings['Interpolator'] = itk.sitkBSpline
        settings['resampledPixelSpacing'] = [0.7, 0.7, 5]  # 3,3,3
        settings['voxelArrayShift'] = 1000  # 300
        settings['normalize'] = True
        settings['normalizeScale'] = 100
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        #extractor = featureextractor.RadiomicsFeatureExtractor()
        # print('Extraction parameters:\n\t', extractor.settings)
    
        # # extractor.enableImageTypeByName('LoG')
        # extractor.enableImageTypeByName('Wavelet')
        # # extractor.enableImageTypeByName('Gradient')
        # # extractor.enableAllFeatures()
        # extractor.enableFeaturesByName(firstorder=['Entropy', 'Variance'])
        # # extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio',
        # #                                       'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion',
        # #                                       'Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn',
        # #                                       'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
        # #                                       'LeastAxisLength', 'Elongation', 'Flatness'])
    
        extractor.enableImageTypeByName('LoG')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Gradient')
        extractor.enableAllFeatures()
        extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile',
                                                   '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange',
                                                   'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
                                                   'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis',
                                                   'Variance', 'Uniformity'])
        extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio',
                                              'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion',
                                              'Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn',
                                              'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
                                              'LeastAxisLength', 'Elongation', 'Flatness'])
    
    
        feature_cur = []
        feature_name = []
        result = extractor.execute(imagePath, maskPath, label=1)
        for key, value in six.iteritems(result):
            # print('\t', key, ':', value)
            feature_name.append(key)
            feature_cur.append(value)
        # print(len(feature_cur[37:]))
        name = feature_name[37:]
        name = np.array(name)
        '''
        flag=1
        if flag:
            name = np.array(feature_name)
            name_df = pd.DataFrame(name)
            writer = pd.ExcelWriter('key.xlsx')
            name_df.to_excel(writer)
            writer.save()
            flag = 0
        '''
        for i in range(len(feature_cur[37:])):
            #if type(feature_cur[i+22]) != type(feature_cur[30]):
            feature_cur[i+37] = float(feature_cur[i+37])
        return feature_cur[37:], name
    
    def cal_liver(self):
        self.statusbar.showMessage("Calculate the volume of liver")
        self.show_message_compute_volume()
        if os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == False and self.livermask is None:
            self.get_livermask()
        elif os.path.exists(self.filename.split('.nii.gz')[0] + '_liver.nii.gz') == True:
            livermask = itk.ReadImage(self.filename.split('.nii.gz')[0] + '_liver.nii.gz')
            self.livermask = itk.GetArrayFromImage(livermask)
            self.livermask = np.where(self.livermask != 0, 1, 0)
        space = itk.ReadImage(self.filename).GetSpacing()
        voxel_volume = space[0] * space[1] * space[2]

        voxel_sum = np.where(self.livermask != 0)
        volume = voxel_sum[0].shape[0] * voxel_volume
        self.volume_lv.setText(f"{volume:>.2f}")



    def cal_spleen(self):
        self.statusbar.showMessage("Calculate the volume of spleen")
        if self.img is None:
            self.statusbar.showMessage("Please load an image first")
            return
        if self.mask is None:
            self.statusbar.showMessage("Please load or generate spleen mask first")
            return
        
        space = itk.ReadImage(self.filename).GetSpacing()
        voxel_volume = space[0] * space[1] * space[2]

        voxel_sum = np.where(self.mask != 0)
        volume = voxel_sum[0].shape[0] * voxel_volume
        self.volume_sp.setText(f"{volume:>.2f}")
        self.statusbar.showMessage(f"Spleen volume: {volume:.2f}")

    def slideroriginal_function(self):

        if self.prinimg is not None:
            self.slideroriginal.setMinimum(10)
            self.slideroriginal.setMaximum(40)
            # self.slideroriginal.setSliderPosition(self.img.shape[0] // 2)
            self.slideroriginal.setTickInterval(1)
            self.numprint = self.slideroriginal.value()
            self.scale_ratio = self.numprint/10 - self.scale_ratio
            # ori_face_w = self.face_w
            # ori_face_h = self.face_h
            old_face_w = self.face_w
            old_face_h = self.face_h

            self.face_w = int(500 * (self.numprint/10))
            self.face_h = int(420 * (self.numprint/10))
            self.showpic_xyz(int(self.leng_img), int(self.width_img), int(self.high_img), self.face_w, self.face_h)

            self.x_x = self.x_x / old_face_w * self.face_w
            self.x_y = self.x_y / old_face_h * self.face_h
            self.y_x = self.y_x / old_face_w * self.face_w
            self.y_y = self.y_y / old_face_h * self.face_h
            self.z_x = self.z_x / old_face_w * self.face_w
            self.z_y = self.z_y / old_face_h * self.face_h

            self.draw_line_x(self.x_x, self.x_y)
            self.draw_line_y(self.y_x, self.y_y)
            self.draw_line_z(self.z_x, self.z_y)

    def show_mask(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load spleen mask', directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0 and self.img is not None:
            img = itk.ReadImage(fname[0])
            self.showmask = itk.GetArrayFromImage(img)
            if self.showmask.shape == self.img.shape:
                self.affine = nib.load(self.filename).affine
                self.mask_np = nib.load(fname[0]).get_fdata()
                self.img_np = nib.load(self.filename).get_fdata()
                self.new_img = self.mask_np * self.img_np
                self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
                nib.save(self.new_img, os.path.join(self.parent_file, 'new_img.nii.gz'))
                if self.prinimg is not None:
                    self.printmask(0.5)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)

                    reader = vtkNIFTIImageReader()
                    reader.SetFileName(os.path.join(self.parent_file, 'new_img.nii.gz'))
                    reader.Update()
                    volumeMapper = vtkGPUVolumeRayCastMapper()
                    volumeMapper.SetInputData(reader.GetOutput())
                    volumeProperty = vtkVolumeProperty()
                    volumeProperty.SetInterpolationTypeToLinear()
                    volumeProperty.ShadeOn()
                    volumeProperty.SetAmbient(0.4)
                    volumeProperty.SetDiffuse(0.6)
                    volumeProperty.SetSpecular(0.2)
                    compositeOpacity = vtkPiecewiseFunction()
                    compositeOpacity.AddPoint(70, 0.00)
                    compositeOpacity.AddPoint(90, 0.4)
                    compositeOpacity.AddPoint(180, 0.6)
                    volumeProperty.SetScalarOpacity(compositeOpacity)
                    volumeGradientOpacity = vtkPiecewiseFunction()
                    volumeGradientOpacity.AddPoint(10, 0.0)
                    volumeGradientOpacity.AddPoint(90, 0.5)
                    volumeGradientOpacity.AddPoint(100, 1.0)
                    color = vtkColorTransferFunction()
                    color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                    color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                    color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                    volumeProperty.SetColor(color)
                    volume = vtkVolume()
                    volume.SetMapper(volumeMapper)
                    volume.SetProperty(volumeProperty)
                    if self.volume_old is not None:
                        self.ren.RemoveViewProp(self.volume_old)
                    self.ren.AddViewProp(volume)
                    self.volume_old = volume
                    camera = self.ren.GetActiveCamera()
                    c = volume.GetCenter()
                    camera.SetViewUp(0, 0, 1)
                    camera.SetPosition(c[0], c[1] - 800, c[2]-200)
                    camera.SetFocalPoint(c[0], c[1], c[2])
                    camera.Azimuth(30.0)
                    camera.Elevation(30.0)
                    self.iren.Initialize()
                    os.remove(os.path.join(self.parent_file, 'new_img.nii.gz'))
                    self.statusbar.showMessage("The segmentation result has been displayed")
            else:
                self.statusbar.showMessage("Please load a correspronding segmentation result")
        elif self.img is None:
            self.statusbar.showMessage("Please load a image first")

    # def rbclicked(self):
    #     sender = self.sender()
    #     if sender == self.bg1:
    #         if self.bg1.checkedId() == 11:
    #             self.factor1 = 0
    #         elif self.bg1.checkedId() == 12:
    #             self.factor1 = 1

    def clinicf_read(self):
        # 读取11个特征值
        # box1=Age, box2=Sodium, box3=APRI, box4=CP Score, box5=Creatinine Conversion
        # box6=Splenic Vein, box7=PV Diameter, box8=SFVS CT Change Rate
        # box9=Median Angle SFVS, box10=GRS Diameter, box11=Paraumbilical Vein
        boxes = [self.box1, self.box2, self.box3, self.box4, self.box5,
                 self.box6, self.box7, self.box8, self.box9, self.box10, self.box11]
        values = [b.text() for b in boxes]

        if all(self.is_number(v) for v in values):
            self.show_message_clinicf()
            self.statusbar.showMessage("Start to load factors.")

            # 构建11特征向量 (1, 11)
            feature_array = np.array([[float(v) for v in values]])
            self.clion = feature_array
            # 标记所有数据已就绪（不再需要CT和radiomics分开处理）
            self.clion_CT = True
            self.clion_radiomics = True
            self.statusbar.showMessage("All 11 factors have been loaded.")
        else:
            self.statusbar.showMessage("Please input all 11 features as required.")


    def showmask_path(self, path):
        img = itk.ReadImage(path)
        self.showmask = itk.GetArrayFromImage(img)
        if self.showmask.shape == self.img.shape:
            self.affine = nib.load(self.filename).affine
            self.mask_np = nib.load(path).get_fdata()
            self.img_np = nib.load(self.filename).get_fdata()
            self.new_img = self.mask_np * self.img_np
            self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
            nib.save(self.new_img, os.path.join(self.parent_file, 'new_img.nii.gz'))
            if self.prinimg is not None:
                self.printmask(0.5)
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

                reader = vtkNIFTIImageReader()
                reader.SetFileName(os.path.join(self.parent_file, 'new_img.nii.gz'))
                reader.Update()
                volumeMapper = vtkGPUVolumeRayCastMapper()
                volumeMapper.SetInputData(reader.GetOutput())
                volumeProperty = vtkVolumeProperty()
                volumeProperty.SetInterpolationTypeToLinear()
                volumeProperty.ShadeOn()
                volumeProperty.SetAmbient(0.4)
                volumeProperty.SetDiffuse(0.6)
                volumeProperty.SetSpecular(0.2)
                compositeOpacity = vtkPiecewiseFunction()
                compositeOpacity.AddPoint(70, 0.00)
                compositeOpacity.AddPoint(90, 0.4)
                compositeOpacity.AddPoint(180, 0.6)
                volumeProperty.SetScalarOpacity(compositeOpacity)
                volumeGradientOpacity = vtkPiecewiseFunction()
                volumeGradientOpacity.AddPoint(10, 0.0)
                volumeGradientOpacity.AddPoint(90, 0.5)
                volumeGradientOpacity.AddPoint(100, 1.0)
                color = vtkColorTransferFunction()
                color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                volumeProperty.SetColor(color)
                volume = vtkVolume()
                volume.SetMapper(volumeMapper)
                volume.SetProperty(volumeProperty)
                if self.volume_old is not None:
                    self.ren.RemoveViewProp(self.volume_old)
                self.ren.AddViewProp(volume)
                self.volume_old = volume
                camera = self.ren.GetActiveCamera()
                c = volume.GetCenter()
                camera.SetViewUp(0, 0, 1)
                camera.SetPosition(c[0], c[1] - 800, c[2]-200)
                camera.SetFocalPoint(c[0], c[1], c[2])
                camera.Azimuth(30.0)
                camera.Elevation(30.0)
                self.iren.Initialize()
                os.remove(os.path.join(self.parent_file, 'new_img.nii.gz'))

    def loadmaskf(self):
        # fname = QFileDialog.getOpenFileName(self, '加载图片', 'C:\\')
        fname = QFileDialog.getOpenFileName(self, caption='Load spleen mask',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            self.tumor_path = fname[0]
            mask = itk.GetArrayFromImage(img)
            if self.img is not None and mask.shape == self.img.shape:
                self.mask = itk.GetArrayFromImage(img)
                self.mask = np.where(self.mask != 0, 1, 0)
                self.showmask = self.mask.copy()
                if self.prinimg is not None:
                    self.printmask(0.5)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)
                self.statusbar.showMessage("The segmentation result of spleen has been loaded")
            else:
                self.statusbar.showMessage("Please load a corresponding segmentation result")

    def printmask(self, alpha):
        new_prinimg = []
        for i in range(self.leng_max):
            # camone = np.ones((self.width_max, self.high_max))
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max-1) - i, ...]
            imgone = self.normilize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            # heatmap = applyColorMap(np.uint8(255 * camone), cv2.COLORMAP_JET)
            heatmap = np.ones((self.width_max, self.high_max, 3))
            heatmap[:, :, 0] = 255
            heatmap[:, :, 1] = 0
            heatmap[:, :, 2] = 0
            cam_img = alpha * heatmap * maskone_ + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg

    def loadlivermaskf(self):
        # fname = QFileDialog.getOpenFileName(self, '加载图片', 'C:\\')
        fname = QFileDialog.getOpenFileName(self, caption='Load liver mask',
                                            directory='testdata',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0:
            img = itk.ReadImage(fname[0])
            livermask = itk.GetArrayFromImage(img)
            if livermask.shape == self.img.shape:
                self.livermask = livermask
                # self.coord.setText('肝脏分割结果已读取')
                self.statusbar.showMessage("The segmentation result of liver has been loaded")
                self.livermask = np.where(self.livermask != 0, 1, 0)
            else:
                # self.coord.setText('请导入与图像对应的分割结果')
                self.statusbar.showMessage("Please load a correspronding segmentation result")
    def showclion(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load factors',
                                            directory='testdata',
                                            filter="CSV(*.csv)")
        if len(fname[1]) != 0:
            df = pd.read_csv(fname[0], encoding='windows-1252')
            # CSV应包含11列特征（按顺序: Age, Sodium, APRI, CP Score, Conversion,
            #   Splenic Vein, PV Diameter, SFVS CT Change Rate, Median Angle SFVS,
            #   GRS Diameter, Paraumbilical Vein）
            data = df.iloc[:, :].values
            if data.shape[1] >= 12:
                data = data[:, 1:]  # 跳过ID列
            if data.shape[1] >= 11:
                self.clion = data[:1, :11].astype(np.float64)
                self.clion_CT = True
                self.clion_radiomics = True
                # 填充输入框
                boxes = [self.box1, self.box2, self.box3, self.box4, self.box5,
                         self.box6, self.box7, self.box8, self.box9, self.box10, self.box11]
                for i, box in enumerate(boxes):
                    box.setText(str(self.clion[0, i]))
                self.statusbar.showMessage("All 11 factors have been loaded from CSV")
            else:
                self.statusbar.showMessage(f"CSV needs at least 11 feature columns, got {data.shape[1]}")

    def is_number(self, str):
        try:
            # 因为使用float有一个例外是'NaN'
            if str=='NaN':
                return False
            float(str)
            return True
        except ValueError:
            return False

    def readnumresult(self):
        # self.printdim_r.setPlaceholderText('请输入')
        # print(self.printdim.placeholderText())
        se = self.printdim_r.text()
        if self.prinimg is not None and self.prinimg.ndim == 4 and self.heatmaprongqi is not None:
            if self.is_number(se) == True:
                se = float(se)
                if se <= 1 and se >= 0:
                    # alpha = self.resultprint / 100
                    self.prinimg = self.ori_prinimg
                    self.trans_prinimg(se)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)
                    self.statusbar.showMessage(f"The new transparency is:{se}")
                else:
                    # self.explain_CAM.setText('错误的输入')
                    self.statusbar.showMessage("Wrong input")
            elif self.is_number(se) == False:
                self.statusbar.showMessage("Wrong input")
        elif self.prinimg is not None and self.prinimg.ndim == 4 and self.showmask is not None:
            if self.is_number(se) == True:
                se = float(se)
                if se <= 1 and se >= 0:
                    self.prinimg = self.ori_prinimg
                    self.printmask(se)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line_x(self.x_x, self.x_y)
                    self.draw_line_y(self.y_x, self.y_y)
                    self.draw_line_z(self.z_x, self.z_y)
                else:
                    # self.explain_CAM.setText('错误的输入')
                    self.statusbar.showMessage("Wrong input")
            elif self.is_number(se) == False:
                self.statusbar.showMessage("Wrong input")
        elif self.prinimg is None:
            self.statusbar.showMessage("Please import data first")
        else:
            # self.explain_CAM.setText('请先导入数据')
            self.statusbar.showMessage("Please import data first")

    def embedinput(self, x):
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] + i * 2
        return x

    # def load_GPUS(self, model):
    #     state_dict = torch.load('model78.pkl', map_location='cpu')
    #     # create new OrderedDict that does not contain `module.`
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove `module.`
    #         new_state_dict[name] = v
    #     # load params
    #     model.load_state_dict(new_state_dict)

    def show_message_predict(self):
        QMessageBox.information(self, "Note", "The prediction is about to be made and will take longer if no "
                                              "segmentation results are entered and automatic segmentation is not performed",
                                QMessageBox.Yes)

    def show_message_compute_volume(self):
        QMessageBox.information(self, "Note", "If the mask is not imported in advance and there are no related "
                                              "files in the directory, the segmentation will be performed automatically, "
                                              "which will take a longer time",
                                QMessageBox.Yes)

    def show_message_radiomics(self):
        QMessageBox.information(self, "Note", "This step requires the segmentation result of liver and spleen; "
                                              "if masks are not imported in advance, the segmentation will be performed "
                                              "automatically, which will take longer",
                                QMessageBox.Yes)


    def show_message_clinicf(self):
        QMessageBox.information(self, "Note", "If the CSV file has been read, "
                                              "this operation overwrites the data imported from the CSV file",
                                QMessageBox.Yes)

    def predictf(self):
        if self.clion is not None and self.clion_CT is not None and self.clion_radiomics is not None:
            QMessageBox.information(self, "Note", "Start to predict",
                                    QMessageBox.Yes)
            self.statusbar.showMessage("Predicting, please wait a moment.")

            # 使用11特征管道模型直接预测（模型内含StandardScaler）
            pred = self.model.predict_proba(self.clion)[0, 1]

            if pred > self.youden_threshold:
                self.plotresult.setTextColor(QtCore.Qt.red)
            else:
                self.plotresult.setTextColor(QtCore.Qt.green)
            self.plotresult.setText("{0:0.5f}".format(pred))
            self.statusbar.showMessage(f"Prediction: {pred:.5f} ({'High Risk' if pred > self.youden_threshold else 'Low Risk'})")

        else:
            self.statusbar.showMessage("Incomplete input data, please input all 11 features")


    def trans_prinimg(self, alpha):
        # self.ori_prinimg = self.prinimg
        # self.prinimg = np.repeat(np.expand_dims(self.prinimg, axis=-1), 3, axis=-1)
        new_prinimg = []
        for i in range(self.leng_max):
            camone = self.heatmaprongqi[(self.leng_max-1) - i, ...]
            imgone = self.ori_prinimg[i, ...]
            maskone = self.mask[(self.leng_max-1) - i, ...]
            imgone = self.normilize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            heatmap = applyColorMap(np.uint8(255 * camone), cv2.COLORMAP_JET)
            cam_img = alpha * heatmap * maskone_ + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg


    def preprocessimg(self):
        img = self.img
        mask = self.mask
        space = self.space
        newresolutionxy = 0.7675
        newresolutionz = 1.0
        rsize = [int(img.shape[0] * space[2] / newresolutionz),
                 int(img.shape[1] * space[1] / newresolutionxy), int(img.shape[2] * space[0] / newresolutionxy)]
        space = (newresolutionxy, newresolutionxy, newresolutionz)
        # maskspace = (newresolutionxy,newresolutionxy,newresolutionz)
        img = st.resize(img, output_shape=rsize, order=1, mode='constant', clip=False, preserve_range=True)
        mask = st.resize(mask, output_shape=rsize,order=0, mode='constant', clip=False, preserve_range=True)
        img = np.clip(img, -17.0, 201.0)
        img = (img - 99.40078) / 39.392952
        return img, mask, space

    def show_message_liver(self):
        QMessageBox.information(self, "提示", "即将提取肝脏，这将花费一定时间",
                                QMessageBox.Yes)

    def show_message_tumor(self):
        QMessageBox.information(self, "提示", "即将提取脾脏，这将花费一定时间",
                                QMessageBox.Yes)

    def liver_seg(self):
        # self.coord.setText(f"开始提取肝脏,请稍等")
        if self.img is not None:
            self.statusbar.showMessage("Begin to segment liver, please wait a moment")
            self.show_message_liver()
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))
            # inference.predict_simple.main()
            # os.system("nnUNet_predict -i testdata/nnUNet_in -o testdata/liver_output  -t 501 -f all  "
            #           "-m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
            os.system("nnUNet_predict -i  testdata/nnUNet_in  -o testdata/liver_output  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
                         -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta")
            readmask = itk.ReadImage(os.path.join('testdata/liver_output', 'Liver_0.nii.gz'))
            space = readmask.GetSpacing()
            origin = readmask.GetOrigin()
            direction = readmask.GetDirection()
            readmask = itk.GetArrayFromImage(readmask)
            readmask[readmask != 1] = 0
            readmask = itk.GetImageFromArray(readmask)
            readmask.SetOrigin(origin)
            readmask.SetSpacing(space)
            readmask.SetDirection(direction)
            self.livermask_path = self.filename.split('.nii.gz')[0] + '_liver.nii.gz'
            itk.WriteImage(readmask, self.livermask_path)
            os.remove(os.path.join('testdata/liver_output', 'Liver_0.nii.gz'))
            os.remove(os.path.join('testdata/liver_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))
            self.showmask_path(self.livermask_path)

            livermask = itk.ReadImage(self.livermask_path)
            self.livermask = itk.GetArrayFromImage(livermask)
            #self.livermask = np.where(self.livermask != 0, 1, 0)
            # 仅保留 mask 数组中值为1的部分
            # self.livermask = np.where(self.livermask == 1, 1, 0)
            # self.coord.setText(f"肝脏分割已完成")
            self.statusbar.showMessage("Liver segmentation has been done")
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load a image first')

    def get_livermask(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))
            #os.system("nnUNet_predict -i testdata/nnUNet_in -o testdata/liver_output  -t 501 -f all  "
                     # "-m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
            os.system("nnUNet_predict -i  testdata/nnUNet_in  -o testdata/liver_output  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
                                    -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta")

            # readmask = nib.load(os.path.join('testdata/liver_output', 'Liver_0.nii.gz'))
            # readmask[readmask != 1] = 0
            readmask = itk.ReadImage(os.path.join('testdata/liver_output', 'Liver_0.nii.gz'))
            space = readmask.GetSpacing()
            origin = readmask.GetOrigin()
            direction = readmask.GetDirection()
            readmask = itk.GetArrayFromImage(readmask)
            readmask[readmask != 1] = 0
            readmask = itk.GetImageFromArray(readmask)
            readmask.SetOrigin(origin)
            readmask.SetSpacing(space)
            readmask.SetDirection(direction)
            self.livermask_path = self.filename.split('.nii.gz')[0] + '_liver.nii.gz'
            itk.WriteImage(readmask, self.livermask_path)
            os.remove(os.path.join('testdata/liver_output', 'Liver_0.nii.gz'))
            os.remove(os.path.join('testdata/liver_output', 'plans.pkl'))
            os.remove(os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))

            livermask = itk.ReadImage(self.livermask_path)
            self.livermask = itk.GetArrayFromImage(livermask)
            #self.livermask = np.where(self.livermask != 0, 1, 0)
            # 仅保留 mask 数组中值为1的部分
            # self.livermask = np.where(self.livermask == 1, 1, 0)
            # self.coord.setText(f"肝脏分割已完成")
            self.statusbar.showMessage("Liver segmentation has been done")
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage("Please load a image first")

    def GetLargestConnectedCompont(self, binaryitk_image):
        cc = itk.ConnectedComponent(binaryitk_image)
        stats = itk.LabelIntensityStatisticsImageFilter()
        stats.SetGlobalDefaultNumberOfThreads(8)
        stats.Execute(cc, binaryitk_image)
        maxlabel = 0
        maxsize = 0
        for l in stats.GetLabels():
            size = stats.GetPhysicalSize(l)
            if maxsize < size:
                maxlabel = l
                maxsize = size
        labelmaskimage = itk.GetArrayFromImage(cc)
        outmask = labelmaskimage.copy()
        outmask[labelmaskimage == maxlabel] = 1
        outmask[labelmaskimage != maxlabel] = 0
        outmask_itk = itk.GetImageFromArray(outmask)
        outmask_itk.SetDirection(binaryitk_image.GetDirection())
        outmask_itk.SetSpacing(binaryitk_image.GetSpacing())
        outmask_itk.SetOrigin(binaryitk_image.GetOrigin())
        return outmask_itk

    def tumor_seg(self):
        # self.coord.setText(f"开始提取肿瘤,请稍等")

        if self.img is not None:
            readimg = nib.load(self.filename)
            self.statusbar.showMessage("Start to segment spleen, please wait a moment")
            self.show_message_tumor()
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))
            #os.system("nnUNet_predict -i testdata/nnUNet_in "
                   #   "-o testdata/spleen_output -t 502 -f all  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
            os.system("nnUNet_predict -i  testdata/nnUNet_in  -o testdata/spleen_output  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
                                    -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta")

            # readmask = nib.load(os.path.join('testdata/spleen_output', 'Liver_0.nii.gz'))
            readmask = itk.ReadImage(os.path.join('testdata/spleen_output', 'Liver_0.nii.gz'))
            space = readmask.GetSpacing()
            origin = readmask.GetOrigin()
            direction = readmask.GetDirection()
            readmask = itk.GetArrayFromImage(readmask)
            readmask[readmask != 3] = 0
            readmask[readmask == 3] = 1
            readmask = itk.GetImageFromArray(readmask)
            readmask.SetOrigin(origin)
            readmask.SetSpacing(space)
            readmask.SetDirection(direction)

            self.mask_path = self.filename.split('.nii.gz')[0] + '_spleen.nii.gz'
            # nib.save(readmask, self.mask_path)
            itk.WriteImage(readmask, self.mask_path)
            sitk_src = itk.ReadImage(self.mask_path, itk.sitkUInt8)
            sitk_open = self.GetLargestConnectedCompont(sitk_src)
            itk.WriteImage(sitk_open, self.mask_path)
            os.remove(os.path.join('testdata/spleen_output', 'Liver_0.nii.gz'))
            os.remove(os.path.join('testdata/spleen_output', 'plans.pkl'))
            self.showmask_path(self.mask_path)

            mask = itk.ReadImage(self.mask_path)
            self.mask = itk.GetArrayFromImage(mask)
            self.mask = np.where(self.mask != 3, 1, 0)
            # self.coord.setText(f"肿瘤分割已完成")
            self.statusbar.showMessage("Spleen segmentation has been done")
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load image first')

    def get_tumormask(self):
        if self.img is not None:
            readimg = nib.load(self.filename)
            nib.save(readimg, os.path.join('testdata/nnUNet_in', 'Liver_0_0000.nii.gz'))
            #os.system("nnUNet_predict -i testdata/nnUNet_in "
                     # "-o testdata/spleen_output -t 502 -f all  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
            os.system("nnUNet_predict -i  testdata/nnUNet_in  -o testdata/spleen_output  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
                                    -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta")
            readmask = itk.ReadImage(os.path.join('testdata/spleen_output', 'Liver_0.nii.gz'))
            space = readmask.GetSpacing()
            origin = readmask.GetOrigin()
            direction = readmask.GetDirection()
            readmask = itk.GetArrayFromImage(readmask)
            readmask[readmask != 3] = 0
            readmask[readmask == 3] = 1
            readmask = itk.GetImageFromArray(readmask)
            readmask.SetOrigin(origin)
            readmask.SetSpacing(space)
            readmask.SetDirection(direction)
            self.mask_path = self.filename.split('.nii.gz')[0] + '_spleen.nii.gz'
            itk.WriteImage(readmask, self.mask_path)
            sitk_src = itk.ReadImage(self.mask_path, itk.sitkUInt8)
            sitk_open = self.GetLargestConnectedCompont(sitk_src)
            itk.WriteImage(sitk_open, self.mask_path)
            os.remove(os.path.join('testdata/spleen_output', 'Liver_0.nii.gz'))
            os.remove(os.path.join('testdata/spleen_output', 'plans.pkl'))

            mask = itk.ReadImage(self.mask_path)
            self.mask = itk.GetArrayFromImage(mask)
            self.mask = np.where(self.mask != 3, 1, 0)
            # self.coord.setText(f"肿瘤分割已完成")
            self.statusbar.showMessage("Spleen segmentation has been done")
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage("Please load a image first")


    def extract_liver(self,img, mask):
        wholemask = 0
        if self.livermask is not None:

            self.livermask = np.where(self.livermask != 0, 1, 0)

            if np.prod(self.livermask.shape) != np.prod(img.shape):
                self.livermask = st.resize(self.livermask, output_shape=img.shape, order=0, mode='constant', clip=False, preserve_range=True)

            # pred = np.where(mask != 0, 2, pred)
            ind = np.where(mask != 0)[2]
            if int(0.6 * img.shape[2]) < ind.max():
                self.flag = True
                img = img * self.livermask
                img = img[:, :, :ind.max()]
                ind = np.where(img != 0)
                segimg = img[np.min(ind[0]): np.max(ind[0]), np.min(ind[1]): np.max(ind[1]), np.min(ind[2]): np.max(ind[2])]
            else:
                self.flag = False
                oriimg = img
                # orimask = mask
                img = img * self.livermask
                img = img[:, :, :int(0.6 * oriimg.shape[2])]
                # mask = mask[:, :, :int(0.6 * oriimg.shape[2])]
                ind = np.where(img != 0)
                # print(np.min(ind[0]), np.max(ind[0]), np.min(ind[1]), np.max(ind[1]), np.min(ind[2]), np.max(ind[2]))
                segimg = img[np.min(ind[0]): np.max(ind[0]), np.min(ind[1]): np.max(ind[1]),
                         np.min(ind[2]): np.max(ind[2])]
                # segmask = mask[np.min(ind[0]): np.max(ind[0]), np.min(ind[1]): np.max(ind[1]),
                #           np.min(ind[2]): np.max(ind[2])]

            return segimg, ind
        else:
            return False, False


    def gettimes(self, new_shape, input_size, strides):
        num_x = 1 + math.ceil((new_shape[0] - input_size[0]) / strides[0])
        num_y = 1 + math.ceil((new_shape[1] - input_size[1]) / strides[1])
        num_z = 1 + math.ceil((new_shape[2] - input_size[2]) / strides[2])
        # print(CT_origianl.shape, change_spacing_shape, new_shape, num_x, num_y, num_z)
        return num_x, num_y, num_z

    # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def normilize(self, x):
        maa = x.max()
        mii = x.min()
        x = (x - mii) * 255 / (maa - mii)
        return x

    def cam_show_img3d(self, img, oriimg, mask, orimask,  getind, feature_map, grads, out_dir):
        oH, oW, oL = oriimg.shape
        heatmaprongqi = np.zeros(oriimg.shape, dtype=float)
        ind = np.where(mask != 0)[2]
        if not self.flag:
            cutheatmaprongqi = np.zeros([oH, oW, int(0.6 * oL)], dtype=float)
        else:
            cutheatmaprongqi = np.zeros([oH, oW, ind.max()], dtype=float)
        # print('2', cutheatmaprongqi.shape) # (205, 435, 435)

        H, W, L = img.shape
        # print('3', img.shape) # (149, 219, 212)
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
        grads = grads.reshape([grads.shape[0], -1])  # 5
        weights = np.mean(grads, axis=1)  # 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :, :]  # 7
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = st.resize(cam, (H, W, L), order=1, clip=False, preserve_range=True)

        cutheatmaprongqi[np.min(getind[0]): np.max(getind[0]), np.min(getind[1]): np.max(getind[1]),
        np.min(getind[2]): np.max(getind[2])] = cam

        if not self.flag:
            heatmaprongqi[:, :, :int(0.6 * oL)] = cutheatmaprongqi
        else:
            heatmaprongqi[:, :, :ind.max()] = cutheatmaprongqi


        self.heatmaprongqi = st.resize(heatmaprongqi, output_shape=self.img.shape,  clip=False, preserve_range=True)
        self.trans_prinimg(0.3)

        self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
        self.draw_line_x(self.x_x, self.x_y)
        self.draw_line_y(self.y_x, self.y_y)
        self.draw_line_z(self.z_x, self.z_y)

    def eventFilter(self, source, event):
        if source is self.view1:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 1
                return True

        elif source is self.view2:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 2
                return True


        elif source is self.view3:
            # print(event.type)
            if event.type() == QtCore.QEvent.HoverMove:
                self.face_flage = 3
                # print('3')
                return True
        else:
            self.face_flage = 0

        return False

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img + 1
                if self.leng_img >= self.leng_max:
                    self.leng_img = self.leng_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img + 1
                if self.width_img >= self.width_max:
                    self.width_img = self.width_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img + 1
                if self.high_img >= self.high_max:
                    self.high_img = self.high_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

        elif event.angleDelta().y() < 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img - 1
                if self.leng_img < 0:
                    self.leng_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img - 1
                if self.width_img < 0:
                    self.width_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)

            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img - 1
                if self.high_img < 0:
                    self.high_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line_x(self.x_x, self.x_y)
                self.draw_line_y(self.y_x, self.y_y)
                self.draw_line_z(self.z_x, self.z_y)




    def render_3d_matplotlib(self, mask_data, spacing=(1, 1, 1)):
        """直接渲染mask的所有非零体素为3D散点图（无任何表面提取处理）"""
        try:
            mask_bin = (mask_data > 0)
            if mask_bin.sum() == 0:
                self.statusbar.showMessage("Mask is empty, cannot render 3D")
                return

            # 获取所有非零体素坐标
            z_idx, y_idx, x_idx = np.where(mask_bin)
            # 转换为物理坐标
            z_phys = z_idx * spacing[2]
            y_phys = y_idx * spacing[1]
            x_phys = x_idx * spacing[0]

            # 如果点太多(>50000)，随机采样以加速渲染
            n_points = len(z_idx)
            max_points = 50000
            if n_points > max_points:
                indices = np.random.choice(n_points, max_points, replace=False)
                z_phys = z_phys[indices]
                y_phys = y_phys[indices]
                x_phys = x_phys[indices]

            fig = plt.figure(figsize=(5.04, 4.24), dpi=100, facecolor='black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')

            ax.scatter(x_phys, y_phys, z_phys, c='red', s=0.3, alpha=0.6, edgecolors='none')

            ax.set_axis_off()
            # 等比例
            max_range = max(x_phys.max()-x_phys.min(), y_phys.max()-y_phys.min(), z_phys.max()-z_phys.min()) / 2
            mid = [(x_phys.max()+x_phys.min())/2, (y_phys.max()+y_phys.min())/2, (z_phys.max()+z_phys.min())/2]
            ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
            ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
            ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
            ax.view_init(elev=20, azim=-60)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close(fig)
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            pixmap = pixmap.scaled(504, 424, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            if not hasattr(self, '_mpl_3d_label') or self._mpl_3d_label is None:
                self._mpl_3d_label = QtWidgets.QLabel(self.pic_box_vface)
                self._mpl_3d_label.setGeometry(0, 0, 504, 424)
                self._mpl_3d_label.setAlignment(QtCore.Qt.AlignCenter)
                self._mpl_3d_label.setStyleSheet("background-color: black;")
            self._mpl_3d_label.setPixmap(pixmap)
            self._mpl_3d_label.show()

            self.statusbar.showMessage(f"3D rendered ({min(n_points, max_points)} points)")
        except Exception as e:
            print(f"[WARNING] matplotlib 3D rendering failed: {e}")
            import traceback
            traceback.print_exc()
            self.statusbar.showMessage(f"3D rendering failed: {e}")


if __name__ == '__main__':
    # gpu_id = "4"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    mw = MyWindow1()
    mw.show()
    sys.exit(app.exec_())