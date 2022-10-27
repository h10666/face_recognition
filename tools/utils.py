import os.path
import time

import cv2
import dlib

import tools.db


def face_distance(f1, f2):
    """计算人脸距离,需传入同维度的人脸特征, 官方文档说明距离小于0.6可认为同一个人"""
    import numpy as np
    feature1, feature2 = np.array(f1), np.array(f2)
    return np.sqrt(np.sum(np.square(feature1 - feature2)))


def is_same_person(f1, f2):
    """
    传入两个人脸特征表示
    :return: 1为同一个人, 0 为不同的人
    """
    return face_distance(f1, f2) < 0.4


class FaceRecognition:
    def __init__(self, con: tools.db.MyDB):
        """
        :param con:数据库对象
        """
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')  # 人脸关键点检测器
        self.detector = dlib.get_frontal_face_detector()  # 检测人脸位置
        self.model = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.con = con
        self.saved_feature = con.load_all_feature()

    def register_face(self, name: str, img):
        """

        :param name: 姓名
        :param img: 包含人脸的图片路径,或者给定任意数字调用摄像头，仅当摄像头中存在一个人脸时注册该人脸
        :return: 0为失败，1为成功
        """
        if isinstance(img, str):
            if os.path.isfile(img):
                face_feature = self.extract_feature_128d(img)
            else:
                print('路径不存在')
                return 0
        else:
            face = self._get_one_face_by_camera()
            face_feature = self.extract_feature_128d(face)
        self.con.insert_feature(name, face_feature)
        return 1

    def extract_feature_128d(self, img):
        """
        抽取图片中人脸的128维特征，图片中仅能存在一个人脸
        :param img: 图片路径或打开的图片
        :return: 人脸的128维特征
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        if img.ndim == 2:
            img_gray = img
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(img_gray, 0)
        if len(faces) != 1:
            return None
        shape = self.predictor(img_gray, faces[0])
        return self.model.compute_face_descriptor(img, shape)

    def _get_one_face_by_camera(self):
        """通过摄像头获取一张人脸图片"""
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        if not success:
            cap.release()
            return

        while True:
            success, img = cap.read()
            if ord('q') == cv2.waitKey(1):
                cv2.destroyAllWindows()  # 当接收到q，摧毁窗口
                break
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray, 1)
            if len(faces) > 1:
                continue
            return img

    def show_keypoint_by_camera(self):
        """调用设备摄像头，检测人脸，并在人脸上显示关键点, 按q或esc退出"""
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        if not success:
            cap.release()
            return

        while True:
            success, img = cap.read()
            if ord('q') == cv2.waitKey(1):
                cv2.destroyAllWindows()  # 当接收到q，摧毁窗口
                break
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray, 1)
            for face in faces:
                shape = self.predictor(img_gray, face)
                for index, pt in enumerate(shape.parts()):
                    pt_pos = (pt.x, pt.y)
                    cv2.circle(img, pt_pos, 1, (255, 0, 0), 2)
                    cv2.putText(img, str(index + 1), pt_pos, self.font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('img', img)
        cap.release()

    def video_show(self):
        """
        调用摄像头识别人脸
        """
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        if not success:
            cap.release()
            return
        cv2.imshow('img', img)
        while True:
            success, img = cap.read()
            start_time = time.time()
            if ord('q') == cv2.waitKey(1):
                cv2.destroyAllWindows()  # 当接收到q，摧毁窗口
                break
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector(img_gray, 0)
            for face in faces:
                shape = self.predictor(img_gray, face)
                feature = self.model.compute_face_descriptor(img, shape)
                cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
                for _saved_feature in self.saved_feature.items():
                    if is_same_person(feature, _saved_feature[1]):
                        print(time.strftime('%H:%M:%S') + '存在' + _saved_feature[0])
                        cv2.putText(img, _saved_feature[0], (face.left(), face.top()), self.font, 0.3, (0, 0, 255), 1,
                                    cv2.LINE_AA)
            end_time = time.time()
            FPS = int(1 / (end_time - start_time))
            cv2.putText(img, 'FPS:' + str(FPS), (10, 10), self.font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('img', img)
        cap.release()
