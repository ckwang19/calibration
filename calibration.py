# calibration
# https://blog.csdn.net/u010128736/article/details/52875137
# pose estimate 
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
# calibration + perspective
# http://road2ai.info/2019/01/01/Nano01_02_L07/
# perspective transform with PIL
# https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
# rvec->rmatrix by cv2.Rodrigues() 
# http://hant.ask.helplib.com/python/post_5579275

import numpy as np
import cv2
import glob

x = 9
y = 6

class Image():
    @staticmethod
    def draw_im(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        print (corner)
        print (tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    @staticmethod
    def save_im(img, save_path):
        cv2.imwrite(save_path, img)

    @staticmethod
    def show_im(img, msec):
        cv2.imshow('img',img)
        cv2.waitKey(msec)

    @staticmethod
    def get_im_path(images_base_path):
        return glob.glob(images_base_path)

class Camera():
    def __init__(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((y*x,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1,2)

    def initial_param(self):
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

    def set_im(self, im_path):
        self.rgb_im = cv2.imread(im_path)
        self.gray_im = cv2.cvtColor(self.rgb_im, cv2.COLOR_BGR2GRAY)

    def find_corners(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(self.gray_im, (x,y), None)
        self.corners = cv2.cornerSubPix(self.gray_im,corners,(11,11),(-1,-1),criteria)
        self.objpoints.append(self.objp)
        self.imgpoints.append(self.corners)
        return ret

    def calibration(self):
        img_size = (self.rgb_im.shape[1], self.rgb_im.shape[0])
        ret, self.intri_mtx, self.distort_coef, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

    def undistort(self, im_path):
        self.undistort_img = cv2.undistort(self.rgb_im, self.intri_mtx, self.distort_coef, None, self.intri_mtx)
        save_path = im_path.replace('ori', 'undistortion')
        cv2.imwrite(save_path, self.undistort_img)

    def camera_process(self, im_path):
        ret = self.find_corners()
        if ret:
            self.calibration()
            self.undistort(im_path)
        return ret

    def project_point(self):
        for i in range(len(self.objpoints)):
            imgpoints2, jac = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.intri_mtx, self.distort_coef)
            #print ("imgpoints2: {}".format(imgpoints2))
            #print ("objpoints[i]: {}".format(self.objpoints[i]))
            #print ("rvecs[i]: {}".format(self.rvecs[i]))
            #print ("tvecs[i]: {}".format(self.tvecs[i]))
            #print ("intri_mtx: {}".format(self.intri_mtx))
            #print ("distort_coef: {}".format(self.distort_coef))
            #input('wait')

    def get_param(self):
        axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
        return self.intri_mtx, self.distort_coef, axis

    def get_rt_matrix(self, rvecs, tvecs):
        R = np.zeros((4,4), np.float32)
        R[3,3] = 1.0
        T = np.zeros((4,4), np.float32)
        T[0] = (1,0,0,0)
        T[1] = (0,1,0,0)
        T[2] = (0,0,1,0)
        T[3] = (0,0,0,1)
        rodRotMat = cv2.Rodrigues(rvecs)
        R[:3,:3] = rodRotMat[0]
        T[0,3] = tvecs[0]
        T[1,3] = tvecs[1]
        T[2,3] = tvecs[2]
        return R, T

    def world_to_im(self, A1, R, T, mtx):
        first = np.dot(R, A1)
        second = np.dot(T, first)[0:3]
        #print (second)
        finalCalc = np.dot(mtx, second)
        finalNorm = finalCalc/(finalCalc[2][0])
        return finalNorm

def get_world_point():
    A1 = np.zeros((4,1), np.float32)
    A1[0] = (1)
    A1[1] = (0)
    A1[2] = (0)
    A1[3] = (1)
    return A1

def main():
    camera = Camera()
    im = Image()
    images_path = im.get_im_path('im/ori/*.jpg')

    for im_path in images_path:
        camera.initial_param()
        camera.set_im(im_path)
        ret = camera.camera_process(im_path) #get R & T vector、undistortion
        if ret:
            camera.project_point()
            mtx, dist, axis = camera.get_param()

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(camera.objp, camera.corners, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            R, T = camera.get_rt_matrix(rvecs, tvecs)
            world_point = get_world_point()
            im_point = camera.world_to_im(world_point, R, T, mtx)

            img = im.draw_im(camera.rgb_im,camera.corners,imgpts)
            im.show_im(img,msec=500)
            im.save_im(img, im_path.replace('ori', 'pose'))

if __name__ == '__main__':
    main()
