import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from gtsam import (
    ISAM2, NonlinearFactorGraph, Values, Pose3, Point3, Rot3,
    PreintegrationParams, PreintegratedImuMeasurements,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias,
    BetweenFactorConstantBias, ImuFactor, noiseModel, imuBias, BetweenFactorPose3)
from visualization import plotting
from visualization.video import play_trip
from bag_of_words import BoW, make_stackimage
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import X, V, B
from dataclasses import dataclass
from typing import List, Tuple 
from gtsam.utils import plot as gplot
from gtsam import Marginals
@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray  # Shape (3,), in body frame [ax, ay, az]
    gyro: np.ndarray   # Shape (3,), in body frame [wx, wy, wz]
def np_Rt_to_Pose3(T:np.ndarray) -> Pose3:
    """Convert a 4x4 numpy array to a gtsam Pose3 object."""
    R = Rot3(T[:3, 0:3])
    t = Point3(T[:3, 3])
    return Pose3(R, t)
def vector3(x,y,z):
    return np.array([x,y,z],dtype=float)
g = 9.81
n_gravity = vector3(0, 0, -g)
class VisualOdometry():
    def __init__(self, data_dir):
        # Store the base directory for context
        self.base_dir = os.path.abspath(os.path.dirname(data_dir))
        self.data_dir = data_dir
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
        self.gt_poses = self._load_poses(data_dir + '/poses.txt')
        self.images_l = self._load_images(data_dir + '/image_l')
        self.images_r = self._load_images(data_dir + '/image_r')
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
                # --- IMU / iSAM2 initialization ---
        self.params = PreintegrationParams.MakeSharedU(g)   # gravity along -Z
        I3 = np.eye(3)
        self.params.setAccelerometerCovariance(I3 * 0.1)
        self.params.setGyroscopeCovariance(I3 * 0.1)
        self.params.setIntegrationCovariance(I3 * 0.1)

        # Factor graph + solver
        self.graph = NonlinearFactorGraph()
        self.initial = Values()
        self.isam = ISAM2()

        # Keys & noises
        self.k = 0
        self.has_initialized_isam = False
        self.bias_key = B(0)
        self.cur_vel_key = V(0)
        self.pose_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])    # rx,ry,rz, tx,ty,tz
        )
        self.vel_prior_noise  = noiseModel.Isotropic.Sigma(3, 0.1)
        self.bias_prior_noise = noiseModel.Isotropic.Sigma(6, 0.1)

        # Initial bias and preintegrator (this is the 'accum' your error mentions)
        self.bias0 = imuBias.ConstantBias()
        self.accum = PreintegratedImuMeasurements(self.params, self.bias0)

        # Optional: placeholders for IMU & camera times (fill later)
        self.imu = []          # list of ImuSample(t, acc, gyro) or similar tuples
        self.cam_times = []    # per-frame timestamps in seconds

                # --- VO-only factor graph setup (works with no IMU) ---
        self.graph = NonlinearFactorGraph()
        self.isam = ISAM2()
        self.initial = Values()

        # Index of current pose in the graph
        self.k = 0
        self.has_initialized_isam = False

        # Noise models (tune as needed)
        # Pose prior noise: [rx, ry, rz, tx, ty, tz] sigmas
        self.pose_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.20, 0.20, 0.20], dtype=float)
        )
        # VO odometry noise (between-factor)
        self.odo_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10], dtype=float)
        )

    def _initialize_isam(self, pose0: Pose3):
        # Pose prior at k=0
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        # Velocity prior at k=0
        self.graph.add(PriorFactorVector(V(0), np.zeros(3), self.vel_prior_noise))
        self.initial.insert(V(0), np.zeros(3))
        # Bias prior at B(0)
        self.graph.add(PriorFactorConstantBias(B(0), self.bias0, self.bias_prior_noise))
        self.initial.insert(B(0), self.bias0)

        # Initial pose value
        self.initial.insert(X(0), pose0)

        # First update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()
        self.has_initialized_isam = True

    def _imu_window(self, t0, t1):
        return [s for s in self.imu if (s.t > t0 and s.t <= t1)]

    def fuse_vo_imu_step(self, T_km1_k_4x4: np.ndarray, t_km1: float | None = None, t_k: float | None = None) -> Pose3:
        # Initialize once
        if not self.has_initialized_isam:
            pose0 = Pose3()  # or Pose3 from your first GT pose
            self._initialize_isam(pose0)

        k = self.k + 1

        # Convert VO delta to Pose3
        R = Rot3(T_km1_k_4x4[:3, :3])
        t = Point3(*T_km1_k_4x4[:3, 3])
        T_meas = Pose3(R, t)
        odo_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10])
        )

        # ---- Try to add IMU factor if we have samples in (t_{k-1}, t_k] ----
        added_imu = False
        if t_km1 is not None and t_k is not None and hasattr(self, "imu") and len(self.imu) > 0:
            window = self._imu_window(t_km1, t_k)
            if len(window) > 0:
                self.accum.resetIntegration()
                prev_t = t_km1
                for s in window:
                    dt = max(1e-6, s.t - prev_t)
                    self.accum.integrateMeasurement(s.acc, s.gyro, dt)
                    prev_t = s.t
                # IMU factor couples (X(k-1),V(k-1),B) -> (X(k),V(k))
                self.graph.add(ImuFactor(X(k-1), V(k-1), X(k), V(k), self.bias_key, self.accum))
                added_imu = True

        # Always add a VO odometry factor so the graph is constrained even without IMU
        self.graph.add(BetweenFactorPose3(X(k - 1), X(k), T_meas, odo_noise))

        # Optional: bias chaining only matters if we’re actually using IMU
        if added_imu and (k % 5 == 0):
            new_bias_key = B(k // 5)
            self.graph.add(BetweenFactorConstantBias(self.bias_key, new_bias_key,
                                                    gtsam.imuBias.ConstantBias(),
                                                    noiseModel.Isotropic.Variance(6, 0.1)))
            self.initial.insert(new_bias_key, gtsam.imuBias.ConstantBias())
            self.bias_key = new_bias_key

        # Initial guesses for new states
        prev_pose = self.isam.calculateEstimate().atPose3(X(k - 1))
        self.initial.insert(X(k), prev_pose.compose(T_meas))
        # Velocity exists only if we’re running IMU; keep a benign guess anyway
        self.initial.insert(V(k), np.zeros(3))

        # Update & return current pose
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        self.k = k
        return result.atPose3(X(k))

    def _pose3_from_T(self, T_4x4: np.ndarray) -> Pose3:
        R = Rot3(T_4x4[:3, :3])
        t = Point3(*T_4x4[:3, 3])
        return Pose3(R, t)

    def initialize_isam_with_prior(self, T0_4x4: np.ndarray | None = None):
        """One-time: add PriorFactorPose3 at X(0), insert initial, and do first isam.update()."""
        if self.has_initialized_isam:
            return
        pose0 = self._pose3_from_T(T0_4x4) if (T0_4x4 is not None) else Pose3()

        # Prior factor at X(0)
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        # Initial guess for X(0)
        self.initial.insert(X(0), pose0)

        # First iSAM update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()
        self.k = 0
        self.has_initialized_isam = True

    def add_vo_between_and_update(self, T_km1_k_4x4: np.ndarray):
        """
        Add a VO BetweenFactorPose3 between X(k-1) and X(k), provide initial for X(k),
        then run a single iSAM2 update. Returns the current Pose3 at X(k).
        """
        # Next index
        k = self.k + 1

        # Measurement as Pose3
        T_meas = self._pose3_from_T(T_km1_k_4x4)

        # Between factor: X(k-1) --T_meas--> X(k)
        self.graph.add(BetweenFactorPose3(X(k - 1), X(k), T_meas, self.odo_noise))

        # Initial guess for X(k): compose last estimate with VO delta
        prev_pose = self.isam.calculateEstimate().atPose3(X(k - 1))
        self.initial.insert(X(k), prev_pose.compose(T_meas))

        # Incremental solve
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()

        # Reset temporary containers
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        # Advance index
        self.k = k
        return result.atPose3(X(k))

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get teh tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix

    def attach_imu(self, imu_samples: List[IMUSample], cam_times: List[float]):
        """
        Call this once before main loop.
        - imu_samples: high-rate IMU in body frame.
        - cam_times: time stamps per left image (seconds, same epoch as IMU)
        """
        self.imu = imu_samples
        self.cam_times = cam_times
    
    def _initialise_isam(self, pose0: Pose3):
        #Pose prior
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        #Bias Prior
        self.graph.add(PriorFactorConstantBias(self.bias_key, gtsam.imuBias.ConstantBias(), self.bias_prior_noise))
        self.initial.insert(self.bias_key, gtsam.imuBias.ConstantBias())
        # Velocity prior (start at zero or small value)
        self.graph.add(PriorFactorVector(self.cur_vel_key, np.zeros(3), self.vel_prior_noise))
        self.initial.insert(self.cur_vel_key, np.zeros(3))
        # Also seed pose0
        self.initial.insert(X(0), pose0)

        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        self.has_initialized_isam = True

def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip
    
    if len(vo.gt_poses) > 0:
        T0 = vo.gt_poses[0]
        vo.initialize_isam_with_prior(T0_4x4=T0)
    else:
        vo.initialize_isam_with_prior(T0_4x4=None)

    gt_path = []
    fused_path = []

    #vo.attach_imu(imu_samples, cam_times)  # Attach IMU data if available
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i == 0:
            # current fused pose is simply X(0) after prior
            est_pose = vo.isam.calculateEstimate().atPose3(X(0))
        else:
            # VO delta from k-1 to k (4x4)
            T_km1_k = vo.get_pose(i)
            # Add BetweenFactorPose3 and update iSAM2
            est_pose = vo.add_vo_between_and_update(T_km1_k)

        # Collect for your 2D HTML path plot (x,z)
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        E = est_pose.matrix()
        fused_path.append((E[0, 3], E[2, 3]))

    # Your existing HTML path visualizer
    plotting.visualize_paths(
        gt_path, fused_path, "Stereo VO (Factor Graph VO-only)",
        file_out=os.path.basename(data_dir) + "_vo_graph.html"
    )
    



if __name__ == "__main__":
    main()
