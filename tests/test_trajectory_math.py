import numpy as np
import pytest
from modules.trajectory.trajectory_math import sk, unsk, q2R, R2q, qProd

class TestTrajectoryMath:
    def test_skew_symmetric_mapping(self):
        u = np.array([1.0, 2.0, 3.0])
        U_x = sk(u)
        
        np.testing.assert_array_almost_equal(U_x.T, -U_x, err_msg="Matrix transpose does not equal its negative.")
        
        u_recovered = unsk(U_x)
        np.testing.assert_array_almost_equal(u, u_recovered, err_msg="unsk did not recover the original vector.")

    def test_quaternion_rotation_conversion(self):
        angle = np.pi / 2.0
        q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
        
        R = q2R(q)
        
        R_expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(R, R_expected, err_msg="q2R failed to produce the correct rotation matrix.")
        
        q_recovered = R2q(R)
        
        sign = np.sign(q[0] * q_recovered[0])
        np.testing.assert_array_almost_equal(q, sign * q_recovered, err_msg="R2q did not recover the equivalent quaternion.")

    def test_quaternion_product_identity(self):
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        q_target = np.array([0.0, 1.0, 0.0, 0.0])
        
        q_result = qProd(q_identity, q_target)
        
        np.testing.assert_array_almost_equal(q_result, q_target, err_msg="qProd failed the identity property.")

    def test_so3_log_vec(self):
        from modules.trajectory.trajectory_math import so3_log_vec
        
        angle = np.pi / 3.0
        axis = np.array([0.0, 1.0, 0.0])
        R = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)]
        ])
        
        log_map_result = so3_log_vec(R)
        expected_result = angle * axis
        
        np.testing.assert_array_almost_equal(log_map_result, expected_result, err_msg="so3_log_vec failed to map rotation matrix to rotation vector.")