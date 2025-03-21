import numpy as np
from scipy.spatial.transform import Rotation as Rot

def parse_pose_args(pose_args):
    if len(pose_args) == 2:
        pos, orn = pose_args
        pose = Transform(orn, pos)
    elif isinstance(pose_args, Transform):
        pose = pose_args
    else:
        raise ValueError
    return pose


class Transform:
    def __init__(self, *args):
        if len(args) == 0:
            raise NotImplementedError
        elif len(args) == 7:
            raise NotImplementedError
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Transform):
                self.T = arg.T
            elif isinstance(arg, np.ndarray) and arg.shape == (4, 4):
                self.T = arg
            else:
                raise NotImplementedError
        elif len(args) == 2:
            arg_0_array = np.asarray(args[0])
            n_elem_rot = len(arg_0_array.flatten())
            if n_elem_rot == 4:
                xyzw = np.asarray(args[0]).flatten()
                wxyz = [xyzw[-1], *xyzw[:-1]]
                assert len(wxyz) == 4
                q = Rot.from_quat(wxyz)
                R = q.as_matrix()
            elif n_elem_rot == 9:
                assert arg_0_array.shape == (3, 3)
                R = arg_0_array
            t = np.asarray(args[1])
            assert len(t) == 3
            self.T = np.hstack((R, t.reshape(3, 1)))
            self.T = np.vstack((self.T, np.array([0, 0, 0, 1])))
        else:
            raise NotImplementedError

    def __mul__(self, other):
        T = np.dot(self.T, other.T)
        return Transform(T)

    def __matmul__(self, other):
        if not isinstance(other, Transform):
            raise NotImplementedError
        return self.__mul__(other)

    def inverse(self):
        R_inv = self.T[:3, :3].T
        t_inv = -np.dot(R_inv, self.T[:3, 3])
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return Transform(T_inv)

    def __str__(self):
        return str(self.T)

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return 7

    def toHomogeneousMatrix(self):
        return self.T

    @property
    def translation(self):
        return self.T[:3, 3]

    @property
    def quaternion(self):
        R = self.T[:3, :3]
        q = Rot.from_matrix(R).as_quat()
        return q