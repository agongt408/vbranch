import cv2
import numpy as np

keypoint_map = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel":24,
    "Background":25
}

def plot_im_keypoints(bodyKeypoints, im_path, title=None):
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    height, width, _ = im.shape

    ax = plt.gca()
    plt.imshow(im)
    plt.axis('off')

    for x, y, c in bodyKeypoints:
        circle = plt.Circle((x*width, y*height), radius=2, alpha=c)
        ax.add_patch(circle)

    plt.title(title)
    plt.show()

class Coord:
    def __init__(self, x, y, height=2, width=1):
        self.x = x * width
        self.y = y * height

    def __str__(self):
        return '({},{})'.format(self.x, self.y)

def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_xy(keypoints, name):
    x, y, c = keypoints[0][keypoint_map[name]]
    return Coord(x, y)

def get_pose_score(bodyKeypoints):
    RShoulder = get_xy(bodyKeypoints, 'RShoulder')
    LShoulder = get_xy(bodyKeypoints, 'LShoulder')
    RHip = get_xy(bodyKeypoints, 'RHip')
    LHip = get_xy(bodyKeypoints, 'LHip')

    eps = 1e-4
    mu = -(RShoulder.x - LShoulder.x) / (eps + abs(RShoulder.x - LShoulder.x))
    torso_width = distance(RShoulder, LShoulder)
    torso_height = 0.5*(distance(RShoulder,RHip)+distance(LShoulder,LHip))+eps

    return mu *  torso_width / torso_height

def get_theta(bodyKeypoints):
    cos = min(max(get_pose_score(bodyKeypoints), -1), 1)
    return np.arccos(cos)

def get_pose(bodyKeypoints, n=2):
    """
    Get pose orientation of image
    Args:
        - name: file name
        - n: number of possible pose orientations (0=front, `n-1`=back)
    """
    theta = get_theta(bodyKeypoints)
    for i in range(1, n):
        if theta < i * np.pi / n:
            return i
    # back
    return n - 1
