

# Face Selected points indices for Head Pose Estimation
indices_pose = [1, 33, 61, 199, 263, 291]

# 用户颧骨外缘之间的水平距离（以毫米为单位）。此测量用于缩放头部姿态估计的3D模型点 测量您的面宽并相应调整该值。
USER_FACE_WIDTH = 140  # [mm]

IS_RECORDING = False
PRINT_DATA = True
LOG_DATA = True
LOG_FOLDER = "logs"
SHOW_ALL_FEATURES = True
LOG_ALL_FEATURES = False
SHOW_ON_SCREEN_DATA = True
ENABLE_HEAD_POSE = True

# TOTAL_BLINKS: Counter for the total number of blinks detected.
TOTAL_BLINKS = 0 # Tracks the total number of blinks detected
# EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
EYES_BLINK_FRAME_COUNTER = 0
# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51
# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
calibrated = False

# Server
SERVER_IP = "127.0.0.1"
SERVER_PORT = 7070
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)