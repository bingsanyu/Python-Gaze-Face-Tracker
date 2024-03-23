
from head_pose_estimator import HeadPoseEstimator
from eye_tracker import EyeTracker
from utils import initialize_camera, initialize_socket, Logger
from constants import *
from AngleBuffer import AngleBuffer
from face_mesh_detector import FaceMeshDetector
import cv2 as cv
import numpy as np
from constants import *


def main():
    IS_RECORDING = True
    EYES_BLINK_FRAME_COUNTER = 0
    initial_pitch, initial_yaw, initial_roll = None, None, None

    cap = initialize_camera()
    logger = Logger()
    iris_socket = initialize_socket()
    eye_tracker = EyeTracker()
    head_pose_estimator = HeadPoseEstimator()
    face_mesh_detector = FaceMeshDetector(
        MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
    )
    try:
        angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            face_mesh_detector.detect_face_mesh(rgb_frame)
            mesh_points, mesh_points_3D = face_mesh_detector.mesh_points, face_mesh_detector.mesh_points_3D
            # Process eye features
            if mesh_points is not None:
                eyes_aspect_ratio = eye_tracker.blinking_ratio(mesh_points_3D)
                if eyes_aspect_ratio <= BLINK_THRESHOLD:
                    EYES_BLINK_FRAME_COUNTER += 1
                else:
                    if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINKS += 1
                    EYES_BLINK_FRAME_COUNTER = 0
                if ENABLE_HEAD_POSE:
                    pitch, yaw, roll = head_pose_estimator.estimate_head_pose(mesh_points, (img_h, img_w))
                    angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = angle_buffer.get_average()
                    if initial_pitch is None or (key == ord('c') and calibrated):
                        initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                        calibrated = True
                        if PRINT_DATA:
                            print("Head pose recalibrated.")
                    if calibrated:
                        pitch -= initial_pitch
                        yaw -= initial_yaw
                        roll -= initial_roll
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")


                if LOG_DATA:
                    l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy = face_mesh_detector.process_image(frame=frame)
                    logger.log_data(l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, mesh_points, pitch, yaw, roll)

                # Sending data through socket
                packet = np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32)
                iris_socket.sendto(bytes(packet), SERVER_ADDRESS)

            # Writing the on screen data on the frame
                if SHOW_ON_SCREEN_DATA:
                    if IS_RECORDING:
                        cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
                    cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    if ENABLE_HEAD_POSE:
                        cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                        cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                        cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            
            cv.imshow("Eye Tracking", frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord('c'):
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                if PRINT_DATA:
                    print("Head pose recalibrated.")
                    
            if key == ord('r'):
                IS_RECORDING = not IS_RECORDING
                if IS_RECORDING:
                    print("Recording started.")
                else:
                    print("Recording paused.")

            if key == ord('q'):
                if PRINT_DATA:
                    print("Exiting program...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()
        iris_socket.close()
        if PRINT_DATA:
            print("Program exited successfully.")
        if LOG_DATA and IS_RECORDING:
            logger.close_resources(cap, iris_socket)

if __name__ == "__main__":
    main()

"""
眼球追踪和头部姿态估计

特性：
- 实时眼球追踪，计算每帧的眼睛闭合比例和眨眼次数。
- 头部姿态估计，确定用户头部的方向，包括俯仰角，偏航角和滚动角。
- 校准功能，将初始头部姿态设为参考零位置。
- 数据记录，用于进一步分析和调试。

方法：
- 脚本使用MediaPipe的FaceMesh模型提供的468个面部标记。
- 通过计算每只眼睛的眼睛闭合比例(EAR)并基于EAR阈值检测眨眼来实现眼球追踪。
- 使用solvePnP算法和预定义的3D面部模型以及从摄像头获取的对应2D标记来估计头部姿态。
- 角度被标准化为直观的范围（俯仰角：[-90, 90]，偏航和滚动角：[-180, 180]）。

理论：
- EAR被用作一个简单而有效的眼睛闭合检测指标。
- 头部姿态角度是使用透视n点方法得出的，该方法根据其2D图像点和3D模型点估计物体的姿态。

参数：
你可以在代码中更改参数，如面部宽度，移动平均窗口，网络摄像头ID，终端输出，屏幕数据，日志详细信息等。

使用：
- 在安装了必要依赖的Python环境中运行脚本。脚本接受命令行参数进行摄像头源配置。
- 按'c'键重新校准头部姿态估计到当前方向。
- 按'r'键开始/停止记录。
- 按'q'键退出程序。
- 输出显示在一个带有实时反馈和注释的窗口中，并记录到CSV文件中以供进一步分析。

"""


# parser = argparse.ArgumentParser(description="Eye Tracking Application")
# parser.add_argument(
#     "-c", "--camSource", help="Source of camera", default=str(0)
# )
# args = parser.parse_args()

