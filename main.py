import asyncio
import cv2
import mediapipe as mp
import numpy as np
import json
import websockets
import click
import time
from deepface import DeepFace
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R


def normalize_dict(d, fn):
    return {k: fn(v) for k, v in d.items()}


def remap_keys(d, key_map):
    return {key_map.get(k, k): v for k, v in d.items()}


def expand_emotion_keys(emotion, key_map):
    result = {}
    for k, v in emotion.items():
        if k in key_map and isinstance(key_map[k], list):
            for new_key in key_map[k]:
                result[new_key] = v  # Assign original value to each new key
        else:
            result[k] = v  # Keep original key-value pair if no mapping
    return result


def max_value_key(d):
    return max(d, key=d.get) if d else None


def estimate_head_quaternion_from_pose(results):
    if not results.pose_world_landmarks:
        return None

    landmarks = results.pose_world_landmarks.landmark

    left_shoulder = np.array(
        [
            landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].x,
            landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].y,
            landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER].z,
        ]
    )

    right_shoulder = np.array(
        [
            landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].x,
            landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].y,
            landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER].z,
        ]
    )

    nose = np.array(
        [
            landmarks[mp.solutions.holistic.PoseLandmark.NOSE].x,
            landmarks[mp.solutions.holistic.PoseLandmark.NOSE].y,
            landmarks[mp.solutions.holistic.PoseLandmark.NOSE].z,
        ]
    )

    shoulder_dir = right_shoulder - left_shoulder
    shoulder_dir /= np.linalg.norm(shoulder_dir)

    forward_dir = nose - (left_shoulder + right_shoulder) / 2
    forward_dir /= np.linalg.norm(forward_dir)

    up_dir = np.cross(shoulder_dir, forward_dir)
    up_dir /= np.linalg.norm(up_dir)

    # Construct rotation matrix (3x3)
    rot_mat = np.stack([shoulder_dir, up_dir, -forward_dir], axis=1)

    # Convert to quaternion
    rot = R.from_matrix(rot_mat)
    quat = rot.as_quat()  # returns [x, y, z, w]

    return {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quat[2]),
        "w": float(quat[3]),
    }


def extract_iris_movement(landmarks, image_width, image_height):
    def lm(idx):
        return np.array(
            [
                landmarks.landmark[idx].x * image_width,
                landmarks.landmark[idx].y * image_height,
            ]
        )

    # Define key points
    left_iris_center = np.mean([lm(i) for i in range(468, 473)], axis=0)
    right_iris_center = np.mean([lm(i) for i in range(473, 478)], axis=0)

    # Eye bounds (to normalize movement)
    left_eye_left = lm(33)
    left_eye_right = lm(133)
    left_eye_top = lm(159)
    left_eye_bottom = lm(145)

    right_eye_left = lm(362)
    right_eye_right = lm(263)
    right_eye_top = lm(386)
    right_eye_bottom = lm(374)

    def normalize(center, left, right, top, bottom):
        horizontal = np.clip((center[0] - left[0]) / (right[0] - left[0]), 0.0, 1.0)
        vertical = np.clip((center[1] - top[1]) / (bottom[1] - top[1]), 0.0, 1.0)
        # Return center-relative offset: [-1, 1]
        return {
            "x": (horizontal - 0.5) * 2.0,
            "y": (vertical - 0.5) * 2.0,
        }

    return {
        "left": normalize(
            left_iris_center,
            left_eye_left,
            left_eye_right,
            left_eye_top,
            left_eye_bottom,
        ),
        "right": normalize(
            right_iris_center,
            right_eye_left,
            right_eye_right,
            right_eye_top,
            right_eye_bottom,
        ),
    }


class ExpressionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = defaultdict(lambda: deque(maxlen=window_size))

    def smooth(self, current: dict) -> dict:
        smoothed = {}
        for key, value in current.items():
            self.history[key].append(value)
            smoothed[key] = sum(self.history[key]) / len(self.history[key])
        return smoothed


class FrameCounter:
    def __init__(self):
        self.count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self._show_frame_count = True
        self._show_fps = True

    def increment(self):
        self.count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.count / elapsed_time

    def get_count(self):
        return self.count

    def get_fps(self):
        return round(self.fps, 2)

    def show_frame_count(self, show: bool):
        self._show_frame_count = show

    def show_fps(self, show: bool):
        self._show_fps = show

    def draw(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1

        if self._show_frame_count:
            frame_text = f"Frame: {self.get_count()}"
            position_frame = (10, 20)  # Top-left for frame count
            cv2.putText(
                frame, frame_text, position_frame, font, font_scale, color, thickness
            )

        if self._show_fps:
            fps_text = f"FPS: {self.get_fps()}"
            position_fps = (10, 40)  # Below frame count for FPS
            cv2.putText(
                frame, fps_text, position_fps, font, font_scale, color, thickness
            )

        return frame


class HeadPoseComponent:
    def process(self, results, **kwargs):
        if not results.pose_world_landmarks:
            return None
        quat = estimate_head_quaternion_from_pose(results)
        return {"rotation": quat} if quat else None


class IrisMovementComponent:
    def process(self, results, frame_rgb, **kwargs):
        if not results.face_landmarks:
            return None
        h, w, _ = frame_rgb.shape
        iris_data = extract_iris_movement(results.face_landmarks, w, h)
        return {"iris": iris_data}


class ExpressionsComponent:
    def __init__(self, simple=False):
        self.expr_smoother = ExpressionSmoother(window_size=2)
        self.simple = simple

    def process(self, results, **kwargs):
        if not results.face_landmarks:
            return None
        if self.simple:
            expr_data = self.extract_expressions_simple(results.face_landmarks)
        else:
            expr_data = self.extract_expressions(results.face_landmarks)
        return {"blend": expr_data}

    def extract_expressions_simple(self, face_landmarks):
        def dist(a, b):
            return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

        def normalize_for_godot(value: float) -> float:
            return max(-1.0, min(1.0, (value * 2.0) - 1.0))

        expr = {}
        eye_left = 1.0 - min(
            1.0, dist(face_landmarks.landmark[159], face_landmarks.landmark[145]) * 30
        )
        eye_right = 1.0 - min(
            1.0, dist(face_landmarks.landmark[386], face_landmarks.landmark[374]) * 30
        )
        mouth_vertical = min(
            1.0, dist(face_landmarks.landmark[0], face_landmarks.landmark[17]) * 15
        )
        mouth_horizontal = min(
            1.0, dist(face_landmarks.landmark[61], face_landmarks.landmark[291]) * 5
        )

        expr["Fcl_EYE_Close_L"] = normalize_for_godot(eye_left)
        expr["Fcl_EYE_Close_R"] = normalize_for_godot(eye_right)
        expr["Fcl_MTH_A"] = normalize_for_godot(mouth_vertical)
        expr["Fcl_MTH_I"] = normalize_for_godot(mouth_horizontal)

        return self.expr_smoother.smooth(expr)

    def extract_expressions(self, face_landmarks):
        def dist(a, b):
            return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

        def normalize_for_godot(value: float) -> float:
            return max(-1.0, min(1.0, (value * 2.0) - 1.0))

        expr = {}

        # A - mouth vertical opening (jaw drop)
        mouth_vertical = min(
            1.0, dist(face_landmarks.landmark[0], face_landmarks.landmark[17]) * 15
        )
        expr["Fcl_MTH_A"] = normalize_for_godot(mouth_vertical)

        # I - mouth horizontal stretching (wide smile)
        mouth_horizontal = min(
            1.0, dist(face_landmarks.landmark[61], face_landmarks.landmark[291]) * 5
        )
        expr["Fcl_MTH_I"] = normalize_for_godot(mouth_horizontal)

        # U - lip protrusion/rounding (pucker)
        upper_lip_center = face_landmarks.landmark[13]
        lower_lip_center = face_landmarks.landmark[14]
        mouth_corners_dist = dist(
            face_landmarks.landmark[61], face_landmarks.landmark[291]
        )
        mouth_height = dist(upper_lip_center, lower_lip_center)
        lip_roundness = min(1.0, (mouth_height / (mouth_corners_dist + 0.001)) * 3)
        expr["Fcl_MTH_U"] = normalize_for_godot(lip_roundness)

        # E - mid-open mouth with slight horizontal stretch
        mouth_mid_opening = min(1.0, mouth_vertical * 0.6 + mouth_horizontal * 0.4)
        expr["Fcl_MTH_E"] = normalize_for_godot(mouth_mid_opening)

        # O - rounded mouth opening (similar to U but more open)
        mouth_area_ratio = min(
            1.0, (mouth_vertical * mouth_height) / (mouth_corners_dist + 0.001) * 2
        )
        expr["Fcl_MTH_O"] = normalize_for_godot(mouth_area_ratio)

        return self.expr_smoother.smooth(expr)


class EmotionDetectionComponent:
    def __init__(self):
        self.frame_skip = 3
        self.frame_count = 0
        self.last_emotion_result = None
        self.last_analysis_time = 0
        self.min_analysis_interval = 0.1

    def process(self, results, frame_bgr_flipped, **kwargs):
        face_landmarks = results.face_landmarks if results else None
        emotion_result = self.detect_emotions(frame_bgr_flipped, face_landmarks)
        if emotion_result:
            return {
                "emotion": emotion_result.get("emotion"),
                "dominant_emotion": emotion_result.get("dominant_emotion"),
            }
        return None

    def detect_emotions(self, frame, face_landmarks=None):
        """Optimized emotion detection using DeepFace with face cropping."""
        current_time = time.time()
        self.frame_count += 1

        # Skip analysis if not enough time has passed or not on a skip frame
        if (
            current_time - self.last_analysis_time < self.min_analysis_interval
            or self.frame_count % self.frame_skip != 0
        ):
            return self.last_emotion_result

        try:
            # Crop to face bounding box if landmarks are available
            if face_landmarks:
                # Calculate bounding box from face landmarks
                h, w, _ = frame.shape
                x_coords = [landmark.x for landmark in face_landmarks.landmark]
                y_coords = [landmark.y for landmark in face_landmarks.landmark]

                x1 = max(0, int(min(x_coords) * w) - 20)  # Add small padding
                y1 = max(0, int(min(y_coords) * h) - 20)
                x2 = min(w, int(max(x_coords) * w) + 20)
                y2 = min(h, int(max(y_coords) * h) + 20)

                # Crop the frame to the face region
                cropped_frame = frame[y1:y2, x1:x2]
                if cropped_frame.size == 0:  # If crop is invalid, use original frame
                    cropped_frame = frame
            else:
                cropped_frame = frame

            # Resize cropped frame for faster processing
            small_frame = cv2.resize(cropped_frame, (320, 240))
            # Analyze emotion
            result = DeepFace.analyze(
                small_frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )[0]

            emotion = normalize_dict(result.get("emotion", {}), lambda x: x / 100)
            emotion = expand_emotion_keys(
                emotion,
                {
                    "neutral": ["Fcl_BRW_Neutral", "Fcl_EYE_Neutral"],
                    "angry": ["Fcl_BRW_Angry", "Fcl_EYE_Angry"],
                    "disgust": ["Fcl_BRW_Angry", "Fcl_EYE_Angry"],
                    "happy": ["Fcl_BRW_Fun", "Fcl_EYE_Fun"],
                    "sad": ["Fcl_BRW_Sorrow", "Fcl_EYE_Sorrow"],
                    "fear": ["Fcl_BRW_Sorrow", "Fcl_EYE_Sorrow"],
                    "surprise": ["Fcl_BRW_Surprised", "Fcl_EYE_Surprised"],
                },
            )

            self.last_emotion_result = {
                "emotion": emotion,
                "dominant_emotion": max_value_key(emotion),
            }

            self.last_analysis_time = current_time
            return self.last_emotion_result
        except Exception:
            return self.last_emotion_result


class ShoulderTrackingComponent:
    def process(self, results, **kwargs):
        if not results.pose_world_landmarks:
            return None

        landmarks = results.pose_world_landmarks.landmark

        left_shoulder = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]

        shoulders_data = {
            "left": {
                "x": left_shoulder.x,
                "y": left_shoulder.y,
                "z": left_shoulder.z,
            },
            "right": {
                "x": right_shoulder.x,
                "y": right_shoulder.y,
                "z": right_shoulder.z,
            },
        }

        return {"shoulders": shoulders_data}


class MediaPipeClient:
    def __init__(self, camera_index, ws_uri, user_id, components):
        self.camera_index = camera_index
        self.ws_uri = ws_uri
        self.user_id = user_id
        self.components = components
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_counter = FrameCounter()

    async def run(self):
        print(f"Connecting to WebSocket server at {self.ws_uri}")
        async with websockets.connect(self.ws_uri) as websocket:
            with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=True,
            ) as holistic:
                while self.cap.isOpened():
                    self.frame_counter.increment()
                    success, frame = self.cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.flip(frame_rgb, 1)
                    frame_rgb.flags.writeable = False
                    results = holistic.process(frame_rgb)
                    frame_bgr_flipped = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    payload = {"id": self.user_id}

                    if results.face_landmarks or results.pose_world_landmarks:
                        for component in self.components:
                            component_data = component.process(
                                results=results,
                                frame_rgb=frame_rgb,
                                frame_bgr_flipped=frame_bgr_flipped,
                            )
                            if component_data:
                                payload.update(component_data)

                        if len(payload) > 1:
                            await websocket.send(json.dumps(payload))
                            print("Sent:", json.dumps(payload))

                    debug_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    self.frame_counter.draw(debug_frame)

                    if results.face_landmarks:
                        if results.pose_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                debug_frame,
                                results.pose_landmarks,
                                mp.solutions.holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                            )
                        cv2.imshow("MediaPipe Face", debug_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        self.cap.release()
        cv2.destroyAllWindows()


@click.command()
@click.option("--camera", default=0, help="Camera index to use (default: 0)")
@click.option(
    "--host", default="localhost", help="WebSocket server host (default: localhost)"
)
@click.option("--port", default=8082, help="WebSocket server port (default: 8081)")
@click.option("--user-id", required=True, help="User/Character ID to send to Godot")
@click.option(
    "--emotion-detection",
    is_flag=True,
    default=False,
    help="Enable emotion detection.",
)
def main(camera, host, port, user_id, emotion_detection):
    ws_uri = f"ws://{host}:{port}"

    components = [
        HeadPoseComponent(),
        IrisMovementComponent(),
        ExpressionsComponent(),
        # ShoulderTrackingComponent(),
    ]

    if emotion_detection:
        components.append(EmotionDetectionComponent())

    client = MediaPipeClient(
        camera_index=camera,
        ws_uri=ws_uri,
        user_id=user_id,
        components=components,
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
