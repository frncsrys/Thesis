#!/usr/bin/env python3
"""
perception_node.py  —  ROS 1
ROS 1 wrapper around the AV-Robots distance/object detection pipeline.

Replaces the standalone USB-camera loop in system_updated.py with a ROS image
subscriber so the perception system fits cleanly into the ORB-SLAM3 nav stack.

Data flow (real robot):
  /camera/image_raw (sensor_msgs/Image)
        │
        ▼
  YOLOv8n  +  GPM  +  FSM  (from system_updated.py logic)
        │
        ├─→ /perception/nav_command   (std_msgs/String)  GO | SLOW | STOP | OVERTAKE
        ├─→ /perception/nearest_dist  (std_msgs/Float32) nearest obstacle distance (m)
        └─→ /perception/vis           (visualization_msgs/MarkerArray) RViz overlays

Parameters (set via launch file or rosparam):
  ~model_path        : path to NCNN model dir   (default: best_ncnn_model)
  ~calib_path        : path to calibration NPZ  (default: calibration_params.npz)
  ~camera_height_m   : camera height above ground  (default: 0.23)
  ~camera_tilt_deg   : camera tilt angle (deg)     (default: 12.0)
  ~image_topic       : input image topic (default: /camera/image_raw)
  ~target_width      : resize width before inference (default: 640)
  ~target_height     : resize height before inference (default: 360)
  ~skip_frames       : run YOLO every N frames (default: 1)
  ~publish_debug_img : publish annotated image on /perception/debug_image (default: false)

The FSM and all threshold constants are identical to system_updated.py so
behaviour is unchanged — only the I/O layer is swapped.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

import rospy
import cv_bridge

from sensor_msgs.msg   import Image
from std_msgs.msg      import String, Float32
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


# ──────────────────────────────────────────────────────────────────────────────
# FSM thresholds  (keep in sync with system_updated.py)
# ──────────────────────────────────────────────────────────────────────────────
D_STOP              = 0.52
D_SLOW              = 0.80
TTC_THRESHOLD       = 2.0
D_SECONDARY_BLOCK   = 0.90
TIME_OBSERVE        = 1.5
TIME_TURN           = 1.0
TIME_PASS           = 1.5
TIME_RETURN         = 1.0
MEDIAN_WINDOW       = 5
DIST_ALPHA          = 0.6
SPEED_ALPHA         = 0.3
BLIND_SPOT_TIMEOUT  = 2.0
STATIC_CONFIRM_FRAMES = 5
CAMERA_BUMPER_OFFSET_M = 0.16
MAX_STEERING_ANGLE   = 40.0


class State:
    FOLLOW       = "FOLLOW_GLOBAL_PATH"
    SLOW         = "LOCAL_SLOW"
    OBSERVE      = "LOCAL_AVOID_STATIC (OBSERVE)"
    DECIDE       = "LOCAL_AVOID_STATIC (DECIDE)"
    OVERTAKE     = "LOCAL_AVOID_STATIC (OVERTAKE)"
    REJOIN       = "REJOIN_PATH"
    STOP_DYNAMIC = "LOCAL_AVOID_DYNAMIC (STOP)"


# ──────────────────────────────────────────────────────────────────────────────
# Kalman filter  (identical to system_updated.py)
# ──────────────────────────────────────────────────────────────────────────────
class KalmanFilter1D:
    def __init__(self, initial_dist):
        self.x = np.array([[initial_dist], [0.0]])
        self.P = np.eye(2)
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])
        self.Q = np.eye(2) * 0.01

    def update_and_predict(self, measurement, dt):
        A      = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + self.Q
        innov  = measurement - self.H @ x_pred
        S      = self.H @ P_pred @ self.H.T + self.R
        K      = P_pred @ self.H.T * (1.0 / float(S[0, 0]))
        self.x = x_pred + K @ innov
        self.P = P_pred - K @ self.H @ P_pred
        return float(self.x[0][0]), float(self.x[1][0])


# ──────────────────────────────────────────────────────────────────────────────
# Object tracker  (identical to system_updated.py)
# ──────────────────────────────────────────────────────────────────────────────
class ObjectTracker:
    def __init__(self):
        self.tracks = {}

    def update(self, object_id, raw_distance, dt, current_fsm_state, fp_x):
        current_time = time.time()

        if object_id not in self.tracks:
            self.tracks[object_id] = {
                'history':        deque(maxlen=MEDIAN_WINDOW),
                'x_history':      deque(maxlen=MEDIAN_WINDOW),
                'exp_dist':       raw_distance,
                'exp_x':          fp_x,
                'speed':          0.0,
                'last_time':      current_time,
                'kalman':         KalmanFilter1D(raw_distance),
                'dynamic_history': deque([True] * STATIC_CONFIRM_FRAMES,
                                         maxlen=STATIC_CONFIRM_FRAMES),
            }
            return raw_distance, 0.0, float('inf'), 0.0, False

        track = self.tracks[object_id]

        track['history'].append(raw_distance)
        median_dist   = float(np.median(track['history']))
        exp_dist      = DIST_ALPHA * median_dist + (1 - DIST_ALPHA) * track['exp_dist']
        track['exp_dist'] = exp_dist

        final_dist, kalman_speed = track['kalman'].update_and_predict(exp_dist, dt)
        new_speed     = SPEED_ALPHA * kalman_speed + (1 - SPEED_ALPHA) * track['speed']
        closing_speed = -new_speed

        track['x_history'].append(fp_x)
        median_x     = float(np.median(track['x_history']))
        exp_x        = DIST_ALPHA * median_x + (1 - DIST_ALPHA) * track['exp_x']
        lat_speed_px = abs(exp_x - track['exp_x']) / dt if dt > 0 else 0.0
        track['exp_x'] = exp_x

        ttc = float('inf')
        if closing_speed > 0.01:
            ttc = final_dist / closing_speed

        raw_dynamic = (
            lat_speed_px > 40.0
            or (current_fsm_state in [State.OBSERVE, State.STOP_DYNAMIC]
                and closing_speed > 0.03)
        )
        track['dynamic_history'].append(raw_dynamic)
        confirmed_dynamic = any(track['dynamic_history'])

        track['speed']     = new_speed
        track['last_time'] = current_time

        return final_dist, closing_speed, ttc, lat_speed_px, confirmed_dynamic


# ──────────────────────────────────────────────────────────────────────────────
# GPM distance helper  (identical to system_updated.py)
# ──────────────────────────────────────────────────────────────────────────────
def calculate_calibrated_distance(fp_x, fp_y, h, tilt_deg,
                                   cam_mat, dist_coeffs, cx, cy, fy):
    pt  = np.array([[[float(fp_x), float(fp_y)]]], dtype=np.float32)
    upt = cv2.undistortPoints(pt, cam_mat, dist_coeffs, P=cam_mat)
    v   = upt[0][0][1]
    ang = math.radians(tilt_deg) + math.atan((v - cy) / fy)
    return h / math.tan(ang) if ang > 0 else float('inf')


# ──────────────────────────────────────────────────────────────────────────────
# Main ROS node
# ──────────────────────────────────────────────────────────────────────────────
class PerceptionNode:

    def __init__(self):
        rospy.init_node('perception_node', anonymous=False)

        # ── ROS parameters ────────────────────────────────────────────────────
        model_path   = rospy.get_param('~model_path',        'best_ncnn_model')
        calib_path   = rospy.get_param('~calib_path',        'calibration_params.npz')
        self.cam_h   = float(rospy.get_param('~camera_height_m',  0.23))
        self.cam_tilt= float(rospy.get_param('~camera_tilt_deg',  12.0))
        img_topic    = rospy.get_param('~image_topic',        '/camera/image_raw')
        self.tw      = int(rospy.get_param('~target_width',   640))
        self.th      = int(rospy.get_param('~target_height',  360))
        self.skip    = int(rospy.get_param('~skip_frames',    1))
        self.pub_dbg = bool(rospy.get_param('~publish_debug_img', False))

        # ── Load calibration ──────────────────────────────────────────────────
        try:
            cal = np.load(calib_path)
            self.cam_mat    = cal['camera_matrix'].copy()
            self.dist_coeff = cal['dist_coeffs']
            # Scale camera matrix from calib resolution (1280×720) to target
            sx = self.tw / 1280.0
            sy = self.th / 720.0
            self.cam_mat[0, 0] *= sx;  self.cam_mat[0, 2] *= sx
            self.cam_mat[1, 1] *= sy;  self.cam_mat[1, 2] *= sy
            self.CX = float(self.cam_mat[0, 2])
            self.CY = float(self.cam_mat[1, 2])
            self.FX = float(self.cam_mat[0, 0])
            self.FY = float(self.cam_mat[1, 1])
            rospy.loginfo("[perception] Calibration loaded from %s", calib_path)
        except Exception as e:
            rospy.logerr("[perception] Cannot load calibration: %s", e)
            raise

        # ── Load YOLO model ───────────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            rospy.loginfo("[perception] YOLO model loaded from %s", model_path)
        except Exception as e:
            rospy.logerr("[perception] Cannot load YOLO model: %s", e)
            raise

        # ── Internal state ────────────────────────────────────────────────────
        self.tracker              = ObjectTracker()
        self.active_trackers      = {}
        self.current_state        = State.FOLLOW
        self.state_start_time     = 0.0
        self.overtake_direction   = "NONE"
        self.overtake_angle       = 0.0
        self.last_obstacle_seen   = time.time()
        self.frame_count          = 0
        self.prev_time            = time.time()
        self.bridge               = cv_bridge.CvBridge()

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cmd  = rospy.Publisher('/perception/nav_command',  String,  queue_size=1)
        self.pub_dist = rospy.Publisher('/perception/nearest_dist', Float32, queue_size=1)
        self.pub_vis  = rospy.Publisher('/perception/vis', MarkerArray, queue_size=1)
        if self.pub_dbg:
            self.pub_img = rospy.Publisher('/perception/debug_image', Image, queue_size=1)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.sub_img = rospy.Subscriber(img_topic, Image, self._image_cb, queue_size=1,
                                        buff_size=2**24)

        rospy.loginfo(
            "[perception] Ready — listening on %s | "
            "target=%dx%d | skip=%d | debug_img=%s",
            img_topic, self.tw, self.th, self.skip, self.pub_dbg
        )

    # ── Image callback ────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr_throttle(5.0, "[perception] cv_bridge error: %s", e)
            return

        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 1.0 / 30.0
        self.prev_time = current_time
        self.frame_count += 1

        # Resize to target resolution for inference
        frame = cv2.resize(raw, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)

        all_detections = []

        # ── YOLO detection ────────────────────────────────────────────────────
        if self.frame_count % self.skip == 0 or self.frame_count == 1:
            results = self.model.track(frame, persist=True, stream=True, verbose=False)
            self.active_trackers.clear()

            for r in results:
                if r.boxes.id is None:
                    continue
                track_ids = r.boxes.id.int().cpu().tolist()
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    oid        = track_ids[i]
                    label_name = self.model.names[int(box.cls[0])]
                    fp_x       = int((x1 + x2) / 2)
                    fp_y       = int(y2)

                    raw_dist = calculate_calibrated_distance(
                        fp_x, fp_y, self.cam_h, self.cam_tilt,
                        self.cam_mat, self.dist_coeff,
                        self.CX, self.CY, self.FY)

                    if raw_dist != float('inf'):
                        dist, closing, ttc, lat_spd, is_dyn = self.tracker.update(
                            oid, raw_dist, dt, self.current_state, fp_x)
                        all_detections.append({
                            'id': oid, 'label': label_name,
                            'box': (x1, y1, x2, y2),
                            'fp': (fp_x, fp_y),
                            'dist': dist, 'speed': closing, 'ttc': ttc,
                            'lat_speed': lat_spd, 'center_x': fp_x,
                            'is_dynamic': is_dyn
                        })

                        # Keep KCF tracker initialised if using skip>1
                        if self.skip > 1:
                            bbox = (x1, y1, x2 - x1, y2 - y1)
                            try:
                                kcf = cv2.TrackerKCF.create()
                            except AttributeError:
                                try:
                                    kcf = cv2.legacy.TrackerKCF_create()
                                except AttributeError:
                                    kcf = cv2.TrackerKCF_create()
                            kcf.init(frame, bbox)
                            self.active_trackers[oid] = {'tracker': kcf, 'label': label_name}
        else:
            # KCF gap frames
            for oid, info in list(self.active_trackers.items()):
                ok, bbox = info['tracker'].update(frame)
                if not ok:
                    del self.active_trackers[oid]
                    continue
                x, y, wb, hb   = map(int, bbox)
                x1, y1, x2, y2 = x, y, x + wb, y + hb
                fp_x = int((x1 + x2) / 2);  fp_y = int(y2)
                raw_dist = calculate_calibrated_distance(
                    fp_x, fp_y, self.cam_h, self.cam_tilt,
                    self.cam_mat, self.dist_coeff,
                    self.CX, self.CY, self.FY)
                if raw_dist != float('inf'):
                    dist, closing, ttc, lat_spd, is_dyn = self.tracker.update(
                        oid, raw_dist, dt, self.current_state, fp_x)
                    all_detections.append({
                        'id': oid, 'label': info['label'],
                        'box': (x1, y1, x2, y2),
                        'fp': (fp_x, fp_y),
                        'dist': dist, 'speed': closing, 'ttc': ttc,
                        'lat_speed': lat_spd, 'center_x': fp_x,
                        'is_dynamic': is_dyn
                    })

        # ── Find critical (nearest) obstacle ──────────────────────────────────
        critical = None
        min_dist = float('inf')
        for d in all_detections:
            if d['dist'] < min_dist:
                min_dist = d['dist']
                critical = d
        if critical is not None:
            self.last_obstacle_seen = current_time

        # ── 3-tier FSM  (identical logic to system_updated.py) ────────────────
        global_action = self._run_fsm(critical, all_detections, current_time)

        # ── Publish ROS outputs ───────────────────────────────────────────────
        self.pub_cmd.publish(String(data=global_action))
        self.pub_dist.publish(Float32(data=float(min_dist) if critical else float('inf')))
        self._publish_markers(all_detections, critical, global_action, msg.header.stamp)

        if self.pub_dbg:
            self._publish_debug_image(frame, all_detections, critical, global_action, msg)

    # ── FSM (ported 1-to-1 from system_updated.py) ───────────────────────────
    def _run_fsm(self, critical, all_detections, current_time) -> str:
        global_action = "GO"

        if self.current_state == State.OVERTAKE:
            elapsed = time.time() - self.state_start_time
            global_action = "OVERTAKE"
            if elapsed >= TIME_TURN + TIME_PASS + TIME_RETURN:
                self.current_state    = State.REJOIN
                self.state_start_time = time.time()

        elif self.current_state == State.REJOIN:
            global_action = "OVERTAKE"
            if time.time() - self.state_start_time > 1.0:
                self.current_state = State.FOLLOW

        elif critical:
            dist       = critical['dist']
            ttc        = critical['ttc']
            closing    = critical['speed']
            is_dynamic = critical['is_dynamic']

            if self.current_state in [State.FOLLOW, State.SLOW]:
                if ttc <= TTC_THRESHOLD or dist <= D_STOP:
                    self.current_state    = State.OBSERVE
                    self.state_start_time = time.time()
                    global_action = "STOP"
                elif dist <= D_SLOW:
                    self.current_state = State.SLOW
                    global_action      = "SLOW"

            elif self.current_state == State.OBSERVE:
                global_action = "STOP"
                if is_dynamic:
                    self.current_state = State.STOP_DYNAMIC
                elif time.time() - self.state_start_time > TIME_OBSERVE:
                    self.current_state    = State.DECIDE
                    self.state_start_time = time.time()

            elif self.current_state == State.STOP_DYNAMIC:
                global_action = "STOP"
                if not is_dynamic and closing < 0.02:
                    self.current_state = State.FOLLOW

            elif self.current_state == State.DECIDE:
                global_action  = "STOP"
                any_dynamic    = any(d['is_dynamic'] for d in all_detections)
                if any_dynamic:
                    self.current_state = State.STOP_DYNAMIC
                else:
                    secondary_too_close = any(
                        d['id'] != critical['id'] and d['dist'] <= D_SECONDARY_BLOCK
                        for d in all_detections
                    )
                    if not secondary_too_close:
                        self.overtake_direction = "LEFT" if critical['center_x'] > self.CX else "RIGHT"
                        x1, y1, x2, y2 = critical['box']
                        obs_w  = (abs(x2 - x1) * critical['dist']) / self.FX
                        lat_off = (obs_w / 2) + 0.20
                        bumper  = max(0.01, critical['dist'] - CAMERA_BUMPER_OFFSET_M)
                        self.overtake_angle   = min(
                            math.degrees(math.atan2(lat_off, bumper)),
                            MAX_STEERING_ANGLE)
                        self.current_state    = State.OVERTAKE
                        self.state_start_time = time.time()
        else:
            # No obstacle visible
            if self.current_state in [State.STOP_DYNAMIC, State.OBSERVE]:
                self.current_state = State.FOLLOW
            elif self.current_state == State.DECIDE:
                if current_time - self.last_obstacle_seen > BLIND_SPOT_TIMEOUT:
                    self.current_state = State.FOLLOW

        rospy.logdebug("[perception] state=%s  action=%s",
                       self.current_state, global_action)
        return global_action

    # ── RViz markers ─────────────────────────────────────────────────────────
    def _publish_markers(self, detections, critical, action, stamp):
        ma    = MarkerArray()
        now   = stamp
        frame = 'camera_link'

        ACTION_COLORS = {
            'GO':       (0.0, 1.0, 0.0),
            'SLOW':     (0.0, 1.0, 1.0),
            'STOP':     (1.0, 0.0, 0.0),
            'OVERTAKE': (1.0, 0.65, 0.0),
        }
        r, g, b = ACTION_COLORS.get(action, (1.0, 1.0, 1.0))

        # One sphere per detected obstacle (in camera frame — approximated)
        for i, d in enumerate(detections):
            m             = Marker()
            m.header.stamp    = now
            m.header.frame_id = frame
            m.ns          = 'perception'
            m.id          = i
            m.type        = Marker.SPHERE
            m.action      = Marker.ADD
            m.pose.position.x = float(d['dist'])   # forward in camera frame
            m.pose.position.y = 0.0
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.a = 0.8
            m.color.r = r;  m.color.g = g;  m.color.b = b
            ma.markers.append(m)

        # HUD text marker above the robot
        t                 = Marker()
        t.header.stamp    = now
        t.header.frame_id = frame
        t.ns              = 'perception'
        t.id              = 100
        t.type            = Marker.TEXT_VIEW_FACING
        t.action          = Marker.ADD
        t.pose.position.x = 0.0
        t.pose.position.y = 0.0
        t.pose.position.z = 0.6
        t.pose.orientation.w = 1.0
        t.scale.z         = 0.18
        t.color.a = t.color.r = t.color.g = t.color.b = 1.0
        nearest = f"{critical['dist']*100:.1f}cm" if critical else "—"
        t.text = (
            f"Perception: {action}\n"
            f"State: {self.current_state}\n"
            f"Nearest: {nearest}\n"
            f"Objects: {len(detections)}"
        )
        ma.markers.append(t)
        self.pub_vis.publish(ma)

    # ── Optional debug image ──────────────────────────────────────────────────
    @staticmethod
    def _outlined_text(img, text, pos, scale, color):
        """Black-outlined text — same as draw_outlined_text() in system_updated.py."""
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   2)

    def _publish_debug_image(self, frame, detections, critical, action, orig_msg):
        ACTION_COLORS = {
            'GO':       (0, 255, 0),
            'SLOW':     (0, 255, 255),
            'STOP':     (0, 0, 255),
            'OVERTAKE': (0, 165, 255),
        }
        hud_color = ACTION_COLORS.get(action, (255, 255, 255))
        out = frame.copy()

        # ── Build detailed action text (matches system_updated.py) ────────────
        if self.current_state == State.OVERTAKE:
            elapsed = time.time() - self.state_start_time
            if elapsed < TIME_TURN:
                action_text = f"TURN {self.overtake_direction} {self.overtake_angle:.1f} DEG"
            elif elapsed < TIME_TURN + TIME_PASS:
                action_text = "DRIVE STRAIGHT"
            else:
                return_dir  = "RIGHT" if self.overtake_direction == "LEFT" else "LEFT"
                action_text = f"RETURN {return_dir} {self.overtake_angle:.1f} DEG"
        elif self.current_state == State.REJOIN:
            action_text = "REALIGNING"
        elif self.current_state == State.OBSERVE:
            action_text = "WAITING TO CONFIRM STATIC"
        elif self.current_state == State.STOP_DYNAMIC:
            action_text = "STOP (DYNAMIC THREAT)"
        elif self.current_state == State.DECIDE:
            action_text = "DECIDING"
        elif action == "SLOW":
            action_text = "SLOW DOWN"
        elif action == "STOP":
            action_text = "BRAKING TO OBSERVE"
        else:
            action_text = action   # GO

        # ── Closing speed for HUD ─────────────────────────────────────────────
        show_speed         = False
        display_speed_cm_s = 0.0
        if critical:
            closing = critical.get('speed', 0.0)
            if closing > 0:
                show_speed         = True
                display_speed_cm_s = closing * 100.0

        # ── Per-detection boxes ───────────────────────────────────────────────
        for d in detections:
            x1, y1, x2, y2 = d['box']
            is_crit   = critical and d['id'] == critical['id']
            col       = hud_color if is_crit else (200, 200, 200)
            state_str = "DYN" if d['is_dynamic'] else "STAT"
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.circle(out, d['fp'], 5, (0, 0, 255), -1)
            ttc_str = f"{d['ttc']:.1f}s" if d['ttc'] != float('inf') else "inf"
            self._outlined_text(
                out,
                f"{d['label']} {d['dist']*100:.1f}cm | TTC:{ttc_str} | {state_str}",
                (x1, max(y1 - 5, 10)), 0.4, col
            )

        # ── Main HUD overlay (matches system_updated.py layout) ───────────────
        self._outlined_text(out, "LIVE DIORAMA TEST",          (10, 20),  0.6, (255, 255, 255))
        self._outlined_text(out, f"STATE: {self.current_state}",(10, 45), 0.6, (255, 255, 255))
        self._outlined_text(out, action_text,                   (10, 70), 0.6, hud_color)
        if show_speed:
            self._outlined_text(out,
                                f"Closing Speed: {display_speed_cm_s:.1f} cm/s",
                                (10, 95), 0.5, (0, 255, 255))

        try:
            dbg_msg        = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
            dbg_msg.header = orig_msg.header
            self.pub_img.publish(dbg_msg)
        except cv_bridge.CvBridgeError:
            pass

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = PerceptionNode()
    node.spin()