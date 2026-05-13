#!/usr/bin/env python3
"""
Pure-Pursuit path follower — ROS 1 / rospy port of simple_nav/pure_pursuit.py
Subscribes : /global_path  (nav_msgs/Path)
Publishes  : /cmd_vel      (geometry_msgs/Twist)  — direct, no obstacle gate
             /<debug_ns>/lookahead_pose  (geometry_msgs/PoseStamped)
             /<debug_ns>/markers        (visualization_msgs/Marker)
             /<debug_ns>/text           (visualization_msgs/Marker)
TF source  : parent_frame -> base_frame  (provided by pose_simulator)
"""

import math
from typing import Optional, Tuple

import rospy
import tf

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PurePursuit:
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.path_topic    = rospy.get_param('~path_topic',    '/global_path')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        # Must match ros_mono.cc parent_frame / child_frame params
        self.parent_frame  = rospy.get_param('~parent_frame',  'world')
        self.base_frame    = rospy.get_param('~base_frame',    'camera_link')

        self.lookahead   = float(rospy.get_param('~lookahead',       0.6))
        self.v_max       = float(rospy.get_param('~v_max',           0.25))
        self.v_min       = float(rospy.get_param('~v_min',           0.05))
        self.w_max       = float(rospy.get_param('~w_max',           1.2))
        self.goal_tol    = float(rospy.get_param('~goal_tolerance',  0.20))
        self.wheelbase   = float(rospy.get_param('~wheelbase',       0.26))
        self.publish_debug = bool(rospy.get_param('~publish_debug',  True))
        self.debug_ns    = str(rospy.get_param('~debug_ns',          'pp'))

        # ── TF listener ───────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── State ─────────────────────────────────────────────────────────────
        self.path       = None
        self.target_idx = 0

        # ── Publishers / Subscribers ──────────────────────────────────────────
        self.sub_path = rospy.Subscriber(self.path_topic, Path, self.cb_path, queue_size=10)
        self.pub_cmd  = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        if self.publish_debug:
            ns = self.debug_ns
            self.pub_lookahead = rospy.Publisher(f'/{ns}/lookahead_pose', PoseStamped, queue_size=10)
            self.pub_markers   = rospy.Publisher(f'/{ns}/markers',        Marker,      queue_size=10)
            self.pub_text      = rospy.Publisher(f'/{ns}/text',           Marker,      queue_size=10)

        # 20 Hz control loop
        rospy.Timer(rospy.Duration(0.05), self._control_loop_cb)

        rospy.loginfo(
            f"[pure_pursuit] path={self.path_topic} cmd_vel={self.cmd_vel_topic} "
            f"TF: {self.parent_frame} -> {self.base_frame}"
        )

    # ── Path callback ─────────────────────────────────────────────────────────
    def cb_path(self, msg: Path):
        self.path       = msg
        self.target_idx = 0
        rospy.loginfo(f"[pure_pursuit] Received path with {len(msg.poses)} poses.")

    # ── TF helper ─────────────────────────────────────────────────────────────
    def get_robot_pose_2d(self) -> Optional[Tuple[float, float, float]]:
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.parent_frame, self.base_frame, rospy.Time(0)
            )
            x   = trans[0]
            y   = trans[1]
            yaw = yaw_from_quat(rot[0], rot[1], rot[2], rot[3])
            return float(x), float(y), float(yaw)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(2.0, "[pure_pursuit] No TF yet.")
            return None

    def stop(self):
        self.pub_cmd.publish(Twist())

    # ── Debug visualisation ──────────────────────────────────────────────────
    def _publish_debug(self, rx, ry, tx, ty, v, w, kappa, steer_deg, dist_goal):
        if not self.publish_debug:
            return

        now = rospy.Time.now()

        # Lookahead pose
        ps                     = PoseStamped()
        ps.header.stamp        = now
        ps.header.frame_id     = self.parent_frame
        ps.pose.position.x     = float(tx)
        ps.pose.position.y     = float(ty)
        ps.pose.position.z     = 0.0
        ps.pose.orientation.w  = 1.0
        self.pub_lookahead.publish(ps)

        # Line robot → target
        m             = Marker()
        m.header      = ps.header
        m.ns          = 'pp'
        m.id          = 1
        m.type        = Marker.LINE_STRIP
        m.action      = Marker.ADD
        m.scale.x     = 0.03
        m.color.a     = 1.0
        m.color.r     = 1.0
        m.color.g     = 0.2
        m.color.b     = 0.2
        m.points      = [
            Point(x=float(rx), y=float(ry), z=0.02),
            Point(x=float(tx), y=float(ty), z=0.02),
        ]
        self.pub_markers.publish(m)

        # Target sphere
        s                 = Marker()
        s.header          = ps.header
        s.ns              = 'pp'
        s.id              = 2
        s.type            = Marker.SPHERE
        s.action          = Marker.ADD
        s.pose.position.x = float(tx)
        s.pose.position.y = float(ty)
        s.pose.position.z = 0.04
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.12
        s.color.a = 1.0
        s.color.r = 0.2
        s.color.g = 1.0
        s.color.b = 0.2
        self.pub_markers.publish(s)

        # HUD text above robot
        t                 = Marker()
        t.header          = ps.header
        t.ns              = 'pp'
        t.id              = 3
        t.type            = Marker.TEXT_VIEW_FACING
        t.action          = Marker.ADD
        t.pose.position.x = float(rx)
        t.pose.position.y = float(ry)
        t.pose.position.z = 0.35
        t.pose.orientation.w = 1.0
        t.scale.z         = 0.18
        t.color.a = t.color.r = t.color.g = t.color.b = 1.0
        t.text = (
            f"Pure Pursuit\n"
            f"v = {v:.3f} m/s\n"
            f"w = {w:.3f} rad/s\n"
            f"kappa = {kappa:.3f} 1/m\n"
            f"steer ~= {steer_deg:.1f} deg\n"
            f"lookahead = {self.lookahead:.2f} m\n"
            f"dist_goal = {dist_goal:.2f} m\n"
            f"target_idx = {self.target_idx}"
        )
        self.pub_text.publish(t)

    # ── 20 Hz control loop ────────────────────────────────────────────────────
    def _control_loop_cb(self, _event):
        if self.path is None or len(self.path.poses) < 2:
            return

        pose = self.get_robot_pose_2d()
        if pose is None:
            return

        rx, ry, ryaw = pose
        poses        = self.path.poses

        gx        = poses[-1].pose.position.x
        gy        = poses[-1].pose.position.y
        dist_goal = math.hypot(gx - rx, gy - ry)

        if dist_goal < self.goal_tol:
            self.stop()
            rospy.loginfo_throttle(5.0, "[pure_pursuit] Goal reached.")
            return

        # Find lookahead target
        target = None
        for i in range(self.target_idx, len(poses)):
            px, py = poses[i].pose.position.x, poses[i].pose.position.y
            if math.hypot(px - rx, py - ry) >= self.lookahead:
                target          = (px, py)
                self.target_idx = i
                break

        if target is None:
            target          = (poses[-1].pose.position.x, poses[-1].pose.position.y)
            self.target_idx = len(poses) - 1

        tx, ty = target

        # Transform target into robot frame
        dx, dy  = tx - rx, ty - ry
        cos_y   = math.cos(-ryaw)
        sin_y   = math.sin(-ryaw)
        x_r     = dx * cos_y - dy * sin_y
        y_r     = dx * sin_y + dy * cos_y

        Ld      = max(self.lookahead, 1e-3)
        kappa   = 2.0 * y_r / (Ld * Ld)

        v       = self.v_max / (1.0 + 1.5 * abs(kappa))
        v       = clamp(v, self.v_min, self.v_max)
        w       = clamp(v * kappa, -self.w_max, self.w_max)

        steer     = math.atan(self.wheelbase * w / v) if abs(v) > 1e-3 else 0.0
        steer_deg = math.degrees(steer)

        cmd             = Twist()
        cmd.linear.x    = float(v)
        cmd.angular.z   = float(w)
        self.pub_cmd.publish(cmd)

        self._publish_debug(rx, ry, tx, ty, v, w, kappa, steer_deg, dist_goal)

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = PurePursuit()
    node.spin()