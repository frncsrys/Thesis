#!/usr/bin/env python3
"""
pose_simulator.py  —  ROS 1
Simulates vehicle movement for Waze-style RViz navigation WITHOUT real hardware.

Replaces ORB-SLAM3 as the pose source. Integrates /cmd_vel Twist commands
(published by pure_pursuit) into a 2-D (x, y, yaw) pose using simple Euler
integration, then broadcasts:
  - TF:  world  →  camera_link   (consumed by astar_planner + pure_pursuit)
  - nav_msgs/Odometry on /orb_slam3/pose  (consumed by waypoint_manager)
  - geometry_msgs/PoseStamped on /sim/pose  (handy for RViz PoseStamped display)

Usage
-----
Set the starting pose via ROS params or just click "2D Pose Estimate" in RViz
(subscribe to /initialpose and reset the sim pose on the fly).

RViz checklist for Waze-style display
--------------------------------------
  Display type        Topic               Notes
  ─────────────────────────────────────────────────────────────────────
  Map                 /map                OccupancyGrid from astar_planner
  Path                /global_path        nav_msgs/Path — the planned route
  Pose                /sim/pose           Shows car arrow on the map
  Marker              /pp/markers         Lookahead sphere + line (pure_pursuit)
  Marker              /pp/text            Speed / HUD overlay
  TF                                      Enable to see world→camera_link

Set the goal by adding a "2D Nav Goal" tool that publishes to /goal_pose.
"""

import math
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PoseSimulator:
    def __init__(self):
        rospy.init_node('pose_simulator', anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.parent_frame   = rospy.get_param('~parent_frame',   'world')
        self.base_frame     = rospy.get_param('~base_frame',     'camera_link')
        self.cmd_vel_topic  = rospy.get_param('~cmd_vel_topic',  '/cmd_vel')
        self.odom_topic     = rospy.get_param('~odom_topic',     '/orb_slam3/pose')
        self.pose_topic     = rospy.get_param('~pose_topic',     '/sim/pose')
        self.update_rate    = float(rospy.get_param('~update_rate', 50.0))   # Hz

        # Starting pose (override with "2D Pose Estimate" in RViz)
        self.x   = float(rospy.get_param('~start_x',   0.0))
        self.y   = float(rospy.get_param('~start_y',   0.0))
        self.yaw = float(rospy.get_param('~start_yaw', 0.0))

        # Velocity caps — prevent sim from running away on bad commands
        self.v_max = float(rospy.get_param('~v_max', 2.0))   # m/s
        self.w_max = float(rospy.get_param('~w_max', 4.0))   # rad/s

        # ── State ─────────────────────────────────────────────────────────────
        self.v = 0.0   # current linear  velocity command  (m/s)
        self.w = 0.0   # current angular velocity command  (rad/s)
        self.last_cmd_time = rospy.Time(0)
        self.cmd_timeout   = rospy.Duration(0.5)   # zero vel if no cmd for 0.5 s

        # ── TF broadcaster ────────────────────────────────────────────────────
        self.tf_broadcaster = tf.TransformBroadcaster()

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_odom = rospy.Publisher(self.odom_topic, Odometry,      queue_size=10)
        self.pub_pose = rospy.Publisher(self.pose_topic, PoseStamped,   queue_size=10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_cmd  = rospy.Subscriber(self.cmd_vel_topic, Twist,
                                         self._cmd_cb, queue_size=10)

        # RViz "2D Pose Estimate" publishes PoseWithCovarianceStamped to /initialpose
        self.sub_init = rospy.Subscriber(
            '/initialpose', PoseWithCovarianceStamped,
            self._init_pose_cov_cb, queue_size=1
        )

        # ── Integration loop ──────────────────────────────────────────────────
        dt = 1.0 / self.update_rate
        self._last_time = rospy.Time.now()
        rospy.Timer(rospy.Duration(dt), self._update_cb)

        rospy.loginfo(
            f"[pose_simulator] start=({self.x:.2f}, {self.y:.2f}, "
            f"yaw={math.degrees(self.yaw):.1f}°) | "
            f"rate={self.update_rate} Hz | "
            f"frames: {self.parent_frame} → {self.base_frame}"
        )
        rospy.loginfo(
            "[pose_simulator] Click '2D Pose Estimate' in RViz to teleport the car."
        )

    # ── Command callback ──────────────────────────────────────────────────────
    def _cmd_cb(self, msg: Twist):
        self.v = clamp(msg.linear.x,  -self.v_max, self.v_max)
        self.w = clamp(msg.angular.z, -self.w_max, self.w_max)
        self.last_cmd_time = rospy.Time.now()

    # ── Pose reset callback (PoseWithCovarianceStamped from RViz) ────────────
    def _init_pose_cov_cb(self, msg: PoseWithCovarianceStamped):
        """Called by RViz '2D Pose Estimate' button."""
        p = msg.pose.pose
        self.x   = p.position.x
        self.y   = p.position.y
        self.yaw = _yaw_from_quat(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w
        )
        self.v = self.w = 0.0
        rospy.loginfo(
            f"[pose_simulator] Pose reset to "
            f"({self.x:.3f}, {self.y:.3f}, {math.degrees(self.yaw):.1f}°)"
        )

    # ── Integration + publishing ──────────────────────────────────────────────
    def _update_cb(self, _event):
        now = rospy.Time.now()
        dt  = (now - self._last_time).to_sec()
        self._last_time = now

        # Stop if no cmd_vel received recently (timeout safety)
        if (now - self.last_cmd_time) > self.cmd_timeout:
            self.v = 0.0
            self.w = 0.0

        # ── Euler integration (unicycle model) ────────────────────────────────
        self.x   += self.v * math.cos(self.yaw) * dt
        self.y   += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt
        # Normalise yaw to (-π, π]
        self.yaw  = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        # ── Build quaternion ───────────────────────────────────────────────────
        q = quaternion_from_euler(0.0, 0.0, self.yaw)

        # ── Broadcast TF: world → camera_link ─────────────────────────────────
        self.tf_broadcaster.sendTransform(
            (self.x, self.y, 0.0),
            q,
            now,
            self.base_frame,
            self.parent_frame,
        )

        # ── Publish Odometry (/orb_slam3/pose) ────────────────────────────────
        odom                         = Odometry()
        odom.header.stamp            = now
        odom.header.frame_id         = self.parent_frame
        odom.child_frame_id          = self.base_frame
        odom.pose.pose.position.x    = self.x
        odom.pose.pose.position.y    = self.y
        odom.pose.pose.position.z    = 0.0
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        odom.twist.twist.linear.x    = self.v
        odom.twist.twist.angular.z   = self.w
        self.pub_odom.publish(odom)

        # ── Publish PoseStamped (/sim/pose — for RViz arrow display) ──────────
        ps                         = PoseStamped()
        ps.header                  = odom.header
        ps.pose                    = odom.pose.pose
        self.pub_pose.publish(ps)

    def spin(self):
        rospy.spin()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


if __name__ == '__main__':
    node = PoseSimulator()
    node.spin()