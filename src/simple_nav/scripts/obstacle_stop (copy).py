#!/usr/bin/env python3
"""
obstacle_stop.py  —  ROS 1
Safety gate that sits between Pure Pursuit and the motors.

Data flow:
  pure_pursuit  →  /cmd_vel_raw  →  [this node]  →  /cmd_vel  →  motors
                                          ↑
                               /orb_slam3/all_points  (PointCloud2)
                               (published by ros_mono every tracked frame)

Logic:
  1. Transform every map-frame map point into the robot (base) frame.
  2. Keep only points inside a forward-facing "danger box":
       |y_robot| < lateral_half_width   (side clearance)
        0 < x_robot < stop_distance     (in front of robot only)
       |z_robot| < height_band          (ignore floor/ceiling artefacts)
  3. If ANY point survives → publish zero Twist (hard stop) on /cmd_vel.
  4. Otherwise → pass the incoming /cmd_vel_raw straight through.

A "clear" counter (clear_count_thresh) prevents flickering: the node
only resumes motion after N consecutive obstacle-free readings.
"""

import math
import numpy as np

import rospy
import tf

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class ObstacleStop:
    def __init__(self):
        rospy.init_node('obstacle_stop', anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.parent_frame        = rospy.get_param('~parent_frame',        'world')
        self.base_frame          = rospy.get_param('~base_frame',          'camera_link')
        self.cmd_vel_in_topic    = rospy.get_param('~cmd_vel_in_topic',    '/cmd_vel_raw')
        self.cmd_vel_out_topic   = rospy.get_param('~cmd_vel_out_topic',   '/cmd_vel')
        self.cloud_topic         = rospy.get_param('~cloud_topic',         '/orb_slam3/all_points')

        # Danger-box dimensions (robot frame: +x forward, +y left, +z up)
        self.stop_distance       = float(rospy.get_param('~stop_distance',       0.60))  # m forward
        self.lateral_half_width  = float(rospy.get_param('~lateral_half_width',  0.30))  # m each side
        self.height_band         = float(rospy.get_param('~height_band',         0.50))  # m |z| limit
        self.min_points_to_stop  = int(rospy.get_param('~min_points_to_stop',    3))     # avoid single outlier
        self.clear_count_thresh  = int(rospy.get_param('~clear_count_thresh',    5))     # frames to resume

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── State ─────────────────────────────────────────────────────────────
        self.obstacle_detected  = False
        self.clear_count        = 0
        self.latest_cmd         = None  # last Twist from pure pursuit

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.pub_cmd   = rospy.Publisher(self.cmd_vel_out_topic, Twist, queue_size=10)
        self.sub_cmd   = rospy.Subscriber(self.cmd_vel_in_topic,  Twist,        self._cmd_cb,   queue_size=10)
        self.sub_cloud = rospy.Subscriber(self.cloud_topic,        PointCloud2,  self._cloud_cb, queue_size=1)

        # 10 Hz watchdog — keep publishing stop while obstacle is present
        rospy.Timer(rospy.Duration(0.1), self._watchdog_cb)

        rospy.loginfo(
            f"[obstacle_stop] gate: {self.cmd_vel_in_topic} → {self.cmd_vel_out_topic} | "
            f"cloud: {self.cloud_topic} | "
            f"stop_dist={self.stop_distance} m  lateral={self.lateral_half_width} m"
        )

    # ── cmd_vel passthrough ───────────────────────────────────────────────────
    def _cmd_cb(self, msg: Twist):
        self.latest_cmd = msg
        if not self.obstacle_detected:
            self.pub_cmd.publish(msg)

    # ── Point cloud obstacle check ────────────────────────────────────────────
    def _cloud_cb(self, msg: PointCloud2):
        # Get robot pose in map frame → build transform map→robot
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.parent_frame, rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return  # no TF yet, can't check

        # Build 4×4 homogeneous transform: map → robot
        T = self._tf_to_matrix(trans, rot)

        danger_count = 0

        for pt in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            # pt is in parent (map) frame → transform to robot frame
            p_map  = np.array([pt[0], pt[1], pt[2], 1.0])
            p_base = T @ p_map

            xr, yr, zr = float(p_base[0]), float(p_base[1]), float(p_base[2])

            # Danger-box test  (x: forward, y: lateral, z: height)
            if (0.0 < xr < self.stop_distance and
                    abs(yr) < self.lateral_half_width and
                    abs(zr) < self.height_band):
                danger_count += 1
                if danger_count >= self.min_points_to_stop:
                    break

        if danger_count >= self.min_points_to_stop:
            if not self.obstacle_detected:
                rospy.logwarn(
                    f"[obstacle_stop] OBSTACLE — {danger_count} points "
                    f"within {self.stop_distance} m. Stopping."
                )
            self.obstacle_detected = True
            self.clear_count       = 0
            self.pub_cmd.publish(Twist())  # immediate zero velocity
        else:
            if self.obstacle_detected:
                self.clear_count += 1
                if self.clear_count >= self.clear_count_thresh:
                    self.obstacle_detected = False
                    self.clear_count       = 0
                    rospy.loginfo("[obstacle_stop] Path clear — resuming.")

    # ── Watchdog: keep publishing stop if blocked ─────────────────────────────
    def _watchdog_cb(self, _event):
        if self.obstacle_detected:
            self.pub_cmd.publish(Twist())

    # ── Helper: tf tuple → 4×4 numpy matrix ──────────────────────────────────
    @staticmethod
    def _tf_to_matrix(trans, rot) -> np.ndarray:
        """
        Build a 4x4 homogeneous transform from a tf (translation, quaternion) pair.
        Equivalent to tf.transformations.quaternion_matrix but avoids the import.
        """
        x, y, z, w = rot
        tx, ty, tz = trans

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        T        = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = ObstacleStop()
    node.spin()