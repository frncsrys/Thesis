#!/usr/bin/env python3
"""
obstacle_stop.py  —  ROS 1
Safety gate that sits between Pure Pursuit and the motors.

Data flow:
  pure_pursuit  →  /cmd_vel_raw  →  [this node]  →  /cmd_vel  →  motors
                                          ↑
                               /orb_slam3/all_points  (PointCloud2)
                               /perception/nav_command (std_msgs/String)

Two independent safety channels are combined here:

  Channel A — PointCloud2 danger-box (geometry, always active)
    1. Transform every map-frame point into the robot (base) frame.
    2. Keep only points inside a forward-facing "danger box":
           |y_robot| < lateral_half_width   (side clearance)
            0 < x_robot < stop_distance     (in front of robot only)
           |z_robot| < height_band          (ignore floor/ceiling artefacts)
    3. If ≥ min_points_to_stop survive → hard STOP.
    4. A "clear" counter (clear_count_thresh) prevents flickering.

  Channel B — Perception FSM command (semantic, from perception_node.py)
    Subscribes to /perception/nav_command (String: GO | SLOW | STOP | OVERTAKE)
    GO       → pass cmd_vel_raw through at full speed
    SLOW     → scale linear.x down to slow_speed_fraction of the incoming speed
    STOP     → zero velocity (same as Channel A hard stop)
    OVERTAKE → treated as GO (manoeuvre in progress, car is moving)

    If no perception message arrives within perception_timeout seconds the
    channel defaults to GO so a dead perception node never freezes the car.

Priority (highest → lowest):
  1. Channel A hard stop  (geometry never lies)
  2. Channel B STOP
  3. Channel B SLOW
  4. Channel B GO  (full pass-through)

Parameters (all have safe defaults so existing launch files keep working):
  ~parent_frame         : map/world frame id          (default: world)
  ~base_frame           : robot frame id              (default: camera_link)
  ~cmd_vel_in_topic     : input from pure_pursuit     (default: /cmd_vel_raw)
  ~cmd_vel_out_topic    : output to motors            (default: /cmd_vel)
  ~cloud_topic          : ORB-SLAM3 point cloud       (default: /orb_slam3/all_points)
  ~perception_topic     : FSM command from perception (default: /perception/nav_command)
  ~stop_distance        : danger-box depth (m)        (default: 0.60)
  ~lateral_half_width   : danger-box half-width (m)   (default: 0.30)
  ~height_band          : danger-box |z| limit (m)    (default: 0.50)
  ~min_points_to_stop   : outlier rejection count     (default: 3)
  ~clear_count_thresh   : frames before resuming      (default: 5)
  ~slow_speed_fraction  : fraction of v for SLOW      (default: 0.35)
  ~perception_timeout   : seconds before GO default   (default: 1.0)
"""

import math
import numpy as np

import rospy
import tf

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2


class ObstacleStop:
    def __init__(self):
        rospy.init_node('obstacle_stop', anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.parent_frame       = rospy.get_param('~parent_frame',       'world')
        self.base_frame         = rospy.get_param('~base_frame',         'camera_link')
        self.cmd_vel_in_topic   = rospy.get_param('~cmd_vel_in_topic',   '/cmd_vel_raw')
        self.cmd_vel_out_topic  = rospy.get_param('~cmd_vel_out_topic',  '/cmd_vel')
        self.cloud_topic        = rospy.get_param('~cloud_topic',        '/orb_slam3/all_points')
        self.perception_topic   = rospy.get_param('~perception_topic',   '/perception/nav_command')

        # Danger-box dimensions (robot frame: +x forward, +y left, +z up)
        self.stop_distance      = float(rospy.get_param('~stop_distance',      0.60))
        self.lateral_half_width = float(rospy.get_param('~lateral_half_width', 0.30))
        self.height_band        = float(rospy.get_param('~height_band',        0.50))
        self.min_points_to_stop = int(rospy.get_param(  '~min_points_to_stop', 3))
        self.clear_count_thresh = int(rospy.get_param(  '~clear_count_thresh', 5))

        # Perception channel settings
        # slow_speed_fraction: 0.35 means SLOW = 35% of whatever Pure Pursuit requested
        self.slow_speed_fraction = float(rospy.get_param('~slow_speed_fraction', 0.35))
        self.perception_timeout  = float(rospy.get_param('~perception_timeout',  1.0))

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── State ─────────────────────────────────────────────────────────────
        # Channel A (geometry)
        self.geo_obstacle       = False   # True = danger-box has points
        self.clear_count        = 0

        # Channel B (perception FSM)
        # Initialise to GO so the car moves immediately if perception is slow to start
        self.perception_cmd     = "GO"
        self.last_perception_time = rospy.Time(0)

        # Latest cmd_vel from Pure Pursuit
        self.latest_cmd         = None

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self.pub_cmd      = rospy.Publisher(self.cmd_vel_out_topic, Twist, queue_size=10)

        self.sub_cmd      = rospy.Subscriber(
            self.cmd_vel_in_topic, Twist, self._cmd_cb, queue_size=10)

        self.sub_cloud    = rospy.Subscriber(
            self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)

        self.sub_percept  = rospy.Subscriber(
            self.perception_topic, String, self._perception_cb, queue_size=10)

        # 10 Hz watchdog — keeps publishing stop/slow while blocked
        rospy.Timer(rospy.Duration(0.1), self._watchdog_cb)

        rospy.loginfo(
            "[obstacle_stop] gate: %s → %s | cloud: %s | perception: %s | "
            "stop_dist=%.2f m  lateral=%.2f m  slow_frac=%.2f  percept_timeout=%.1f s",
            self.cmd_vel_in_topic, self.cmd_vel_out_topic,
            self.cloud_topic, self.perception_topic,
            self.stop_distance, self.lateral_half_width,
            self.slow_speed_fraction, self.perception_timeout
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _perception_timed_out(self) -> bool:
        """True if no perception message has arrived within perception_timeout seconds."""
        if self.last_perception_time == rospy.Time(0):
            # Never received anything — check how long the node has been alive.
            # If we just started, give it a grace period before declaring timeout.
            return False
        elapsed = (rospy.Time.now() - self.last_perception_time).to_sec()
        return elapsed > self.perception_timeout

    def _effective_perception_cmd(self) -> str:
        """
        Return the perception command to act on, applying the timeout fallback.
        OVERTAKE is mapped to GO — the car is moving, no override needed.
        """
        if self._perception_timed_out():
            return "GO"
        cmd = self.perception_cmd
        if cmd == "OVERTAKE":
            return "GO"
        return cmd  # GO | SLOW | STOP

    def _apply_velocity_override(self, raw: Twist) -> Twist:
        """
        Build the output Twist by combining Channel A (geometry) and
        Channel B (perception FSM).

        Priority:
          1. geo_obstacle STOP  (always wins)
          2. perception  STOP
          3. perception  SLOW   (scale linear.x)
          4. GO               (full pass-through)
        """
        # Channel A hard stop
        if self.geo_obstacle:
            return Twist()  # zero velocity

        percept = self._effective_perception_cmd()

        if percept == "STOP":
            return Twist()

        if percept == "SLOW":
            out = Twist()
            out.linear.x  = raw.linear.x  * self.slow_speed_fraction
            out.angular.z = raw.angular.z  # keep steering unchanged
            return out

        # GO — full pass-through
        return raw

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist):
        """Receive cmd_vel_raw from Pure Pursuit and immediately forward it."""
        self.latest_cmd = msg
        out = self._apply_velocity_override(msg)
        self.pub_cmd.publish(out)

    def _perception_cb(self, msg: String):
        """Receive GO / SLOW / STOP / OVERTAKE from perception_node FSM."""
        cmd = msg.data.strip().upper()
        if cmd in ("GO", "SLOW", "STOP", "OVERTAKE"):
            if self.perception_cmd != cmd:
                rospy.loginfo("[obstacle_stop] Perception cmd: %s → %s", self.perception_cmd, cmd)
            self.perception_cmd      = cmd
            self.last_perception_time = rospy.Time.now()
        else:
            rospy.logwarn_throttle(
                2.0, "[obstacle_stop] Unknown perception command: '%s'", msg.data)

    def _cloud_cb(self, msg: PointCloud2):
        """Check PointCloud2 for points inside the forward danger box."""
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.base_frame, self.parent_frame, rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return  # TF not ready yet

        T = self._tf_to_matrix(trans, rot)

        danger_count = 0
        for pt in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            p_map  = np.array([pt[0], pt[1], pt[2], 1.0])
            p_base = T @ p_map
            xr, yr, zr = float(p_base[0]), float(p_base[1]), float(p_base[2])

            if (0.0 < xr < self.stop_distance
                    and abs(yr) < self.lateral_half_width
                    and abs(zr) < self.height_band):
                danger_count += 1
                if danger_count >= self.min_points_to_stop:
                    break

        if danger_count >= self.min_points_to_stop:
            if not self.geo_obstacle:
                rospy.logwarn(
                    "[obstacle_stop] GEOMETRY STOP — %d points within %.2f m.",
                    danger_count, self.stop_distance
                )
            self.geo_obstacle = True
            self.clear_count  = 0
            self.pub_cmd.publish(Twist())   # immediate hard stop

        else:
            if self.geo_obstacle:
                self.clear_count += 1
                if self.clear_count >= self.clear_count_thresh:
                    self.geo_obstacle = False
                    self.clear_count  = 0
                    rospy.loginfo("[obstacle_stop] Geometry clear — resuming.")

    def _watchdog_cb(self, _event):
        """
        10 Hz watchdog.
        If geo_obstacle is set, keep hammering zero velocity in case
        Pure Pursuit isn't sending new cmd_vel_raw messages.
        Also re-applies perception SLOW/STOP against the last known command
        in case the perception state changed between cmd_vel_raw messages.
        """
        if self.geo_obstacle:
            self.pub_cmd.publish(Twist())
            return

        # Re-apply perception override against latest command (handles state
        # transitions that occur between Pure Pursuit ticks).
        if self.latest_cmd is not None:
            percept = self._effective_perception_cmd()
            if percept in ("STOP", "SLOW"):
                out = self._apply_velocity_override(self.latest_cmd)
                self.pub_cmd.publish(out)

        # Log perception timeout once when it first fires
        if self._perception_timed_out() and self.perception_cmd != "GO":
            rospy.logwarn_throttle(
                5.0,
                "[obstacle_stop] Perception timeout (%.1f s) — defaulting to GO.",
                self.perception_timeout
            )

    # ── Static helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _tf_to_matrix(trans, rot) -> np.ndarray:
        """Build a 4×4 homogeneous transform from a tf (translation, quaternion) pair."""
        x, y, z, w = rot
        tx, ty, tz  = trans

        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        T         = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = ObstacleStop()
    node.spin()