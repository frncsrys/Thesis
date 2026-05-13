#!/usr/bin/env python3
"""
waypoint_manager.py  —  ROS 1
Loads an ordered list of (x, y) waypoints and drives the robot through them
one at a time. Supports two sending modes:

  use_move_base: false  →  publishes each waypoint to /goal_pose
                            (feeds our A* planner → Pure Pursuit stack)
  use_move_base: true   →  sends each waypoint as a MoveBaseAction goal
                            (uses the move_base nav stack instead)

Waypoints are defined as a ROS param list, e.g. in a launch file:
  <rosparam param="waypoints">
    [1.0, 0.5,   2.0, 1.5,   0.0, 0.0]   # flat list: x0,y0, x1,y1, ...
  </rosparam>

Or passed as a YAML file path via ~waypoints_file:
  waypoints:
    - [1.0, 0.5]
    - [2.0, 1.5]
    - [0.0, 0.0]

Topics published  : /goal_pose  (geometry_msgs/PoseStamped)  — when not using move_base
Topics subscribed : /orb_slam3/pose  (nav_msgs/Odometry)     — to detect waypoint arrival
Actionlib client  : move_base  (move_base_msgs/MoveBaseAction) — when use_move_base=true
"""

import math
import yaml
from typing import List, Tuple, Optional

import rospy
import actionlib
import tf

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

try:
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    MOVE_BASE_AVAILABLE = True
except ImportError:
    MOVE_BASE_AVAILABLE = False


def dist2d(ax, ay, bx, by) -> float:
    return math.hypot(bx - ax, by - ay)


class WaypointManager:
    def __init__(self):
        rospy.init_node('waypoint_manager', anonymous=False)

        # ── Parameters ────────────────────────────────────────────────────────
        self.parent_frame    = rospy.get_param('~parent_frame',    'world')
        self.base_frame      = rospy.get_param('~base_frame',      'camera_link')
        self.goal_topic      = rospy.get_param('~goal_topic',      '/goal_pose')
        self.odom_topic      = rospy.get_param('~odom_topic',      '/orb_slam3/pose')
        self.use_move_base   = bool(rospy.get_param('~use_move_base', False))
        self.goal_tolerance  = float(rospy.get_param('~goal_tolerance', 0.30))   # metres
        self.loop_waypoints  = bool(rospy.get_param('~loop_waypoints', False))
        self.waypoints_file  = rospy.get_param('~waypoints_file',  '')
        # Flat list param: [x0,y0, x1,y1, ...]
        raw_wps              = rospy.get_param('~waypoints', [])

        # ── Load waypoints ────────────────────────────────────────────────────
        self.waypoints: List[Tuple[float, float]] = []

        if self.waypoints_file:
            self.waypoints = self._load_yaml(self.waypoints_file)
        elif raw_wps:
            self.waypoints = self._parse_flat(raw_wps)

        if not self.waypoints:
            rospy.logwarn(
                "[waypoint_manager] No waypoints loaded. "
                "Set ~waypoints or ~waypoints_file param."
            )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_idx    = 0
        self.goal_sent      = False
        self.robot_x        = None
        self.robot_y        = None
        self.mission_done   = False

        # ── TF (fallback position source) ─────────────────────────────────────
        self.tf_listener = tf.TransformListener()

        # ── Publishers / Subscribers ──────────────────────────────────────────
        if not self.use_move_base:
            self.pub_goal = rospy.Publisher(
                self.goal_topic, PoseStamped, queue_size=1, latch=True
            )

        # Use odometry for position updates (fast) — TF as fallback
        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=10
        )

        # ── move_base client ──────────────────────────────────────────────────
        self.mb_client = None
        if self.use_move_base:
            if not MOVE_BASE_AVAILABLE:
                rospy.logerr(
                    "[waypoint_manager] use_move_base=true but "
                    "move_base_msgs is not installed."
                )
            else:
                self.mb_client = actionlib.SimpleActionClient(
                    'move_base', MoveBaseAction
                )
                rospy.loginfo("[waypoint_manager] Waiting for move_base action server…")
                self.mb_client.wait_for_server(timeout=rospy.Duration(10.0))
                rospy.loginfo("[waypoint_manager] move_base connected.")

        # ── 5 Hz supervisor loop ──────────────────────────────────────────────
        rospy.Timer(rospy.Duration(0.2), self._supervisor_cb)

        rospy.loginfo(
            f"[waypoint_manager] {len(self.waypoints)} waypoints loaded | "
            f"use_move_base={self.use_move_base} | "
            f"loop={self.loop_waypoints} | "
            f"tol={self.goal_tolerance} m"
        )
        for i, (x, y) in enumerate(self.waypoints):
            rospy.loginfo(f"  WP{i}: ({x:.3f}, {y:.3f})")

    # ── YAML / param parsing ──────────────────────────────────────────────────
    def _load_yaml(self, path: str) -> List[Tuple[float, float]]:
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            wps = data.get('waypoints', data) if isinstance(data, dict) else data
            result = [(float(p[0]), float(p[1])) for p in wps]
            rospy.loginfo(f"[waypoint_manager] Loaded {len(result)} waypoints from {path}")
            return result
        except Exception as e:
            rospy.logerr(f"[waypoint_manager] Could not load {path}: {e}")
            return []

    def _parse_flat(self, flat) -> List[Tuple[float, float]]:
        """Accept [x0,y0, x1,y1, ...] or [[x0,y0],[x1,y1],...]."""
        if flat and isinstance(flat[0], (list, tuple)):
            return [(float(p[0]), float(p[1])) for p in flat]
        if len(flat) % 2 != 0:
            rospy.logwarn("[waypoint_manager] Odd-length waypoints list — dropping last element.")
            flat = flat[:-1]
        return [(float(flat[i]), float(flat[i+1])) for i in range(0, len(flat), 2)]

    # ── Odometry callback (position update) ───────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    # ── TF fallback position ──────────────────────────────────────────────────
    def _get_robot_xy_tf(self) -> Optional[Tuple[float, float]]:
        try:
            (trans, _) = self.tf_listener.lookupTransform(
                self.parent_frame, self.base_frame, rospy.Time(0)
            )
            return trans[0], trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    # ── Goal sending ──────────────────────────────────────────────────────────
    def _send_goal(self, x: float, y: float):
        ps                     = PoseStamped()
        ps.header.stamp        = rospy.Time.now()
        ps.header.frame_id     = self.parent_frame
        ps.pose.position.x     = x
        ps.pose.position.y     = y
        ps.pose.position.z     = 0.0
        ps.pose.orientation.w  = 1.0

        if self.use_move_base and self.mb_client:
            goal          = MoveBaseGoal()
            goal.target_pose = ps
            self.mb_client.send_goal(
                goal,
                done_cb=self._mb_done_cb,
                active_cb=None,
                feedback_cb=None,
            )
            rospy.loginfo(
                f"[waypoint_manager] → move_base goal WP{self.current_idx}: "
                f"({x:.3f}, {y:.3f})"
            )
        else:
            self.pub_goal.publish(ps)
            rospy.loginfo(
                f"[waypoint_manager] → /goal_pose WP{self.current_idx}: "
                f"({x:.3f}, {y:.3f})"
            )

        self.goal_sent = True

    def _mb_done_cb(self, state, result):
        """Called by move_base actionlib when goal is reached or aborted."""
        import actionlib_msgs.msg as am
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(
                f"[waypoint_manager] move_base reached WP{self.current_idx}."
            )
            self._advance()
        else:
            rospy.logwarn(
                f"[waypoint_manager] move_base failed on WP{self.current_idx} "
                f"(state={state}). Retrying…"
            )
            self.goal_sent = False   # will retry next supervisor tick

    # ── Waypoint advancement ──────────────────────────────────────────────────
    def _advance(self):
        self.current_idx += 1
        self.goal_sent    = False

        if self.current_idx >= len(self.waypoints):
            if self.loop_waypoints:
                self.current_idx = 0
                rospy.loginfo("[waypoint_manager] All waypoints done — looping.")
            else:
                self.mission_done = True
                rospy.loginfo("[waypoint_manager] All waypoints reached. Mission complete.")

    # ── 5 Hz supervisor ───────────────────────────────────────────────────────
    def _supervisor_cb(self, _event):
        if self.mission_done or not self.waypoints:
            return

        # Get robot position (odom first, TF fallback)
        rx = self.robot_x
        ry = self.robot_y
        if rx is None or ry is None:
            pos = self._get_robot_xy_tf()
            if pos is None:
                return
            rx, ry = pos

        wp_x, wp_y = self.waypoints[self.current_idx]

        # Send goal if not yet sent
        if not self.goal_sent:
            self._send_goal(wp_x, wp_y)
            return

        # When not using move_base, we check arrival ourselves
        if not self.use_move_base:
            d = dist2d(rx, ry, wp_x, wp_y)
            if d < self.goal_tolerance:
                rospy.loginfo(
                    f"[waypoint_manager] Reached WP{self.current_idx} "
                    f"({wp_x:.3f}, {wp_y:.3f})  dist={d:.3f} m"
                )
                self._advance()

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = WaypointManager()
    node.spin()