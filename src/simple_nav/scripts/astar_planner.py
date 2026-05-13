#!/usr/bin/env python3

import math
import heapq
from typing import List, Tuple, Optional

import cv2
import yaml
import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid, Path, MapMetaData
from geometry_msgs.msg import Pose, PoseStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class AStarPlanner:
    def __init__(self):
        rospy.init_node('astar_planner', anonymous=False)

        # Parameters
        self.map_yaml_path = rospy.get_param('~map_yaml_path', '/home/rfran/dev/maps/clean/saved_map_clean.yaml')
        self.map_pgm_path = rospy.get_param('~map_pgm_path', '/home/rfran/dev/maps/clean/saved_map_clean.pgm')
        self.map_topic = rospy.get_param('~publish_map_topic', '/map')

        self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        self.path_topic = rospy.get_param('~path_topic', '/global_path')
        self.parent_frame = rospy.get_param('~parent_frame', 'world') #changed to map ->world
        self.base_frame = rospy.get_param('~base_frame', 'camera_link')

        self.occ_thresh = int(rospy.get_param('~occupied_thresh', 65))
        self.unknown_is_occ = bool(rospy.get_param('~treat_unknown_as_occupied', False))

        self.robot_radius = float(rospy.get_param('~robot_radius', 0.10))
        self.infl_extra = float(rospy.get_param('~inflation_extra', 0.05))
        self.allow_diagonal = bool(rospy.get_param('~allow_diagonal', True))

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # State
        self.map_msg: Optional[OccupancyGrid] = None
        self.grid: Optional[List[int]] = None
        self.costmap: Optional[List[int]] = None

        # ROS I/O
        self.pub_map = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1, latch=True)
        self.sub_goal = rospy.Subscriber(self.goal_topic, PoseStamped, self.cb_goal, queue_size=10)
        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=10)

        # Publish map repeatedly so RViz always sees it
        self.map_timer = rospy.Timer(rospy.Duration(1.0), self.publish_map)

        # Load map once at startup
        self.load_map_from_files()

        rospy.loginfo(
            "A* planner loaded map from:\n"
            "  YAML: %s\n"
            "  PGM:  %s\n"
            "Publishing map on %s\n"
            "Waiting for goals on %s",
            self.map_yaml_path,
            self.map_pgm_path,
            self.map_topic,
            self.goal_topic
        )

    def load_map_from_files(self):
        # Load YAML
        try:
            with open(self.map_yaml_path, 'r') as f:
                y = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr("Failed to read YAML: %s", e)
            return

        # Load image
        img = cv2.imread(self.map_pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rospy.logerr("Failed to read PGM: %s", self.map_pgm_path)
            return

        resolution = float(y.get('resolution', 0.05))
        origin = y.get('origin', [-15.0, -15.0, 0.0])
        negate = int(y.get('negate', 0))
        occupied_thresh = float(y.get('occupied_thresh', 0.65))
        free_thresh = float(y.get('free_thresh', 0.196))
        mode = str(y.get('mode', 'trinary')).strip().lower()

        h, w = img.shape[:2]

        # Build ROS OccupancyGrid data
        # 0 free
        # 100 occupied
        # -1 unknown
        data = []

        for row in img[::-1]:  # PGM is top-to-bottom, OccupancyGrid is bottom-to-top
            for pix in row:
                p = float(pix) / 255.0
                if negate == 0:
                    occ = 1.0 - p
                else:
                    occ = p

                if mode == 'trinary':
                    if occ > occupied_thresh:
                        data.append(100)
                    elif occ < free_thresh:
                        data.append(0)
                    else:
                        data.append(-1)
                else:
                    if occ > occupied_thresh:
                        data.append(100)
                    elif occ < free_thresh:
                        data.append(0)
                    else:
                        data.append(-1)

        msg = OccupancyGrid()
        msg.header.frame_id = self.parent_frame

        info = MapMetaData()
        info.resolution = resolution
        info.width = w
        info.height = h

        pose = Pose()
        pose.position.x = float(origin[0])
        pose.position.y = float(origin[1])
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        info.origin = pose

        msg.info = info
        msg.data = data

        self.map_msg = msg
        self.grid = list(data)
        self.costmap = None

        free_count = sum(1 for v in data if v == 0)
        occ_count = sum(1 for v in data if v == 100)
        unk_count = sum(1 for v in data if v == -1)

        rospy.loginfo(
            "Map loaded: %dx%d, res=%.3f, origin=(%.3f, %.3f) | free=%d, occ=%d, unk=%d",
            w, h, resolution, origin[0], origin[1], free_count, occ_count, unk_count
        )

    def publish_map(self, event=None):
        if self.map_msg is None:
            return
        self.map_msg.header.stamp = rospy.Time.now()
        self.map_msg.info.map_load_time = self.map_msg.header.stamp
        self.pub_map.publish(self.map_msg)

    def cb_goal(self, goal: PoseStamped):
        if self.map_msg is None or self.grid is None:
            rospy.logwarn("No map loaded yet.")
            return

        start_xy = self.get_robot_xy()
        if start_xy is None:
            rospy.logwarn("No TF map->camera_link yet. Is ORB-SLAM running?")
            return

        sx, sy = start_xy
        gx, gy = goal.pose.position.x, goal.pose.position.y

        start = self.world_to_cell(sx, sy)
        goal_cell = self.world_to_cell(gx, gy)

        if start is None or goal_cell is None:
            rospy.logwarn("Start or goal is outside the map bounds.")
            return

        if self.costmap is None:
            self.costmap = self.build_inflated_costmap()

        path_cells = self.astar(start, goal_cell, self.costmap)
        if not path_cells:
            rospy.logwarn("A* failed to find a path.")
            return

        path_msg = self.cells_to_path(path_cells)
        self.pub_path.publish(path_msg)
        rospy.loginfo("Published path with %d poses.", len(path_msg.poses))

    def get_robot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.2)
            )
            return (tf.transform.translation.x, tf.transform.translation.y)
        except TransformException:
            return None
        except Exception:
            return None

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        msg = self.map_msg
        if msg is None:
            return None

        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        cx = int(math.floor((x - ox) / res))
        cy = int(math.floor((y - oy) / res))

        if 0 <= cx < msg.info.width and 0 <= cy < msg.info.height:
            return (cx, cy)
        return None

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        msg = self.map_msg
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        x = ox + (cx + 0.5) * res
        y = oy + (cy + 0.5) * res
        return x, y

    def idx(self, cx: int, cy: int) -> int:
        msg = self.map_msg
        return cy * msg.info.width + cx

    def is_occupied(self, v: int) -> bool:
        if v < 0:
            return self.unknown_is_occ
        return v >= self.occ_thresh

    def build_inflated_costmap(self) -> List[int]:
        msg = self.map_msg
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution

        base = [0] * (w * h)
        occ_cells = []

        for cy in range(h):
            row_off = cy * w
            for cx in range(w):
                v = self.grid[row_off + cx]
                if self.is_occupied(v):
                    base[row_off + cx] = 100
                    occ_cells.append((cx, cy))

        inflate_m = self.robot_radius + self.infl_extra
        inflate_cells = int(math.ceil(inflate_m / res))
        inflate_cells = max(inflate_cells, 0)

        if inflate_cells == 0:
            return base

        inflated = base[:]
        for (ox, oy) in occ_cells:
            for dy in range(-inflate_cells, inflate_cells + 1):
                ny = oy + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in range(-inflate_cells, inflate_cells + 1):
                    nx = ox + dx
                    if nx < 0 or nx >= w:
                        continue
                    d = math.hypot(dx, dy) * res
                    if d <= inflate_m:
                        inflated[ny * w + nx] = 100
        return inflated

    def neighbors(self, cx: int, cy: int) -> List[Tuple[int, int, float]]:
        steps = [
            (cx + 1, cy, 1.0),
            (cx - 1, cy, 1.0),
            (cx, cy + 1, 1.0),
            (cx, cy - 1, 1.0),
        ]

        if self.allow_diagonal:
            rt2 = math.sqrt(2.0)
            steps.extend([
                (cx + 1, cy + 1, rt2),
                (cx + 1, cy - 1, rt2),
                (cx - 1, cy + 1, rt2),
                (cx - 1, cy - 1, rt2),
            ])

        msg = self.map_msg
        out = []
        for nx, ny, c in steps:
            if 0 <= nx < msg.info.width and 0 <= ny < msg.info.height:
                out.append((nx, ny, c))
        return out

    def astar(self, start: Tuple[int, int], goal: Tuple[int, int], costmap: List[int]) -> List[Tuple[int, int]]:
        def hfun(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if costmap[self.idx(*start)] >= 100 or costmap[self.idx(*goal)] >= 100:
            rospy.logwarn("Start or goal is in an occupied inflated cell.")
            return []

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        gscore = {start: 0.0}

        closed = set()
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for nx, ny, step_cost in self.neighbors(current[0], current[1]):
                if costmap[self.idx(nx, ny)] >= 100:
                    continue

                n = (nx, ny)
                tentative = gscore[current] + step_cost

                if n not in gscore or tentative < gscore[n]:
                    came_from[n] = current
                    gscore[n] = tentative
                    f = tentative + hfun(n, goal)
                    heapq.heappush(open_heap, (f, n))

        return []

    def reconstruct_path(self, came_from, current):
        out = [current]
        while current in came_from:
            current = came_from[current]
            out.append(current)
        out.reverse()
        return out

    def cells_to_path(self, cells: List[Tuple[int, int]]) -> Path:
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.parent_frame

        for cx, cy in cells:
            ps = PoseStamped()
            ps.header = path.header
            x, y = self.cell_to_world(cx, cy)
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        return path


def main():
    try:
        AStarPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
