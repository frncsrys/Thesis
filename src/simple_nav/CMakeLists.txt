cmake_minimum_required(VERSION 3.0.2)
project(simple_nav)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/astar_planner.py
  scripts/pure_pursuit.py
  scripts/obstacle_stop.py
  scripts/waypoint_manager.py
  scripts/pose_simulator.py
  scripts/perception_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
