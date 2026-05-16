# AV Robots: Robot Localization, Mapping & Navigation Using a Monocular RGB Camera

A low-cost indoor robot localization, mapping, and navigation system using a monocular RGB camera, ORB-SLAM3, A* path planning, and Pure Pursuit control within the ROS framework.

This project implements a complete autonomous navigation pipeline using a single camera, demonstrating that monocular vision can serve as a practical and cost-effective alternative to LiDAR for indoor robot navigation in structured environments. The system performs real-time localization and mapping, builds occupancy grids, plans collision-free paths, and tracks them smoothly to guide the robot to its destination.

The navigation pipeline detects the robot's position using visual features, creates a 3D environment map, converts the 3D map into a 2D occupancy grid map for path planning, and outputs motion commands: 

* **Navigate** – Execute planned path
* **Explore** – Map unknown areas
* **Replan** – Avoid dynamic obstacles
* **Reach Goal** – Arrive at target location

These outputs guide the robot through indoor environments with **4.5% mapping error** and **86.67% path planning success rate**, while costing **43–63% less** than LiDAR-based systems.

---

## Project Overview

Modern robot navigation systems often rely on expensive sensors such as LiDAR or stereo cameras. This project demonstrates that reliable indoor navigation can be achieved using only a monocular camera and lightweight algorithms optimized for embedded hardware.

The localization and navigation pipeline operates by:
1. **Detecting visual features** in the camera feed using ORB-SLAM3
2. **Building and Creating 2D occupancy grids map** of environment in real-time for path planning
3. **Planning collision-free paths** using A* algorithm
4. **Tracking paths smoothly** using Pure Pursuit control

The result is a complete autonomous navigation system running on a Raspberry Pi 5 that rivals expensive robotic platforms while maintaining a fraction of the cost.

---

## Quick Start

### What You Need

**Hardware:**
- Raspberry Pi 5 (8 GB RAM)
- USB Monocular Camera (640×360, 30 FPS)
- Wheeled Robot Platform
- Host Computer
- Local Wi-Fi Network

**Software:**
- Ubuntu 
- ROS 
- Python 3.7+
- C++11 or higher
- OpenCV, Eigen3, Pangolin

## Step 1: Install ROS 
 
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
 
---
 
## Step 2: Install Dependencies
 
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libopencv-dev libeigen3-dev libpangolin-dev python3-pip python3-numpy python-is-python3
 
# Install Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. && cmake --build . && sudo cmake --install .
cd ../..
```
 
---
 
## Step 3: Create Workspace & Clone Repos
 
```bash
mkdir -p ~/slam_ws/src && cd ~/slam_ws
catkin_init_workspace src/
cd src/
git clone https://github.com/frncsrys/Thesis.git
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
cd ~/slam_ws
```
 
---
 
## Step 4: Build ORB-SLAM3
 
```bash
cd ~/slam_ws/src/ORB_SLAM3
chmod +x build.sh
./build.sh
```
 
---
 
## Step 5: Install Dependencies & Build Workspace
 
```bash
cd ~/slam_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make -j4
source devel/setup.bash
echo "source ~/slam_ws/devel/setup.bash" >> ~/.bashrc
```

---
 
## Step 6: Run System
 
**Option 1: Automated**
```bash
cd ~/slam_ws
./control.sh
```
 
**Option 2: Manual (5 terminals)**
```bash
# Terminal 1
roscore
 
# Terminal 2
source ~/slam_ws/devel/setup.bash
roslaunch ORB_SLAM3 mono.launch
 
# Terminal 3
source ~/slam_ws/devel/setup.bash
rosrun simple_nav occupancy_grid_generator.py
 
# Terminal 4
source ~/slam_ws/devel/setup.bash
roslaunch simple_nav navigation.launch
 
# Terminal 5
source ~/slam_ws/devel/setup.bash
rviz -d ~/slam_ws/src/simple_nav/config/rviz_config.rviz
```
 
Use **2D Nav Goal** in RViz to set destinations.
 
---

## How It Works

```
IP Camera Input
    ↓
ORB-SLAM3 (Localization & Mapping)
    ↓
Occupancy Grid Generator
    ↓
A* Path Planner
    ↓
Pure Pursuit Controller
    ↓
Vehicle
```

| Component | Function |
|-----------|----------|
| **ORB-SLAM3** | Real-time visual SLAM; detects features and builds 3D map |
| **Occupancy Grid** | Converts 3D map to 2D navigation grid |
| **A* Planner** | Computes collision-free paths to goal |
| **Pure Pursuit** | Smoothly tracks planned path with steering control |

---

## Hardware Configuration

### Robot Platform

- Raspberry Pi 5 (8 GB RAM)
- USB Monocular Camera (640×360, 30 FPS)
- RC Robot Chassis
- Host Laptop
- 20,000 mAh Power Bank

### Experimental Environment

- Arena Size: 400 cm × 250 cm
- Corridor Width: 55 cm
- Inner Obstacle: 290 cm × 140 cm
- Lighting: Stable
- Surfaces: Textured Cardboard

---

## Key Features

- ✅ Monocular SLAM using ORB-SLAM3
- ✅ IP Camera intergration
- ✅ Real-time 3D environment mapping
- ✅ 2D Occupancy Grid generation
- ✅ A* Global Path Planning
- ✅ Pure Pursuit Path Tracking
- ✅ Embedded Raspberry Pi 5 deployment
- ✅ Integrated with AV-Robots Distance Detection (YOLOv8n obstacle detection, stop/go decisions)

---

## Authors

- Jesse Rey D. Isidro
- Jericho S. Lampano
- Alexis J.V.R. Magno
- Francis Albert M. Reyes

**Institution:** College of Computer Studies, Technological Institute of the Philippines – Manila  
**Adviser:** Dr. Jheanel E. Estrada

---

## References

- [ORB-SLAM3 Repository](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [ROS Documentation](https://www.ros.org/)
- [Pangolin Visualization](https://github.com/stevenlovegrove/Pangolin)
- [AV-Robots Distance Detection](https://github.com/cjricafrente/AV-Robots-Distance_Detection)
- [Project Repository](https://github.com/frncsrys/Thesis)

---

## License

This project is released under the GPLv3 License and is intended for academic and research purposes.

---

## Acknowledgments

- College of Computer Studies, TIP Manila
- ORB-SLAM3 Development Team
- ROS Community