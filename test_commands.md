# Test Commands



# Mapping
python3 /home/rfran/dev/ORB_SLAM3/Examples/ROS/simple_nav/scripts/cam_stream.py

roslaunch ORB_SLAM3 orbgrid.launch

rosrun map_server map_saver -f ~/dev/maps/nice_map4

rostopic echo /orb_slam3/pose -n 1

# Navigation
python3 /home/rfran/dev/ORB_SLAM3/Examples/ROS/simple_nav/scripts/cam_stream.py

roslaunch ORB_SLAM3 orbgrid.launch enable_gmapping:=false

roslaunch simple_nav orbgrid_nav.launch enable_camera:=false

rviz

In RViz:

Fixed Frame → world
Add /map
Add /global_path
Add /sim/pose or /orb_slam3/pose_stamped
Add /pp/markers
Click 2D Nav Goal → click on the map

# Object Detection
python3 /home/rfran/dev/ORB_SLAM3/Examples/ROS/simple_nav/scripts/stream.py

roscore

roslaunch video_stream_opencv camera.launch \
  video_stream_provider:=http://10.33.247.220:5000/video \
  fps:=30 width:=640 height:=360 \
  camera_name:=camera

rosrun simple_nav perception_node.py \
  _image_topic:=/camera/image_raw \
  _model_path:=/home/rfran/slam_ws/src/detection/AV-Robots-Distance_Detection-Mandap/best_ncnn_model \
  _calib_path:=/home/rfran/slam_ws/src/detection/AV-Robots-Distance_Detection-Mandap/calibration_params.npz \
  _target_width:=640 \
  _target_height:=360 \
  _publish_debug_img:=true

rqt_image_view /perception/debug_image