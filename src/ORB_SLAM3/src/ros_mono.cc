#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>

#include "../../../include/System.h"

using namespace std;

// ─── Metrics Counters ─────────────────────────────────────────────
long total_frames = 0;
long tracked_frames = 0;
long tracking_loss_events = 0;
long relocalizations = 0;

int last_tracking_state = -1;


// ─── ImageGrabber Class Definition ───────────────────────────────────────────
class ImageGrabber
{
public:
   ImageGrabber(ORB_SLAM3::System* pSLAM,
                ros::Publisher pose_pub,
                ros::Publisher pose_stamped_pub,
                ros::Publisher cloud_pub,
                const string& parent_frame,
                const string& child_frame)
   {
       mpSLAM               = pSLAM;
       posePublisher        = pose_pub;
       poseStampedPublisher = pose_stamped_pub;
       cloudPublisher       = cloud_pub;
       parentFrame          = parent_frame;
       childFrame           = child_frame;
   }

   void GrabImage(const sensor_msgs::ImageConstPtr& msg);

   ORB_SLAM3::System* mpSLAM;

   ros::Publisher posePublisher;
   ros::Publisher poseStampedPublisher;
   ros::Publisher cloudPublisher;

   tf::TransformBroadcaster tfBroadcaster;

   string parentFrame;
   string childFrame;
};


// ─── Main Node ───────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
   ros::init(argc, argv, "orb_slam3_mono_mapping");

   if(argc != 3)
   {
       cerr << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
       return 1;
   }

   ros::NodeHandle nh;
   ros::NodeHandle nh_private("~");

   string parent_frame, child_frame, pose_topic;
   nh_private.param<string>("parent_frame", parent_frame, "world");
   nh_private.param<string>("child_frame",  child_frame,  "camera_link");
   nh_private.param<string>("pose_topic",   pose_topic,   "/orb_slam3/pose");

   ROS_INFO("TF: %s -> %s  |  pose topic: %s",
            parent_frame.c_str(), child_frame.c_str(), pose_topic.c_str());

   // Initialize ORB-SLAM3
   ORB_SLAM3::System SLAM(
       argv[1],
       argv[2],
       ORB_SLAM3::System::MONOCULAR,
       true
   );

   ros::Publisher pose_pub =
       nh.advertise<nav_msgs::Odometry>(pose_topic, 10);

   ros::Publisher pose_stamped_pub =
       nh.advertise<geometry_msgs::PoseStamped>(pose_topic + "_stamped", 10);

   ros::Publisher cloud_pub =
       nh.advertise<sensor_msgs::PointCloud2>("/orb_slam3/all_points", 1);

   ImageGrabber igb(&SLAM, pose_pub, pose_stamped_pub, cloud_pub,
                    parent_frame, child_frame);

   ros::Subscriber sub =
       nh.subscribe("/camera/image_raw", 1,
                    &ImageGrabber::GrabImage, &igb);

   ros::spin();

   // Shutdown SLAM
   SLAM.Shutdown();
   SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

   // ─── Compute Metrics ─────────────────────────────────────────────
   double tracking_success_rate =
       (double)tracked_frames / total_frames * 100.0;

   double relocalization_rate = 0;

   if(tracking_loss_events > 0)
   {
       relocalization_rate =
           (double)relocalizations / tracking_loss_events * 100.0;
   }

    int keyframes = SLAM.GetTrackedKeyPointsUn().size();
    int mappoints = SLAM.GetTrackedMapPoints().size();

   cout << endl;
   cout << "========== ORB-SLAM3 METRICS ==========" << endl;
   cout << "Total Frames: " << total_frames << endl;
   cout << "Tracked Frames: " << tracked_frames << endl;
   cout << "Tracking Success Rate: " << tracking_success_rate << "%" << endl;
   cout << "Tracking Loss Events: " << tracking_loss_events << endl;
   cout << "Relocalization Rate: " << relocalization_rate << "%" << endl;
   cout << "Keyframes Generated: " << keyframes << endl;
   cout << "Map Points: " << mappoints << endl;
   cout << "========================================" << endl;

   return 0;
}


// ─── GrabImage Implementation ────────────────────────────────────────────────
void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImageConstPtr cv_ptr;

   try
   {
       cv_ptr = cv_bridge::toCvShare(msg);
   }
   catch(cv_bridge::Exception& e)
   {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
   }

   // ── Track Frame ─────────────────────────────────────────────
   Sophus::SE3f pose = mpSLAM->TrackMonocular(
       cv_ptr->image,
       cv_ptr->header.stamp.toSec()
   );

   total_frames++;

   int current_state = mpSLAM->GetTrackingState();

   if(current_state == 2)
   {
       tracked_frames++;

       if(last_tracking_state != 2 && last_tracking_state != -1)
           relocalizations++;
   }
   else
   {
       if(last_tracking_state == 2)
           tracking_loss_events++;

       last_tracking_state = current_state;
       return;
   }

   last_tracking_state = current_state;

   Eigen::Matrix4f Tcw = pose.matrix();
   Eigen::Matrix3f Rcw = Tcw.block<3,3>(0,0);
   Eigen::Vector3f tcw = Tcw.block<3,1>(0,3);

   Eigen::Matrix3f Rwc = Rcw.transpose();
   Eigen::Vector3f twc = -Rwc * tcw;

   Eigen::Matrix3f R_c2r;
   R_c2r <<  0,  0,  1,
            -1,  0,  0,
             0, -1,  0;

   Eigen::Matrix3f Rwc_ros = R_c2r * Rwc * R_c2r.transpose();
   Eigen::Vector3f twc_ros = R_c2r * twc;

   tf::Matrix3x3 tf3d(
       Rwc_ros(0,0), Rwc_ros(0,1), Rwc_ros(0,2),
       Rwc_ros(1,0), Rwc_ros(1,1), Rwc_ros(1,2),
       Rwc_ros(2,0), Rwc_ros(2,1), Rwc_ros(2,2)
   );

   tf::Vector3 tf3d_t(twc_ros(0), twc_ros(1), twc_ros(2));
   tf::Transform transform(tf3d, tf3d_t);

   tfBroadcaster.sendTransform(
       tf::StampedTransform(transform,
                            msg->header.stamp,
                            parentFrame,
                            childFrame));

   tf::Quaternion q = transform.getRotation();

   geometry_msgs::Pose ros_pose;

   ros_pose.position.x = transform.getOrigin().x();
   ros_pose.position.y = transform.getOrigin().y();
   ros_pose.position.z = transform.getOrigin().z();

   ros_pose.orientation.x = q.x();
   ros_pose.orientation.y = q.y();
   ros_pose.orientation.z = q.z();
   ros_pose.orientation.w = q.w();

   nav_msgs::Odometry odom;

   odom.header.stamp = msg->header.stamp;
   odom.header.frame_id = parentFrame;
   odom.child_frame_id = childFrame;
   odom.pose.pose = ros_pose;

   posePublisher.publish(odom);

   geometry_msgs::PoseStamped ps;

   ps.header.stamp = msg->header.stamp;
   ps.header.frame_id = parentFrame;
   ps.pose = ros_pose;

   poseStampedPublisher.publish(ps);

   vector<ORB_SLAM3::MapPoint*> mapPoints = mpSLAM->GetTrackedMapPoints();
   if(mapPoints.empty()) return;

   sensor_msgs::PointCloud2 cloud;

   cloud.header.frame_id = parentFrame;
   cloud.header.stamp = msg->header.stamp;
   cloud.height = 1;
   cloud.width = mapPoints.size();

   sensor_msgs::PointCloud2Modifier modifier(cloud);
   modifier.setPointCloud2FieldsByString(1, "xyz");
   modifier.resize(mapPoints.size());

   sensor_msgs::PointCloud2Iterator<float> iter_x(cloud,"x");
   sensor_msgs::PointCloud2Iterator<float> iter_y(cloud,"y");
   sensor_msgs::PointCloud2Iterator<float> iter_z(cloud,"z");

   int valid_points = 0;

   for(size_t i=0;i<mapPoints.size();i++)
   {
       if(!mapPoints[i] || mapPoints[i]->isBad())
           continue;

       Eigen::Vector3f pos = mapPoints[i]->GetWorldPos();
       Eigen::Vector3f pos_ros = R_c2r * pos;

       *iter_x = pos_ros.x();
       *iter_y = pos_ros.y();
       *iter_z = pos_ros.z();

       ++iter_x;
       ++iter_y;
       ++iter_z;

       valid_points++;
   }

   cloud.width = valid_points;
   cloud.row_step = cloud.width * cloud.point_step;
   cloud.data.resize(cloud.row_step * cloud.height);

   if(valid_points > 0)
       cloudPublisher.publish(cloud);
}