#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/image_encodings.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "image_geometry/pinhole_camera_model.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <boost/bind.hpp>
#include <cmath>
#include <algorithm>
#include <string>

using std::string;

class TargetDetector {
public:
    TargetDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh),
          sub_rgb_(nh_, "/camera/rgb/image_raw", 1),
          sub_depth_(nh_, "/camera/depth/image_raw", 1),
          sub_info_(nh_, "/camera/rgb/camera_info", 1),
          sync_(SyncPolicy(10), sub_rgb_, sub_depth_, sub_info_)
    {
        sync_.registerCallback(boost::bind(&TargetDetector::cb, this, _1, _2, _3));
        pub_pose_ = nh_.advertise<geometry_msgs::PoseStamped>("/target/pose", 1);

        pnh_.param("min_blob_area", min_blob_area_, 200);
        pnh_.param("depth_scale", depth_scale_, 0.001);

        ROS_INFO("detector up (min_blob_area=%d, depth_scale=%f)",min_blob_area_,depth_scale_);
    }

private:
    bool findRed(const cv::Mat &bgr, cv::Point &c) {
        if (bgr.empty()) return false;
        cv::Mat hsv; cv::cvtColor(bgr,hsv,cv::COLOR_BGR2HSV);
        cv::Mat m1,m2,mask;
        cv::inRange(hsv, cv::Scalar(0,100,100),  cv::Scalar(10,255,255),  m1);
        cv::inRange(hsv, cv::Scalar(160,100,100),cv::Scalar(179,255,255), m2);
        mask = m1 | m2;
        cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1),1);
        cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1),1);
        cv::Moments mm = cv::moments(mask,true);
        if (mm.m00 < min_blob_area_) return false;
        double inv = 1.0/(mm.m00+1e-9);
        c.x = (int)(mm.m10*inv);
        c.y = (int)(mm.m01*inv);
        return true;
    }

    void cb(const sensor_msgs::ImageConstPtr &rgb_msg,
            const sensor_msgs::ImageConstPtr &depth_msg,
            const sensor_msgs::CameraInfoConstPtr &info_msg)
    {
        cam_.fromCameraInfo(info_msg);

        cv_bridge::CvImageConstPtr cv_rgb, cv_depth;
        try {
            cv_rgb = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("rgb cv_bridge: %s", e.what());
            return;
        }
        try {
            if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                cv_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            } else {
                cv_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            }
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("depth cv_bridge: %s", e.what());
            return;
        }

        cv::Mat img = cv_rgb->image;
        cv::Mat dep = cv_depth->image;
        if (img.empty() || dep.empty()) return;

        cv::Point c;
        if (!findRed(img,c)) return;
        if (c.x < 0 || c.y < 0 || c.x >= dep.cols || c.y >= dep.rows) return;

        float z = 0.f;
        int t = dep.type();
        if (t == CV_16UC1) {
            uint16_t d = dep.at<uint16_t>(c.y,c.x);
            z = d * depth_scale_;
        } else if (t == CV_32FC1) {
            z = dep.at<float>(c.y,c.x);
        } else {
            return;
        }
        if (z <= 0.1f || z > 10.0f) return;

        cv::Point2d pix((double)c.x,(double)c.y);
        cv::Point3d ray = cam_.projectPixelTo3dRay(pix);
        cv::Point3d p = ray * z;

        geometry_msgs::PoseStamped msg;
        msg.header.stamp    = rgb_msg->header.stamp;
        msg.header.frame_id = info_msg->header.frame_id;
        msg.pose.position.x = p.x;
        msg.pose.position.y = p.y;
        msg.pose.position.z = p.z;
        msg.pose.orientation.x = 0.0;
        msg.pose.orientation.y = 0.0;
        msg.pose.orientation.z = 0.0;
        msg.pose.orientation.w = 1.0;

        pub_pose_.publish(msg);
    }

    ros::NodeHandle nh_, pnh_;
    message_filters::Subscriber<sensor_msgs::Image> sub_rgb_;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_info_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image,sensor_msgs::Image,sensor_msgs::CameraInfo> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync_;

    image_geometry::PinholeCameraModel cam_;
    ros::Publisher pub_pose_;
    int min_blob_area_;
    double depth_scale_;
};

class TargetTracker {
public:
    TargetTracker(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
        sub_pose_ = nh_.subscribe("/target/pose",1,&TargetTracker::poseCb,this);
        pub_cmd_  = nh_.advertise<geometry_msgs::Twist>("/cmd_vel",1);

        pnh_.param("desired_distance",desired_dist_,1.0);
        pnh_.param("k_lin",k_lin_,0.5);
        pnh_.param("k_ang",k_ang_,1.0);
        pnh_.param("max_lin",max_lin_,0.5);
        pnh_.param("max_ang",max_ang_,1.0);
        pnh_.param("timeout",timeout_,0.5);

        last_ = ros::Time(0);
        ROS_INFO("tracker up (d=%.2f)",desired_dist_);
    }

    void step() {
        ros::Time now = ros::Time::now();
        double dt = (now - last_).toSec();
        if (dt > timeout_) {
            geometry_msgs::Twist z;
            pub_cmd_.publish(z);
        }
    }

private:
    void poseCb(const geometry_msgs::PoseStamped::ConstPtr &msg) {
        last_ = msg->header.stamp;

        double x = msg->pose.position.x;
        double z = msg->pose.position.z;

        if (z <= 0.0) return;

        double ang = std::atan2(x,z);
        double err = z - desired_dist_;

        geometry_msgs::Twist cmd;
        double w = k_ang_ * ang;
        if (w >  max_ang_) w =  max_ang_;
        if (w < -max_ang_) w = -max_ang_;
        cmd.angular.z = w;

        double v = k_lin_ * err;
        if (v >  max_lin_) v =  max_lin_;
        if (v < -max_lin_) v = -max_lin_;
        cmd.linear.x = v;

        pub_cmd_.publish(cmd);
    }

    ros::NodeHandle nh_, pnh_;
    ros::Subscriber sub_pose_;
    ros::Publisher  pub_cmd_;

    double desired_dist_, k_lin_, k_ang_, max_lin_, max_ang_, timeout_;
    ros::Time last_;
};

int main(int argc,char **argv) {
    ros::init(argc,argv,"some_node");
    string n = (argc>0 && argv[0]) ? string(argv[0]) : string();

    if (n.find("tracker") != string::npos) {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        TargetTracker tr(nh,pnh);
        ros::Rate r(30);
        while (ros::ok()) {
            ros::spinOnce();
            tr.step();
            r.sleep();
        }
    } else {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");
        TargetDetector det(nh,pnh);
        ros::spin();
    }
    return 0;
}
