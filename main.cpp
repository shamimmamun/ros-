

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <limits>
#include <cmath>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

namespace mv_rgbd_vision {

struct CameraIntrinsics {
    std::string name;
    int width = 0, height = 0;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    double depth_scale = 0.001;
    Eigen::Matrix4d T_cam_world = Eigen::Matrix4d::Identity();
    std::string rgb_pattern;
    std::string depth_pattern;
};

struct RGBDFrame {
    cv::Mat rgb;
    cv::Mat depth;
    double timestamp = 0.0;
};

struct MultiViewFrame {
    std::vector<RGBDFrame> views;
    double timestamp = 0.0;
};

struct Pose3D {
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

struct ColorSegmentationConfig {
    cv::Scalar hsv_lower;
    cv::Scalar hsv_upper;
    int morph_kernel_size = 5;
};

struct EdgeDetectorConfig {
    double low_threshold = 50.0;
    double high_threshold = 150.0;
    int aperture_size = 3;
};

struct FiducialConfig {
    double marker_length_m = 0.04;
    int dictionary_id = cv::aruco::DICT_4X4_50;
};

struct CoverageStats {
    double covered_fraction = 0.0;
    int total_cells = 0;
    int covered_cells = 0;
};

class DatasetLoader {
public:
    DatasetLoader(const std::string& root,const std::vector<CameraIntrinsics>& cams)
        : dataset_root_(root), cameras_(cams), idx_(0) {}

    bool loadNext(MultiViewFrame &out) {
        std::vector<RGBDFrame> v;
        v.reserve(cameras_.size());
        for (auto const& c: cameras_) {
            auto f = loadOne(c, idx_);
            if (f.rgb.empty() || f.depth.empty()) return false;
            v.push_back(std::move(f));
        }
        out.views = std::move(v);
        out.timestamp = (double)idx_;
        ++idx_;
        return true;
    }

    void reset(){ idx_ = 0; }

private:
    std::string dataset_root_;
    std::vector<CameraIntrinsics> cameras_;
    int idx_;

    RGBDFrame loadOne(const CameraIntrinsics& cam,int index) const {
        RGBDFrame fr;
        char buf[512];

        std::snprintf(buf,sizeof(buf),cam.rgb_pattern.c_str(),index);
        std::string rgbp = dataset_root_ + "/" + buf;
        std::snprintf(buf,sizeof(buf),cam.depth_pattern.c_str(),index);
        std::string depthp = dataset_root_ + "/" + buf;

        fr.rgb = cv::imread(rgbp,cv::IMREAD_COLOR);
        fr.depth = cv::imread(depthp,cv::IMREAD_UNCHANGED);
        fr.timestamp = (double)index;
        return fr;
    }
};

class ColorSegmenter {
public:
    explicit ColorSegmenter(const ColorSegmentationConfig& c): cfg_(c) {}
    cv::Mat segment(const cv::Mat &bgr) const {
        cv::Mat hsv,mask;
        cv::cvtColor(bgr,hsv,cv::COLOR_BGR2HSV);
        cv::inRange(hsv,cfg_.hsv_lower,cfg_.hsv_upper,mask);
        int k = cfg_.morph_kernel_size;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(k,k));
        cv::morphologyEx(mask,mask,cv::MORPH_OPEN,kernel);
        cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,kernel);
        return mask;
    }
private:
    ColorSegmentationConfig cfg_;
};

class EdgeDetector {
public:
    explicit EdgeDetector(const EdgeDetectorConfig& c): cfg_(c) {}
    cv::Mat refineMask(const cv::Mat &bgr,const cv::Mat &m0) const {
        cv::Mat gray;
        cv::cvtColor(bgr,gray,cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray,gray,cv::Size(5,5),1.0);

        cv::Mat edges;
        cv::Canny(gray,edges,cfg_.low_threshold,cfg_.high_threshold,cfg_.aperture_size);

        cv::Mat edd;
        cv::dilate(edges,edd,cv::Mat(),cv::Point(-1,-1),1);

        cv::Mat em;
        cv::threshold(edd,em,0,255,cv::THRESH_BINARY);

        cv::Mat out;
        cv::bitwise_and(m0,em,out);
        cv::bitwise_or(out,m0,out);
        return out;
    }
private:
    EdgeDetectorConfig cfg_;
};

class FiducialPoseEstimator {
public:
    FiducialPoseEstimator(const FiducialConfig& c,const CameraIntrinsics& intr)
        : cfg_(c), intr_(intr)
    {
        dict_ = cv::aruco::getPredefinedDictionary(cfg_.dictionary_id);
        camK_ = (cv::Mat_<double>(3,3) <<
            intr_.fx,0,intr_.cx,
            0,intr_.fy,intr_.cy,
            0,0,1);
        dist_ = cv::Mat::zeros(1,5,CV_64F);
    }

    bool estimatePose(const cv::Mat &img, Pose3D &pose) const {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(img,dict_,corners,ids);
        if (ids.empty()) return false;

        std::vector<cv::Vec3d> rvecs,tvecs;
        cv::aruco::estimatePoseSingleMarkers(
            corners,
            cfg_.marker_length_m,
            camK_,
            dist_,
            rvecs,
            tvecs
        );
        cv::Vec3d rv = rvecs[0], tv = tvecs[0];

        cv::Mat Rcv;
        cv::Rodrigues(rv,Rcv);

        Eigen::Matrix3d R;
        cv::cv2eigen(Rcv,R);
        Eigen::Vector3d t(tv[0],tv[1],tv[2]);

        pose.q = Eigen::Quaterniond(R);
        pose.t = t;
        return true;
    }

private:
    FiducialConfig cfg_;
    CameraIntrinsics intr_;
    cv::Ptr<cv::aruco::Dictionary> dict_;
    cv::Mat camK_;
    cv::Mat dist_;
};

class PointCloudFuser {
public:
    PointCloudFuser(const std::vector<CameraIntrinsics>& cams): cams_(cams) {}

    PointCloudXYZ::Ptr generatePointCloud(const RGBDFrame &f,
                                          const CameraIntrinsics &cam,
                                          const Pose3D *obj_pose=nullptr) const {
        PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
        cloud->reserve(cam.width * cam.height);

        bool d16 = (f.depth.type() == CV_16U);

        for (int v=0; v<f.depth.rows; ++v) {
            for (int u=0; u<f.depth.cols; ++u) {
                float d=0.f;
                if (d16) {
                    uint16_t raw = f.depth.at<uint16_t>(v,u);
                    d = raw * (float)cam.depth_scale;
                } else {
                    d = f.depth.at<float>(v,u);
                }
                pcl::PointXYZ pc = backProjectPixel(u,v,d,cam);
                if (!std::isfinite(pc.x)) continue;

                Eigen::Vector4d pch(pc.x,pc.y,pc.z,1.0);
                Eigen::Vector4d pwh = cam.T_cam_world * pch;
                Eigen::Vector3d pw = pwh.head<3>();

                if (obj_pose) {
                    Eigen::Vector3d tmp = pw;
                    Eigen::Vector3d po = obj_pose->q.inverse() * (tmp - obj_pose->t);
                    pw = po;
                }

                cloud->push_back(pcl::PointXYZ(
                    (float)pw.x(),(float)pw.y(),(float)pw.z()
                ));
            }
        }
        return cloud;
    }

    PointCloudXYZ::Ptr fuseViews(const MultiViewFrame &mv,
                                 const Pose3D *obj_pose=nullptr) const {
        PointCloudXYZ::Ptr fused(new PointCloudXYZ);
        for (size_t i=0;i<mv.views.size() && i<cams_.size();++i) {
            auto c = generatePointCloud(mv.views[i],cams_[i],obj_pose);
            *fused += *c;
        }
        return fused;
    }

private:
    std::vector<CameraIntrinsics> cams_;

    pcl::PointXYZ backProjectPixel(int u,int v,float d,const CameraIntrinsics &cam) const {
        if (d<=0.f || !std::isfinite(d)) {
            float nan = std::numeric_limits<float>::quiet_NaN();
            return pcl::PointXYZ(nan,nan,nan);
        }
        float x = ( (float)u - (float)cam.cx ) * d / (float)cam.fx;
        float y = ( (float)v - (float)cam.cy ) * d / (float)cam.fy;
        float z = d;
        return pcl::PointXYZ(x,y,z);
    }
};

class CoverageMapper {
public:
    explicit CoverageMapper(double vs): voxel_size_(vs) {}
    CoverageStats computeCoverage(const PointCloudXYZ::Ptr &cloud) const {
        CoverageStats s;
        if (!cloud || cloud->empty()) return s;

        float minx = std::numeric_limits<float>::max();
        float miny = std::numeric_limits<float>::max();
        float minz = std::numeric_limits<float>::max();
        float maxx = -minx, maxy = -miny, maxz = -minz;

        for (auto const& p: cloud->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            minx = std::min(minx,p.x);
            miny = std::min(miny,p.y);
            minz = std::min(minz,p.z);
            maxx = std::max(maxx,p.x);
            maxy = std::max(maxy,p.y);
            maxz = std::max(maxz,p.z);
        }

        if (!std::isfinite(minx) || !std::isfinite(maxx)) return s;

        int nx = (int)std::ceil((maxx-minx)/voxel_size_) + 1;
        int ny = (int)std::ceil((maxy-miny)/voxel_size_) + 1;
        int nz = (int)std::ceil((maxz-minz)/voxel_size_) + 1;

        std::unordered_set<std::size_t> occ;

        auto hashIndex = [nx,ny](int ix,int iy,int iz)->std::size_t {
            return (std::size_t)ix + (std::size_t)iy * nx +
                   (std::size_t)iz * nx * ny;
        };

        for (auto const& p: cloud->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            int ix = (int)std::floor((p.x-minx)/voxel_size_);
            int iy = (int)std::floor((p.y-miny)/voxel_size_);
            int iz = (int)std::floor((p.z-minz)/voxel_size_);
            occ.insert(hashIndex(ix,iy,iz));
        }

        s.covered_cells = (int)occ.size();
        s.total_cells = nx*ny*nz;
        if (s.total_cells>0) {
            s.covered_fraction = (double)s.covered_cells/(double)s.total_cells;
        }
        return s;
    }
private:
    double voxel_size_;
};

class Pipeline {
public:
    Pipeline(const std::string &root,
             const std::string &cam_cfg,
             const std::string &color_cfg,
             const std::string &marker_cfg)
        : cameras_(loadCameras(cam_cfg)),
          loader_(root,cameras_),
          seg_(loadColorConfig(color_cfg)),
          edge_(EdgeDetectorConfig{}),
          fid_(loadMarkerConfig(marker_cfg),cameras_.at(0)),
          fuser_(cameras_),
          cov_(0.01)
    {}

    void run() {
        MultiViewFrame mv;
        int k = 0;
        while (loader_.loadNext(mv)) {
            std::cout << "frame " << k << "...\n";

            std::vector<cv::Mat> masks;
            masks.reserve(mv.views.size());
            for (auto const& v : mv.views) {
                auto m0 = seg_.segment(v.rgb);
                auto m1 = edge_.refineMask(v.rgb,m0);
                masks.push_back(m1);
            }

            Pose3D obj;
            bool has_pose = fid_.estimatePose(mv.views[0].rgb,obj);

            PointCloudXYZ::Ptr pc;
            if (has_pose) pc = fuser_.fuseViews(mv,&obj);
            else pc = fuser_.fuseViews(mv,nullptr);

            auto st = cov_.computeCoverage(pc);
            std::cout << "  coverage " << st.covered_fraction*100.0
                      << "% (" << st.covered_cells << "/" << st.total_cells
                      << " voxels)\n";

            ++k;
        }
        std::cout << "done\n";
    }

private:
    std::vector<CameraIntrinsics> cameras_;
    DatasetLoader loader_;
    ColorSegmenter seg_;
    EdgeDetector edge_;
    FiducialPoseEstimator fid_;
    PointCloudFuser fuser_;
    CoverageMapper cov_;

    std::vector<CameraIntrinsics> loadCameras(const std::string &path) const {
        std::vector<CameraIntrinsics> out;
        YAML::Node root = YAML::LoadFile(path);
        auto ycams = root["cameras"];
        if (!ycams || !ycams.IsSequence()) {
            throw std::runtime_error("bad camera config");
        }
        for (size_t i=0;i<ycams.size();++i) {
            CameraIntrinsics c;
            c.name          = ycams[i]["name"].as<std::string>();
            c.rgb_pattern   = ycams[i]["rgb_pattern"].as<std::string>();
            c.depth_pattern = ycams[i]["depth_pattern"].as<std::string>();
            c.width         = ycams[i]["width"].as<int>();
            c.height        = ycams[i]["height"].as<int>();
            c.fx            = ycams[i]["fx"].as<double>();
            c.fy            = ycams[i]["fy"].as<double>();
            c.cx            = ycams[i]["cx"].as<double>();
            c.cy            = ycams[i]["cy"].as<double>();
            c.depth_scale   = ycams[i]["depth_scale"].as<double>();

            auto m = ycams[i]["extrinsic"]["matrix"].as<std::vector<double>>();
            if (m.size()==16) {
                Eigen::Matrix4d T;
                for (int r=0;r<4;++r)
                    for (int d=0;d<4;++d)
                        T(r,d) = m[r*4+d];
                c.T_cam_world = T;
            }
            out.push_back(c);
        }
        return out;
    }

    ColorSegmentationConfig loadColorConfig(const std::string &p) const {
        ColorSegmentationConfig cfg;
        YAML::Node root = YAML::LoadFile(p);
        auto lower = root["object_hsv"]["lower"].as<std::vector<int>>();
        auto upper = root["object_hsv"]["upper"].as<std::vector<int>>();
        cfg.hsv_lower = cv::Scalar(lower[0],lower[1],lower[2]);
        cfg.hsv_upper = cv::Scalar(upper[0],upper[1],upper[2]);
        cfg.morph_kernel_size = root["morphology"]["kernel_size"].as<int>();
        return cfg;
    }

    FiducialConfig loadMarkerConfig(const std::string &p) const {
        FiducialConfig cfg;
        YAML::Node root = YAML::LoadFile(p);
        cfg.marker_length_m = root["marker_length_m"].as<double>();
        std::string dn = root["dictionary"].as<std::string>();
        if (dn=="DICT_4X4_50") cfg.dictionary_id = cv::aruco::DICT_4X4_50;
        return cfg;
    }
};

} // namespace mv_rgbd_vision

int main(int argc,char** argv){
    using namespace mv_rgbd_vision;
    if (argc<5){
        std::cerr<<"usage: "<<argv[0]<<" <dataset_root> <camera_cfg> <color_cfg> <marker_cfg>\n";
        return 1;
    }
    try{
        Pipeline p(argv[1],argv[2],argv[3],argv[4]);
        p.run();
    }catch(std::exception &e){
        std::cerr<<"err: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
