/****************************************************************************************
 *
 * Copyright (c) 2024, Shenyang Institute of Automation, Chinese Academy of Sciences
 *
 * Authors: Yanpeng Jia
 * Contact: jiayanpeng@sia.cn
 *
 ****************************************************************************************/

#include "util.h"
#include "tictoc.h"

class ImageProjection
{
private:
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;

    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    image_transport::Publisher pubRangeImageNow;
    image_transport::Publisher pubRangeImageLast;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud;     // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;
    pcl::PointCloud<PointType>::Ptr lastCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat;  // range matrix for range image
    cv::Mat labelMat;  // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t>> neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection() : nh("~")
    {

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2>("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2>("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info>("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);

        image_transport::ImageTransport it(nh);
        pubRangeImageNow = it.advertise("/range_image_now", 10);
        pubRangeImageLast = it.advertise("/range_image_last", 10);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    void allocateMemory()
    {

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        lastCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN * Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN * Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN * Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN * Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = 1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 0;
        neighbor.second = -1;
        neighborIterator.push_back(neighbor);
        neighbor.first = 1;
        neighbor.second = 0;
        neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN * Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN * Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN * Horizon_SCAN];
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection() {}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

        cloudHeader = laserCloudMsg->header;
        cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
        // have "ring" channel in the cloud
        if (useCloudRing == true)
        {
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);
            if (laserCloudInRing->is_dense == false)
            {
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }
        }
    }

    // project pointcloud to range image
    cv::Mat projectPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_in, cv::Mat &range_image_idx)
    {
        // 统一用deg度作为单位
        int kNumRimgRow = std::round(V_FOV * image_res);
        int kNumRimgCol = std::round(H_FOV * image_res);

        // std::vector<float> image_means{12.12, 10.88, 0.23, -1.04, 0.21};
        // std::vector<float> image_stds{12.32, 11.47, 6.91, 0.86, 0.16};
        // float* range_images = new float[1 * kNumRimgRow * kNumRimgCol]();
        // float* range_images_idx = new float[1 * kNumRimgRow * kNumRimgCol]();
        cv::Mat range_image = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(100));
        // cv::Mat range_image_idx = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32SC1, cv::Scalar::all(0));

        int num_points = pcl_in->points.size();
        for (int idx = 0; idx < num_points; idx++)
        {
            const auto &x = pcl_in->points[idx].x;
            const auto &y = pcl_in->points[idx].y;
            const auto &z = pcl_in->points[idx].z;
            // 球面投影算出来的是弧度rad
            const float range = std::sqrt(x * x + y * y + z * z);
            const float yaw = std::atan2(y, x);
            const float pitch = std::atan2(z, std::sqrt(x * x + y * y));

            float proj_row = (1.0f - (pitch * 180.0 / M_PI + std::abs(V_FOV_DOWN)) / (V_FOV - float(0.0))) * kNumRimgRow;
            float proj_col = ((yaw * 180.0 / M_PI + (H_FOV / float(2.0))) / (H_FOV - float(0.0))) * kNumRimgCol;
            proj_row = std::round(proj_row);
            proj_col = std::round(proj_col);
            const int v = clamp<int>(static_cast<int>(proj_row), 0, kNumRimgRow - 1);
            const int u = clamp<int>(static_cast<int>(proj_col), 0, kNumRimgCol - 1);

            // range_image.at<float>(v, u) = range;
            // range_image_idx.at<float>(v, u) = idx;
            // if(range < range_images[0 * width * height + v * width + u]){
            //     range_images[0 * width * height + v * width + u] = range;
            //     range_images_idx[0 * width * height + v * width + u] = idx;
            // }
            if (range < range_image.at<float>(v, u))
            {
                range_image.at<float>(v, u) = range;
                range_image_idx.at<int>(v, u) = idx;
                // range_image_idx.at<cv::Vec3i>(v, u)[c] = idx;
                // c = c < channel - 1 ? c++ : c;
                // range_images_idx[0 * kNumRimgRow * kNumRimgCol + v * kNumRimgCol + u] = idx;
            }
            // range_images[0 * width * height + v * width + u] = range;
            // (range - image_means.at(0)) / image_stds.at(0);
            // range_images_idx[0 * width * height + v * width + u] = idx;
            // range_images[1 * width * height + v * width + u] =
            //     (x - image_means.at(1)) / image_stds.at(1);
            // range_images[2 * width * height + v * width + u] =
            //     (y - image_means.at(2)) / image_stds.at(2);
            // range_images[3 * width * height + v * width + u] =
            //     (z - image_means.at(3)) / image_stds.at(3);
            // range_images[4 * width * height + v * width + u] =
            //     (intensity - image_means.at(4)) / image_stds.at(4);
        }
        // cv::Mat range_image = cv::Mat(height, width, CV_32FC1, static_cast<void*>(range_images));
        // range_image_idx = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32SC1, static_cast<void*>(range_images_idx));
        // cv::Mat normalized_range, u8_range, color_map;
        // cv::normalize(range_image, normalized_range, 255, 0, cv::NORM_MINMAX);
        // normalized_range.convertTo(u8_range, CV_8UC1);
        // cv::applyColorMap(u8_range, color_map, cv::COLORMAP_JET);

        // return std::pair<cv::Mat, cv::Mat>(rangeMat, rangeMat_idx);
        return range_image;
    }

    std::vector<int> getStaticIdxFromDynamicIdx(const std::vector<int> &dynamic_point_idx, int all_points_number)
    {
        std::vector<int> pt_idx_all = linspace<int>(0, all_points_number, all_points_number);

        std::set<int> pt_idx_all_set(pt_idx_all.begin(), pt_idx_all.end());
        for (auto &dyna_pt_idx : dynamic_point_idx)
        {
            pt_idx_all_set.erase(dyna_pt_idx);
        }
        std::vector<int> static_point_indexes(pt_idx_all_set.begin(), pt_idx_all_set.end());
        // std::vector<int> static_point_indexes;
        // std::unordered_set<int> dynamic_hash;
        // for(auto &d : dynamic_point_idx){
        //     dynamic_hash.insert(d);
        // }
        // for(int idx = 0; idx < all_points_number; idx++){
        //     if(dynamic_hash.count(idx) != 0){
        //         continue;
        //     }
        //     static_point_indexes.emplace_back(idx);
        // }

        return static_point_indexes;
    }

    void parseStaticMapPointcloudUsingPtIdx(std::vector<int> &point_idx, pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_new_raw, pcl::PointCloud<pcl::PointXYZI>::Ptr &pcl_out_raw)
    {
        // extractor
        pcl::ExtractIndices<pcl::PointXYZI> extractor;
        boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(point_idx);
        extractor.setInputCloud(pcl_new_raw);
        extractor.setIndices(index_ptr);
        extractor.setNegative(false); // If set to true, you can extract point clouds outside the specified index

        // parse
        pcl_out_raw->clear();
        extractor.filter(*pcl_out_raw);
    }

    /**
     * @brief 使用点索引从动态地图点云中提取指定点并生成新点云
     *
     * 根据给定的点索引列表，从输入点云中提取对应的点，输出提取后的点云。
     *
     * @param point_idx 输入的点索引列表
     * @param pcl_new_raw 输入的原始点云（动态地图点云）
     * @param pcl_out_raw 输出的提取后的点云
     */
    void parseDynamicMapPointcloudUsingPtIdx(std::vector<int> &point_idx, pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_new_raw, pcl::PointCloud<pcl::PointXYZI>::Ptr &pcl_out_raw)
    {
        // 定义点云提取器
        pcl::ExtractIndices<pcl::PointXYZI> extractor;

        // 将点索引列表封装为共享指针
        boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(point_idx);

        // 设置提取器的输入点云
        extractor.setInputCloud(pcl_new_raw);

        // 设置提取的点索引
        extractor.setIndices(index_ptr);

        // 设置提取模式
        // 如果设置为 true，则会提取不在指定索引中的点，否则提取在索引中的点
        extractor.setNegative(false);

        // 清空输出点云
        pcl_out_raw->clear();

        // 执行点云提取，将结果存储到输出点云中
        extractor.filter(*pcl_out_raw);
    }

    void removeDynamicObjects()
    {
        static bool first_flag = 1;
        if (first_flag)
        {
            first_flag = 0;
            lastCloud->clear();
            *lastCloud = *laserCloudIn;
            return;
        }
        cv::Mat rangeImage_now_idx = cv::Mat(V_FOV * image_res, H_FOV * image_res, CV_32SC1, cv::Scalar::all(0));
        cv::Mat rangeImage_submap_idx = cv::Mat(V_FOV * image_res, H_FOV * image_res, CV_32SC1, cv::Scalar::all(0));
        cv::Mat rangeImage_now = projectPointCloud(laserCloudIn, rangeImage_now_idx);
        cv::Mat rangeImage_submap = projectPointCloud(lastCloud, rangeImage_submap_idx);
        cv::Mat rangeImage_diff = cv::Mat(V_FOV * image_res, H_FOV * image_res, CV_32FC1, cv::Scalar::all(0.0)); // float matrix, save range value
        cv::absdiff(rangeImage_now, rangeImage_submap, rangeImage_diff);
        std::vector<int> dynamic_point_idx; // 当前帧动态点索引
        for (int row_idx = 0; row_idx < rangeImage_diff.rows; row_idx++)
        {
            for (int col_idx = 0; col_idx < rangeImage_diff.cols; col_idx++)
            {
                float this_diff = rangeImage_diff.at<float>(row_idx, col_idx);
                float this_range = rangeImage_now.at<float>(row_idx, col_idx);
                float adaptive_coeff = 0.5;
                float adaptive_dynamic_descrepancy_threshold = adaptive_coeff * this_range; // adaptive descrepancy threshold
                if (this_diff > adaptive_dynamic_descrepancy_threshold && this_diff < 100)
                {
                    dynamic_point_idx.emplace_back(rangeImage_now_idx.at<int>(row_idx, col_idx));
                }
            }
        }
        // std::cout << "raw pointcloud number is: " << laserCloudIn->points.size() << std::endl;
        // std::cout << "dynamic count number is: " << dynamic_point_idx.size() << std::endl;
        // 计算当前帧静态点索引
        std::vector<int> static_point_idx = getStaticIdxFromDynamicIdx(dynamic_point_idx, laserCloudIn->points.size());
        // std::cout << "static count number is: " << static_point_idx.size() << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_new_out(new pcl::PointCloud<pcl::PointXYZI>);
        parseStaticMapPointcloudUsingPtIdx(static_point_idx, laserCloudIn, pcl_new_out);
        *lastCloud = *laserCloudIn;
        *laserCloudIn = *pcl_new_out;

        // publish range image
        cv::Mat normalized_range, u8_range, color_map;
        sensor_msgs::ImagePtr rangeImage_now_msg, rangeImage_last_msg;
        cv::normalize(rangeImage_now, normalized_range, 255, 0, cv::NORM_MINMAX);
        normalized_range.convertTo(u8_range, CV_8UC1);
        cv::applyColorMap(u8_range, color_map, cv::COLORMAP_JET);
        rangeImage_now_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_map).toImageMsg();
        pubRangeImageNow.publish(rangeImage_now_msg);
        cv::normalize(rangeImage_submap, normalized_range, 255, 0, cv::NORM_MINMAX);
        normalized_range.convertTo(u8_range, CV_8UC1);
        cv::applyColorMap(u8_range, color_map, cv::COLORMAP_JET);
        rangeImage_last_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_map).toImageMsg();
        pubRangeImageLast.publish(rangeImage_last_msg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        // 2. Remove Dynamic Objects
        if (use_dynamic_removal)
        {
            TicToc dynamic_timer(true);
            removeDynamicObjects();
            dynamic_timer.toc("Dynamic Removal Time is: ");
        }
        // 3. Start and end angle of a scan
        findStartEndAngle();
        // 4. Range image projection
        projectPointCloud();
        // 5. Mark ground points
        TicToc ground_removal_timer(true);
        groundRemoval();
        ground_removal_timer.toc("Ground Extract Time is:");
        // 6. Point cloud segmentation
        cloudSegmentation();
        // 7. Publish all clouds
        publishCloud();
        // 8. Reset parameters for next iteration
        resetParameters();
    }

    void findStartEndAngle()
    {
        // start and end orientation of this cloud
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                       laserCloudIn->points[laserCloudIn->points.size() - 1].x) +
                                2 * M_PI;
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI)
        {
            segMsg.endOrientation -= 2 * M_PI;
        }
        else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    void projectPointCloud()
    {
        // 用于将点云数据投影到二维范围图（range image）

        float verticalAngle, horizonAngle, range;   // 存储垂直角度、水平角度和点的距离
        size_t rowIdn, columnIdn, index, cloudSize; // 存储行索引、列索引、点的索引和点云大小
        PointType thisPoint;                        // 存储单个点的坐标和属性

        cloudSize = laserCloudIn->points.size(); // 获取输入点云的点数

        for (size_t i = 0; i < cloudSize; ++i)
        { // 遍历点云中的每个点

            // 读取当前点的坐标
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // 根据激光雷达点云数据格式确定垂直行索引
            if (useCloudRing == true)
            {
                // 如果点云中带有 ring 信息（某些雷达提供环号）
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else
            {
                // 否则根据垂直角度计算行索引
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI; // 计算垂直角度
                rowIdn = N_SCAN - (verticalAngle + ang_bottom) / ang_res_y;                                                   // 计算该点在范围图中的行索引
            }

            // 如果计算得到的行索引不在有效范围内，跳过当前点
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // 计算水平角度，用于确定列索引
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // 根据水平角度计算列索引
            columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;

            // 如果列索引超过范围，将其调整到有效范围内
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 计算点的欧几里得距离
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);

            // 如果距离小于最小传感器范围，跳过该点
            if (range < sensorMinimumRange)
                continue;

            // 将点的距离存入 rangeMat（范围图矩阵）中
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 将当前点的行号和列号合并保存到 intensity 字段（用于后续处理）
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            // 计算该点在一维数组中的索引
            index = columnIdn + rowIdn * Horizon_SCAN;

            // 将该点保存到 fullCloud 和 fullInfoCloud 中
            fullCloud->points[index] = thisPoint;
            fullInfoCloud->points[index] = thisPoint;

            // 在 fullInfoCloud 中保存点的距离信息到 intensity 字段
            fullInfoCloud->points[index].intensity = range;
        }

        // 以下代码被注释掉：用于将 rangeMat 显示为彩色范围图（可视化），并发布图像消息
        /*
        cv::Mat normalized_range, u8_range, color_map;
        sensor_msgs::ImagePtr rangeImage_now_msg;
        cv::normalize(rangeMat, normalized_range, 255, 0, cv::NORM_MINMAX);
        normalized_range.convertTo(u8_range, CV_8UC1);
        cv::applyColorMap(u8_range, color_map, cv::COLORMAP_JET);
        rangeImage_now_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_map).toImageMsg();
        pubRangeImageNow.publish(rangeImage_now_msg);
        */
    }

    void groundRemoval()
    {
        // 使用 TicToc 进行时间统计（已被注释）
        // TicToc ground_time(true);

        // 使用 RANSAC 方法进行地面分割
        if (use_ransac_ground_segment)
        {
            double tg = ros::Time::now().toSec(); // 记录当前时间

            // 候选地面点云和索引
            pcl::PointCloud<pcl::PointXYZI>::Ptr candicate_ground(new pcl::PointCloud<pcl::PointXYZI>);
            std::vector<size_t> index_points;              // 所有候选点的索引
            std::vector<size_t> index_points_ground;       // 当前迭代的内点索引
            std::vector<size_t> index_points_ground_final; // 最终的地面内点索引

            // 遍历所有点，筛选 Z 轴低于 0.5 的点作为地面候选点
            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                for (size_t i = 0; i < groundScanInd; ++i)
                {
                    size_t p_idx = j + i * Horizon_SCAN;

                    // 如果该点无效，跳过
                    if (fullCloud->points[p_idx].intensity == -1)
                    {
                        groundMat.at<int8_t>(i, j) = -1;
                        continue;
                    }

                    // 筛选 Z 轴低于 0.5 的点
                    if (fullCloud->points[p_idx].z < 0.5)
                    {
                        candicate_ground->points.push_back(fullCloud->points[p_idx]);
                        index_points.push_back(p_idx);
                    }
                }
            }

            int num_points = candicate_ground->points.size(); // 候选点数量
            std::unordered_set<int> inliersResult;            // 最终地面内点集合
            int maxIterations = 5;                            // RANSAC 最大迭代次数

            // RANSAC 平面拟合迭代
            while (maxIterations--)
            {
                std::unordered_set<int> inliers;

                // 随机选取 3 个点用于平面拟合
                while (inliers.size() < 3)
                {
                    inliers.insert(rand() % num_points);
                }

                // 提取三个点的坐标
                auto itr = inliers.begin();
                float x1 = candicate_ground->points[*itr].x, y1 = candicate_ground->points[*itr].y, z1 = candicate_ground->points[*itr].z;
                itr++;
                float x2 = candicate_ground->points[*itr].x, y2 = candicate_ground->points[*itr].y, z2 = candicate_ground->points[*itr].z;
                itr++;
                float x3 = candicate_ground->points[*itr].x, y3 = candicate_ground->points[*itr].y, z3 = candicate_ground->points[*itr].z;

                // 计算平面方程参数 ax + by + cz + d = 0
                float a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
                float b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
                float c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
                float d = -(a * x1 + b * y1 + c * z1);
                float sqrt_abc = sqrt(a * a + b * b + c * c);

                // 遍历所有候选点，计算其到平面的距离
                for (int i = 0; i < num_points; i++)
                {
                    if (inliers.count(i) > 0)
                        continue; // 跳过已经选为内点的点

                    pcl::PointXYZI point = candicate_ground->points[i];
                    float dist = fabs(a * point.x + b * point.y + c * point.z + d) / sqrt_abc; // 计算点到平面的距离
                    float distanceTol = 0.3;                                                   // 平面距离容忍度

                    // 如果距离小于阈值，将该点视为内点
                    if (dist < distanceTol)
                    {
                        inliers.insert(i);
                        index_points_ground.push_back(index_points[i]);
                    }

                    // 更新最终的内点集合
                    if (inliers.size() > inliersResult.size())
                    {
                        index_points_ground_final = index_points_ground;
                        inliersResult = inliers;
                    }
                }
            }

            // 将地面和非地面点分开
            pcl::PointCloud<pcl::PointXYZI>::Ptr out_plane(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr in_plane(new pcl::PointCloud<pcl::PointXYZI>);
            for (int i = 0; i < num_points; i++)
            {
                pcl::PointXYZI pt = candicate_ground->points[i];
                if (inliersResult.count(i))
                {
                    in_plane->points.push_back(pt); // 地面点
                }
                else
                {
                    out_plane->points.push_back(pt); // 非地面点
                }
            }

            // 将最终地面点的索引标记为 1
            for (size_t i = 0; i < index_points_ground_final.size(); i++)
            {
                size_t p_col = index_points_ground_final[i] % Horizon_SCAN;
                size_t p_row = index_points_ground_final[i] / Horizon_SCAN;
                groundMat.at<int8_t>(p_row, p_col) = 1;
            }
        }

        // 如果不使用 RANSAC 地面分割，则使用角度判断法
        if (!use_ransac_ground_segment)
        {
            size_t lowerInd, upperInd;
            float diffX, diffY, diffZ, angle;

            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                for (size_t i = 0; i < groundScanInd; ++i)
                {
                    lowerInd = j + i * Horizon_SCAN;
                    upperInd = j + (i + 1) * Horizon_SCAN;

                    if (fullCloud->points[lowerInd].intensity == -1 ||
                        fullCloud->points[upperInd].intensity == -1)
                    {
                        groundMat.at<int8_t>(i, j) = -1;
                        continue;
                    }

                    diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                    diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                    diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                    angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;

                    if (abs(angle - sensorMountAngle) <= (use_new_ground_segment ? 1 : 10))
                    {
                        groundMat.at<int8_t>(i, j) = 1;
                        groundMat.at<int8_t>(i + 1, j) = 1;
                    }
                }
            }
        }

        // 提取地面点云
        for (size_t i = 0; i < N_SCAN; ++i)
        {
            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                if (groundMat.at<int8_t>(i, j) == 1 || rangeMat.at<float>(i, j) == FLT_MAX)
                {
                    labelMat.at<int>(i, j) = -1;
                }
            }
        }

        // 发布地面点云
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            for (size_t i = 0; i <= groundScanInd; ++i)
            {
                for (size_t j = 0; j < Horizon_SCAN; ++j)
                {
                    if (groundMat.at<int8_t>(i, j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation()
    {
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i, j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i)
        {

            segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j)
            {
                if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1)
                {
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i, j) == 999999)
                    {
                        if (i > groundScanInd && j % 5 == 0)
                        {
                            outlierCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                            continue;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    // majority of ground points are skipped
                    if (groundMat.at<int8_t>(i, j) == 1)
                    {
                        if (j % 5 != 0 && j > 5 && j < Horizon_SCAN - 5)
                            continue;
                    }
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
        }

        // extract segmented cloud for visualization
        if (pubSegmentedCloudPure.getNumSubscribers() != 0)
        {
            for (size_t i = 0; i < N_SCAN; ++i)
            {
                for (size_t j = 0; j < Horizon_SCAN; ++j)
                {
                    if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999)
                    {
                        segmentedCloudPure->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);
                    }
                }
            }
        }
    }

    void labelComponents(int row, int col)
    {
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;

        while (queueSize > 0)
        {
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;
            // Loop through all the neighboring grids of popped grid
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter)
            {
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;
                // prevent infinite loop (caused by put already examined point back)
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));

                if (angle > segmentTheta)
                {

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum)
        {
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;
        }
        // segment is valid, mark these points
        if (feasibleSegment == true)
        {
            ++labelCount;
        }
        else
        { // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i)
            {
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    void publishCloud()
    {
        // 1. Publish Seg Cloud Info
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);
        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);
        // segmented cloud with ground
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);
        // projected full cloud
        if (pubFullCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        if (pubSegmentedCloudPure.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info
        if (pubFullInfoCloud.getNumSubscribers() != 0)
        {
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "c_loam");

    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
