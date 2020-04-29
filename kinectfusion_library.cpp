#include <SLAMBenchAPI.h>
#include <cpptoml.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/DepthSensor.h>
#include <kinectfusion.h>
#include <util.h>

#include <opencv/cv.hpp>
#include <opencv2/core/mat.hpp>

// Define parameters
std::string config_file, default_config_file = "/home/mihai/Projects/slambench4/benchmarks/kinectfusion/src/original/KinectFusionApp/config.toml";

// Define sensors
slambench::io::CameraSensor *rgb_sensor;
slambench::io::DepthSensor *depth_sensor;
kinectfusion::CameraParameters cam_params;

// Define outputs
slambench::outputs::Output *pose_output;
slambench::outputs::Output *pointcloud_output;
slambench::outputs::Output *rgb_frame_output;
slambench::outputs::Output *depth_frame_output;
slambench::outputs::Output *reconstructed_frame_output;

// Define OpenCV image to hold data
cv::Mat *cv_rgb;
cv::Mat *cv_depth;

kinectfusion::GlobalConfiguration configuration;
kinectfusion::Pipeline* pipeline;
/**
 * Setup all the outputs for this SLAM algorithm
 * @param slam_settings
 */
void setup_outputs(SLAMBenchLibraryHelper *slam_settings)
{
  pose_output = new slambench::outputs::Output("KinectFusion Pose", slambench::values::VT_POSE, true);
  slam_settings->GetOutputManager().RegisterOutput(pose_output);

  rgb_frame_output = new slambench::outputs::Output("KinectFusion RGB input", slambench::values::VT_FRAME);
  rgb_frame_output->SetKeepOnlyMostRecent(true);
  slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);

  depth_frame_output = new slambench::outputs::Output("KinectFusion Depth input", slambench::values::VT_FRAME);
  depth_frame_output->SetKeepOnlyMostRecent(true);
  slam_settings->GetOutputManager().RegisterOutput(depth_frame_output);

  reconstructed_frame_output = new slambench::outputs::Output("KinectFusion Reconstructed Frame", slambench::values::VT_FRAME);
  reconstructed_frame_output->SetKeepOnlyMostRecent(true);
  slam_settings->GetOutputManager().RegisterOutput(reconstructed_frame_output);

  pointcloud_output = new slambench::outputs::Output("KinectFusion PointCloud", slambench::values::VT_COLOUREDPOINTCLOUD, true);
  slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);
  pointcloud_output->SetKeepOnlyMostRecent(true);
}

/**
 * Setup all sensors for this SLAM algorithm
 * @param slam_settings
 * @return
 */
bool setup_sensors(SLAMBenchLibraryHelper *slam_settings)
{
  slambench::io::CameraSensorFinder sensor_finder;
  rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
  if(rgb_sensor == nullptr)
  {
    std::cerr << "RGB sensor not found" << std::endl;
    return false;
  }
  if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888)
  {
    std::cerr << "RGB sensor is not in RGB_III_888 format" << std::endl;
    return false;
  }
  if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster)
  {
    std::cerr << "RGB sensor is not in raster format" << std::endl;
    return false;
  }

  depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});
  if(depth_sensor == nullptr)
  {
    std::cerr << "Depth sensor not found" << std::endl;
    return false;
  }
  if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
    std::cerr << "Depth data is in wrong pixel format" << std::endl;
    return false;
  }
  if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
    std::cerr << "Depth data is in wrong format" << std::endl;
    return false;
  }

  // KinectFusion requires that depth and height are the same for colour and depth images:
  if(depth_sensor->Height != rgb_sensor->Height || depth_sensor->Width != rgb_sensor->Width) {
    std::cerr << "Colour and depth sensor sizes are mismatched!" << std::endl;
    return false;
  }
  cv_depth = new cv::Mat(depth_sensor->Height, depth_sensor->Width, CV_16UC1);
  cv_rgb = new cv::Mat(rgb_sensor->Height, rgb_sensor->Width, CV_8UC3);

  cam_params.focal_x = depth_sensor->Intrinsics[0] * depth_sensor->Width;
  cam_params.focal_y = depth_sensor->Intrinsics[1] * depth_sensor->Height;
  cam_params.principal_x = depth_sensor->Intrinsics[2] * depth_sensor->Width;
  cam_params.principal_y = depth_sensor->Intrinsics[3] * depth_sensor->Height;
  cam_params.image_width = depth_sensor->Width;
  cam_params.image_height= depth_sensor->Height;

  return true;
}

void make_configuration(const std::shared_ptr<cpptoml::table>& toml_config)
{
  // cpptoml only supports int64_t, so we need to explicitly cast to int to suppress the warning
  auto volume_size_values = *toml_config->get_qualified_array_of<int64_t>("kinectfusion.volume_size");
  configuration.volume_size = make_int3(static_cast<int>(volume_size_values[0]),
                                        static_cast<int>(volume_size_values[1]),
                                        static_cast<int>(volume_size_values[2]));
  configuration.voxel_scale = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.voxel_scale"));
  configuration.bfilter_kernel_size = *toml_config->get_qualified_as<int>("kinectfusion.bfilter_kernel_size");
  configuration.bfilter_color_sigma  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.bfilter_color_sigma"));
  configuration.bfilter_spatial_sigma  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.bfilter_spatial_sigma"));
  configuration.init_depth  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.init_depth"));
  configuration.use_output_frame = *toml_config->get_qualified_as<bool>("kinectfusion.use_output_frame");
  configuration.truncation_distance  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.truncation_distance"));
  configuration.depth_cutoff_distance  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.depth_cutoff_distance"));
  configuration.num_levels  = *toml_config->get_qualified_as<int>("kinectfusion.num_levels");
  configuration.triangles_buffer_size  = *toml_config->get_qualified_as<int>("kinectfusion.triangles_buffer_size");
  configuration.pointcloud_buffer_size  = *toml_config->get_qualified_as<int>("kinectfusion.pointcloud_buffer_size");
  configuration.distance_threshold  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.distance_threshold"));
  configuration.angle_threshold  = static_cast<float>(*toml_config->get_qualified_as<double>("kinectfusion.angle_threshold"));
  auto icp_iterations_values = *toml_config->get_qualified_array_of<int64_t>("kinectfusion.icp_iterations");
  configuration.icp_iterations = {icp_iterations_values.begin(), icp_iterations_values.end()};
}

/**
 * Declare all command line parameters here
 * @param slam_settings
 * @return
 */
bool sb_new_slam_configuration(SLAMBenchLibraryHelper *slam_settings)
{
  // Declare parameters
  slam_settings->addParameter(TypedParameter<std::string>("c", "config", "Config file path", &config_file, &default_config_file));
  return true;
}

/**
 * Setup all the sensors, outputs, and initialize the SLAM system
 * @param slam_settings
 * @return Whether or not the initialisation is successful
 */
bool sb_init_slam_system(SLAMBenchLibraryHelper *slam_settings)
{
  setup_outputs(slam_settings);
  if(!setup_sensors(slam_settings))
    return false;
  auto toml_config = cpptoml::parse_file(config_file);
  make_configuration(toml_config);
  pipeline = new kinectfusion::Pipeline(cam_params, configuration);
  return true;
}

/**
 * Receive a new frame from the sensors
 * @param frame new input frame (from any sensor type, not necessarily camera)
 * @return true if the frame matches any of the algorithm's sensors, false otherwise
 */
bool depth_ready = false, rgb_ready = false;
bool sb_update_frame(SLAMBenchLibraryHelper*, slambench::io::SLAMFrame *frame)
{
  if(frame->FrameSensor == rgb_sensor)
  {
    memcpy(cv_rgb->data, frame->GetData(), frame->GetSize());
    rgb_ready = true;
    frame->FreeData();
  }
  else if(frame->FrameSensor == depth_sensor) {
    memcpy(cv_depth->data, frame->GetData(), frame->GetSize());
    depth_ready = true;
    frame->FreeData();
  }
  if (depth_ready && rgb_ready) {
    depth_ready = false;
    rgb_ready = false;
    return true;
  }
  return false;
}

/**
 * Execute SLAM pipeline with
 * @param slam_settings
 * @return Whether or not the execution was successful
 */
bool sb_process_once(SLAMBenchLibraryHelper *slam_settings)
{
  cv::Mat depth_map;
  cv_depth->convertTo(depth_map, CV_32FC1, 1/32767.0);
  pipeline->process_frame(depth_map, *cv_rgb);
  return true;
}

/**
 *
 * @param lib
 * @param ts current time stamp
 * @return
 */
bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *ts)
{
  if(pose_output->IsActive())
  {
    Eigen::Matrix4f mat = pipeline->get_last_pose();
    std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
    pose_output->AddPoint(*ts, new slambench::values::PoseValue(mat));
  }
  if(pointcloud_output->IsActive())
  {
    kinectfusion::PointCloud pcl = pipeline->extract_pointcloud();
    auto pointcloud_value = new slambench::values::ColoredPointCloudValue();
    for(int i = 0; i < pcl.num_points; i++)
    {
      slambench::values::ColoredPoint3DF new_vertex(pcl.vertices.at<float>(i,0), pcl.vertices.at<float>(i,1), pcl.vertices.at<float>(i,2));
      new_vertex.R = pcl.color.at<float>(i,0);
      new_vertex.G = pcl.color.at<float>(i,1);
      new_vertex.B = pcl.color.at<float>(i,2);
      pointcloud_value->AddPoint(new_vertex);
    }

    std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
    pointcloud_output->AddPoint(*ts, pointcloud_value);
  }
  if(rgb_frame_output->IsActive())
  {
    std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
    rgb_frame_output->AddPoint(*ts, new slambench::values::FrameValue(cam_params.image_width, cam_params.image_height, slambench::io::pixelformat::RGB_III_888, (void*)(cv_rgb->data)));
  }
  if(depth_frame_output->IsActive())
  {
    std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
    depth_frame_output->AddPoint(*ts, new slambench::values::FrameValue(cam_params.image_width, cam_params.image_height, slambench::io::pixelformat::D_I_16, (void*)(cv_depth->data)));
  }
  if(reconstructed_frame_output->IsActive())
  {
    std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
    reconstructed_frame_output->AddPoint(*ts, new slambench::values::FrameValue(cam_params.image_width, cam_params.image_height, slambench::io::pixelformat::G_I_8, (void*)(pipeline->get_last_model_frame().data)));
  }

  return true;
}

/**
 * Clean up any allocated global pointers
 * @return
 */
bool sb_clean_slam_system()
{
  delete depth_sensor;
  delete rgb_sensor;
  delete pointcloud_output;
  delete rgb_frame_output;
  delete depth_frame_output;
  delete reconstructed_frame_output;
  delete pose_output;
  delete cv_rgb;
  delete cv_depth;
  return true;
}
