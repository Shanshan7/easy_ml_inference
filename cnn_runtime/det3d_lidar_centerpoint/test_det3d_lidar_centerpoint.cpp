// headers in STL
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>

#include "open3d/Open3D.h"
#include "centerpoint/centerpoint.h"


void Getinfo(void) {
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
    printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2]);
  }
  printf("\n");
}

int Txt2Arrary(float *&points_array, std::string file_name,
               int num_feature = 4) {
  std::ifstream in_file;
  in_file.open(file_name.data());
  assert(in_file.is_open());

  std::vector<float> temp_points;
  std::string c;

  while (!in_file.eof()) {
    in_file >> c;
    temp_points.push_back(atof(c.c_str()));
  }
  points_array = new float[temp_points.size()];
  for (size_t i = 0; i < temp_points.size(); ++i) {
    points_array[i] = temp_points[i];
  }

  in_file.close();
  return temp_points.size() / num_feature;
};

int Bin2Arrary(float *&points_array, std::string file_name,
               int in_num_feature = 4, int out_num_feature = 4) {
  std::ifstream in_file;
  in_file.open(file_name.data(), ios::binary);
  assert(in_file.is_open());
  std::vector<float> temp_points;
  float f;

  while (!in_file.eof()) {
    in_file.read((char *)&f, sizeof(f));
    temp_points.push_back(f);
  }
  points_array = new float[temp_points.size()];
  int size = temp_points.size() / in_num_feature;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < out_num_feature; ++j) {
      points_array[i * out_num_feature + j] =
          temp_points[i * in_num_feature + j];
    }
  }

  in_file.close();
  return size;
};

void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name) {
  std::ofstream out_file;
  out_file.open(file_name, std::ios::out);
  if (out_file.is_open()) {
    for (const auto &box : boxes) {
      out_file << box.x << " ";
      out_file << box.y << " ";
      out_file << box.z << " ";
      out_file << box.l << " ";
      out_file << box.w << " ";
      out_file << box.h << " ";
      out_file << box.r << " ";
      out_file << box.score << " ";
      out_file << box.label << "\n";
    }
  }
  out_file.close();
  return;
};

void load_anchors(float *&anchor_data, std::string file_name) {
  std::ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  std::vector<float> temp_points;
  std::string c;

  while (!InFile.eof()) {
    InFile >> c;
    temp_points.push_back(atof(c.c_str()));
  }
  anchor_data = new float[temp_points.size()];
  for (size_t i = 0; i < temp_points.size(); ++i) {
    anchor_data[i] = temp_points[i];
  }
  InFile.close();
  return;
}

void draw_bboxes(open3d::visualization::Visualizer &vis, 
                 std::shared_ptr<open3d::geometry::PointCloud> &points3d,
                 std::vector<Box> &out_detections)
{
    int rot_axis = 2;
    int out_detections_size = out_detections.size();
    // in_box_color = np.array(points_in_box_color)
    // points_in_box_color=(1, 0, 0),

    for (int i = 0; i < out_detections_size; i++)
    {
        Eigen::Vector3d center(out_detections[i].x, out_detections[i].y, out_detections[i].z);
        Eigen::Vector3d dim(out_detections[i].l, out_detections[i].w, out_detections[i].h);
        double yaw = out_detections[i].r;
        Eigen::Matrix3d rot_mat = open3d::utility::RotationMatrixY(yaw);

        center(rot_axis) += dim(rot_axis) / 2;  // bottom center to gravity center
        auto box3d = open3d::geometry::OrientedBoundingBox(center, rot_mat, dim);

        auto line_set = open3d::geometry::LineSet::CreateFromOrientedBoundingBox(box3d);
        // add direction
        line_set->lines_.push_back(Eigen::Vector2i(1, 4));
        line_set->lines_.push_back(Eigen::Vector2i(6, 7));
        line_set->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
        // draw bboxes on visualizer
        vis.AddGeometry(line_set);

        // add arrow-like line
        Eigen::Vector3d offset(dim(0) / 2.0 * cos(yaw), dim(0) / 2.0 * sin(yaw), 0.0);
        auto arrow_set = std::make_shared<open3d::geometry::LineSet>();
        arrow_set->points_.push_back(center);
        arrow_set->points_.push_back(center + offset);
        arrow_set->lines_.push_back(Eigen::Vector2i(0, 1));
        arrow_set->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
        vis.AddGeometry(arrow_set);
        // change the color of points which are in box
        // std::vector<size_t> indices = box3d.GetPointIndicesWithinBoundingBox(point3d->points_);
        // points_colors[indices] = in_box_color
    }
    // update points colors
    // point3d->colors_ = ;
    // vis.UpdateGeometry(point3d);
}

void test(void) {
  const std::string DB_CONF = "../bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);

  std::string pfe_file = config["PfeFile"].as<std::string>();
  std::string backbone_file = config["BackboneFile"].as<std::string>();
  std::string model_config = config["ModelConfig"].as<std::string>();
  // std::string file_name = config["InputFile"].as<std::string>();

  std::cout << "pfe_file: " << pfe_file << std::endl;
  std::cout << "backbone_file: " << backbone_file << std::endl;
  std::cout << "config: " << model_config << std::endl;
  // std::cout << "data: " << file_name << std::endl;

  // 可视化初始化
  open3d::visualization::Visualizer vis;
  vis.CreateVisualizerWindow("Open3D", 1920, 1080);
  // 调整背景颜色
  auto mesh_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, Eigen::Vector3d(0, 0, 0));
  vis.AddGeometry(mesh_frame);

  auto points3d = std::make_shared<open3d::geometry::PointCloud>();
  // 设置点云大小，背景颜色
  vis.GetRenderOption().point_size_ = 1;
  vis.GetRenderOption().background_color_ = {0.05, 0.05, 0.05};
  vis.GetRenderOption().show_coordinate_frame_ = true;
  vis.AddGeometry(points3d);
  bool to_reset = true;

  std::ifstream read_txt;
  std::string line_data;
  read_txt.open("/docker_data/data/nuscenes/samples/lidar.txt");
  if(!read_txt.is_open()){
      std::cout << "Not exits" << std::endl;
      return;
  }
  
  while(std::getline(read_txt, line_data)){  
    std::string file_name = "/docker_data/data/nuscenes/samples/LIDAR_TOP/" + line_data;
    std::cout << file_name << std::endl;

    CenterPoint cp(config, pfe_file, backbone_file, model_config);

    float *points_array;
    int in_num_points;
    in_num_points =
        Bin2Arrary(points_array, file_name, config["LoadDim"].as<int>(),
                  config["UseDim"].as<int>());
    std::cout << "num points: " << in_num_points << std::endl;

    std::vector<Box> out_detections;
    cudaDeviceSynchronize();
    cp.DoInference(points_array, in_num_points, out_detections);
    cudaDeviceSynchronize();
    size_t num_objects = out_detections.size();
    std::cout << "detected objects: " << num_objects << std::endl;

    // 点云数据写入open3d
    int UseDim = config["UseDim"].as<int>();
    points3d->points_.clear();
    points3d->points_.resize(in_num_points);
    for(int i = 0; i < in_num_points; i++)
    {
        points3d->points_[i] << points_array[i * UseDim], points_array[i * UseDim + 1], 
                              points_array[i * UseDim + 2];
    }
    vis.UpdateGeometry(points3d);
    draw_bboxes(vis, points3d, out_detections);

    if(to_reset)
    {
        vis.ResetViewPoint(true);
        to_reset = false;
    }
    vis.PollEvents();
    vis.UpdateRender();
    // vis.Run();

    // 检测结果写入文件
    // std::string boxes_file_name = config["OutputFile"].as<std::string>();
    // Boxes2Txt(out_detections, boxes_file_name);

    delete[] points_array;
  }
  vis.DestroyVisualizerWindow();
};

int main(int argc, char **argv) {
  Getinfo();
  test();
  // show_result_meshlab();
}
