// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <string>
#include <fstream>

#include "open3d/Open3D.h"
#include "centerpoint/centerpoint.h"


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

void load_txt_results(std::string file_name, std::vector<Box> &out_detections)
{
    std::ifstream in_file;
    in_file.open(file_name.data());
    assert(in_file.is_open());

    std::vector<float> temp_points;
    std::string c;

    while (!in_file.eof()) {
        in_file >> c;
        temp_points.push_back(atof(c.c_str()));
    }
    
    for (size_t i = 0; i < temp_points.size() / 9; ++i) {
        Box temp_box;
        temp_box.x = temp_points[i*9];
        temp_box.y = temp_points[i*9+1];
        temp_box.z = temp_points[i*9+2];
        temp_box.l = temp_points[i*9+3];
        temp_box.w = temp_points[i*9+4];
        temp_box.h = temp_points[i*9+5];
        temp_box.r = temp_points[i*9+6];
        temp_box.score = temp_points[i*9+7];
        temp_box.label = temp_points[i*9+8];
        out_detections.push_back(temp_box);
    }
    in_file.close();
}

int main()
{
    std::vector<Box> out_detections;
    load_txt_results("/docker_data/easy_ml_inference/cnn_runtime/det3d_lidar_centerpoint/data/boxes.txt",
                     out_detections);

    int LoadDim = 5;

    std::ifstream read_txt;
    std::string line_data;
    read_txt.open("/docker_data/data/nuscenes/samples/lidar.txt");
    if(!read_txt.is_open()){
        std::cout << "Not exits" << std::endl;
        return -1;
    }

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Open3D", 1920, 1080);

    Eigen::Vector3d origin;
    origin << 0, 0, 0;
    auto mesh_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0, origin);
    vis.AddGeometry(mesh_frame);

    auto points3d = std::make_shared<open3d::geometry::PointCloud>();
    // 设置点云大小，背景颜色
    vis.GetRenderOption().point_size_ = 1;
    vis.GetRenderOption().background_color_ = {0.05, 0.05, 0.05};
    vis.GetRenderOption().show_coordinate_frame_ = true;
    vis.AddGeometry(points3d);

    bool to_reset = true;
    while(std::getline(read_txt, line_data)){  
        // std::string file_name = "/docker_data/data/nuscenes/samples/LIDAR_TOP/" + line_data;
        std::string file_name = "/docker_data/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin";
        std::cout << file_name << std::endl;

        std::ifstream embedding(file_name, std::ios::binary|std::ios::in);
        embedding.seekg(0,std::ios::end);
        int embedding_length = embedding.tellg() / sizeof(float);
        embedding.seekg(0, std::ios::beg);
        float* embedding_coreset = new float[embedding_length];
        embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * embedding_length);
        embedding.close();

        points3d->points_.clear();
        points3d->points_.resize(embedding_length / LoadDim);
        for(int i = 0; i < embedding_length / LoadDim; i++)
        {
            points3d->points_[i] << embedding_coreset[i * 5], embedding_coreset[i * 5 + 1], 
                                 embedding_coreset[i * 5 + 2];
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
        vis.Run();
    }
    vis.DestroyVisualizerWindow();
}