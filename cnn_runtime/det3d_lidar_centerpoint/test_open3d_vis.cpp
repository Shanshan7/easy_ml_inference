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


// int main(int argc, char *argv[]) {
//     int LoadDim = 5;

//     std::ifstream read_txt;
//     std::string line_data;
//     read_txt.open("/docker_data/data/nuscenes/samples/lidar.txt");
//     if(!read_txt.is_open()){
//         std::cout << "Not exits" << std::endl;
//         return -1;
//     }

//     auto cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
//     // open3d::geometry::PointCloud cloud_ptr;
    
//     while(std::getline(read_txt, line_data)){  
//         std::string file_name = "/docker_data/data/nuscenes/samples/LIDAR_TOP/" + line_data;
//         std::cout << file_name << std::endl;

//         std::ifstream embedding(file_name, std::ios::binary|std::ios::in);
//         embedding.seekg(0,std::ios::end);
//         int embedding_length = embedding.tellg() / sizeof(float);
//         embedding.seekg(0, std::ios::beg);
//         float* embedding_coreset = new float[embedding_length];
//         embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * embedding_length);
//         embedding.close();
//         std::cout << "length: " << embedding_length << std::endl;

//         cloud_ptr->points_.clear();
//         cloud_ptr->points_.resize(embedding_length / LoadDim);
//         for(int i = 0; i < embedding_length / LoadDim; i++)
//         {
//             cloud_ptr->points_[i] << embedding_coreset[i * 5], embedding_coreset[i * 5 + 1], 
//                                  embedding_coreset[i * 5 + 2];
//         }

//         // auto cloud_ptr = open3d::io::CreatePointCloudFromFile("/docker_data/easy_ml_inference/cnn_runtime/det3d_lidar_centerpoint/data/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd");
//         open3d::visualization::DrawGeometries({cloud_ptr}, "TestPCDFileFormat", 1920, 1080);
//     }

//     return 0;
// }

int main()
{
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
        std::string file_name = "/docker_data/data/nuscenes/samples/LIDAR_TOP/" + line_data;
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