#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


void RepeatImpl(Eigen::Tensor<float, 4> &input_eigen, std::vector<int> &repeat_shape,
                             Eigen::Tensor<float, 4> &output_eigen)
{
    const Eigen::Tensor<float, 4>::Dimensions & input_shape = input_eigen.dimensions();
    output_eigen = Eigen::Tensor<float, 4>(input_shape[0]*repeat_shape[0], input_shape[1]*repeat_shape[1],
                 input_shape[2]*repeat_shape[2], input_shape[3]*repeat_shape[3]);

    for(int l = 0; l < input_shape[0]*repeat_shape[0]; l++)
    {
        for(int k = 0; k < input_shape[1]*repeat_shape[1]; k++)
        {
            for(int j = 0; j < input_shape[2]*repeat_shape[2]; j++)
            {
                for(int i = 0; i < input_shape[3]*repeat_shape[3]; i++)
                {
                    output_eigen(l, k, j, i) = input_eigen(l%input_shape[0], k%input_shape[1],
                                                        j%input_shape[2], i%input_shape[3]);
                }
            }
        }
    }
}

int main()
{
    float traj[] = {1, 2, 3, 4};
    // auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(&traj[0], 1, 4, 1, 1);
    // Eigen::Tensor<float, 4> t = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 4, Eigen::ColMajor>>(mapped_t);
    auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::ColMajor>>(&traj[0], 1, 4, 1, 1);
    Eigen::Tensor<float, 4> t = mapped_t; 
    // std::cout << "n, c, h, w: " << t.dimension(0) << " " << t.dimension(1) << " " 
    //           << t.dimension(2) << " " << t.dimension(3) << std::endl;

    std::vector<int> repeat_shape = {2, 2, 2, 2};
    Eigen::Tensor<float, 4> output_eigen;
    // output_eigen = Eigen::Tensor<float, 4>(t.dimension(0)*repeat_shape[0], t.dimension(1)*repeat_shape[1],
    //              t.dimension(2)*repeat_shape[2], t.dimension(3)*repeat_shape[3]);
    // std::cout << "n, c, h, w: " << output_eigen.dimension(0) << " " << output_eigen.dimension(1) << " " 
    //         << output_eigen.dimension(2) << " " << output_eigen.dimension(3) << std::endl;
    RepeatImpl(t, repeat_shape, output_eigen);

    for (int o = 0; o < output_eigen.dimension(0); o++) {
        for (int i = 0; i < output_eigen.dimension(1); i++) {
            for (int j = 0; j < output_eigen.dimension(2); j++) {
                for (int k = 0; k < output_eigen.dimension(3); k++) {
                    std::cout << output_eigen(o, i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "------------------" << std::endl;
        std::cout << std::endl;
    }

}