#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


int main()
{
    Eigen::Tensor<int, 4> a(1, 4, 2, 3);
    a.setValues({ { { {0, 100, 200}, {0, 100, 200} }, { {300, 400, 500}, {300, 400, 500} },
                    { {600, 700, 800}, {600, 700, 800} }, { {900, 1000, 1100}, {900, 1000, 1100} } } });

    for (int o = 0; o < a.dimension(0); o++) {
        for (int i = 0; i < a.dimension(1); i++) {
            for (int j = 0; j < a.dimension(2); j++) {
                for (int k = 0; k < a.dimension(3); k++) {
                    std::cout << a(o, i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "------------------" << std::endl;
        std::cout << std::endl;
    }

    Eigen::array<Eigen::DenseIndex, 4> offsets = { 0, 0, 0, 0 };
    Eigen::array<Eigen::DenseIndex, 4> extents = { 1, 2, 2, 3 };
    Eigen::Tensor<int, 4> slice = a.slice(offsets, extents);

    // std::cout << "slice:" << std::endl << slice << std::endl;

    for (int o = 0; o < slice.dimension(0); o++) {
        for (int i = 0; i < slice.dimension(1); i++) {
            for (int j = 0; j < slice.dimension(2); j++) {
                for (int k = 0; k < slice.dimension(3); k++) {
                    std::cout << slice(o, i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "------------------" << std::endl;
        std::cout << std::endl;
    }

    Eigen::Tensor<float, 2> aa(3, 4);
    aa.setConstant(0.3f);

    Eigen::VectorXf a_;
    a_.resize(4);
    a_ << 1, 2, 3, 4;

    // double* a_array = a_.data();
    // for(int i = 0; i < 4; i++)
    // {
    //     std::cout << a_array[i] << std::endl;
    // }
    Eigen::Tensor<float, 2> b = Eigen::TensorMap<Eigen::Tensor<float, 2>>(a_.data(), 2, 2);

    std::cout << b << std::endl;


}