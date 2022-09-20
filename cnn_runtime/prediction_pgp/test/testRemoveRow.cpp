#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


void RemoveRow(Eigen::MatrixXd& eigen_matrix, unsigned int row_to_remove) 
{
  unsigned int num_rows = eigen_matrix.rows() - 1;
  unsigned int num_cols = eigen_matrix.cols();
 
  if(row_to_remove < num_rows) {
    eigen_matrix.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) =
        eigen_matrix.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
  }
 
  eigen_matrix.conservativeResize(num_rows, num_cols);
}

void RemoveRow(Eigen::VectorXd& eigen_vector, unsigned int row_to_remove) 
{
  unsigned int num_rows = eigen_vector.rows() - 1;
  unsigned int num_cols = eigen_vector.cols();
 
  if(row_to_remove < num_rows) {
    eigen_vector.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) =
        eigen_vector.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
  }
 
  eigen_vector.conservativeResize(num_rows);
}

int main()
{
    // Eigen::MatrixXd array1;

    // array1.resize(3,3);
    // array1 << 1, 2, 3,
    //           2, 3, 4,
    //           2, 3, 4;

    // std::cout << "r, w: " << array1.rows() << " " << array1.cols() << std::endl;
    // RemoveRow(array1, 0);
    // std::cout << "r, w: " << array1.rows() << " " << array1.cols() << std::endl;

    // Eigen::VectorXd vector1;
    // vector1.resize(5);
    // vector1 << 1, 2, 3, 4, 5;
    // // std::cout << vector1.size() << std::endl;
    // RemoveRow(vector1, 3);
    // for(int i = 0; i < vector1.size(); i++)
    // {
    //     std::cout << vector1(i) << " ";
    // }
    // std::cout << "r, w: " << vector1.rows() << " " << vector1.cols() << std::endl;

    // float traj[] = {1, 2, 3, 4, 5, 6, 7, 8};
    // Eigen::Tensor<float, 3> traj_eigen(2, 2, 2);
    // traj_eigen = Eigen::Map<Eigen::Tensor<float, 3>>(traj, 2, 2, 2);

    float traj[] = {1, 3, 3, 4, 5, 6, 7, 9};
    auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(&traj[0], 2, 2, 2);
    Eigen::Tensor<float, 3> t = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 3, Eigen::RowMajor>>(mapped_t);
    std::cout << "c, h, w: " << t.dimension(0) << " " << t.dimension(1) << " " << t.dimension(2) << std::endl;

    return 0;
}