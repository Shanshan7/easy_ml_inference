#include <iostream>
#include <algorithm>
#include <Eigen/Dense>


int main()
{
    Eigen::MatrixXd array1;
    Eigen::MatrixXd array2;
    Eigen::MatrixXd out_array;

    array1.resize(2,3);
    array2.resize(3,3);

    array1 << 1, 2, 3.1,
              2, 3, 4;
    array2 << 1, 2, 3,
              2, 3, 4,
              2, 3, 4;

    assert(array1.rows()==array2.rows() && array1.cols()==array2.cols());
    // assert('XA and XB must have the same number of columns ');

    int array1_row = array1.rows();
    int array2_row = array2.rows();
    out_array.resize(array1_row, array2_row);

    for (int i = 0; i < array1_row; i++)
    {
        for (int j = 0; j < array2_row; j++)
        {
            Eigen::VectorXd array1_row_ele;
            array1_row_ele.resize(array1_row);
            array1_row_ele = array1.row(i);

            Eigen::VectorXd array2_col_ele;
            array2_col_ele.resize(array2_row);
            array2_col_ele = array2.transpose().col(j);

            Eigen::VectorXd out_array_ele;
            out_array_ele.resize(array1_row);
            out_array_ele = array1_row_ele - array2_col_ele;

            out_array(i, j) = sqrt(abs(out_array_ele.array().square().sum()));
        }
    }

    for (int m = 0; m < out_array.rows(); m++)
    {
        for (int n = 0; n < out_array.cols(); n++)
        {
            std::cout << "ele: " << m << " " << n << " " << out_array(m,n) << std::endl; 
        }
    }
}
