#include <iostream>
#include <fstream>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define KNEIGHBOURS (9)
#define LENGTH (2515968)

void preprocess(const cv::Mat &src_mat, const cv::Size dst_size, cv::Mat &dst_image)
{
    if(src_mat.empty()){
        return;
    }

    cv::resize(src_mat, dst_image, dst_size, 0, 0, cv::INTER_LINEAR);

    std::vector<float> mean_value{123.675,116.28,103.53};
    std::vector<float> std_value{57.63,57.63,57.63};
    std::vector<cv::Mat> src_channels(3);
    cv::split(src_mat, src_channels);

    for (int i = 0; i < src_channels.size(); i++)
    {
        src_channels[i].convertTo(src_channels[i], CV_32FC1);
        src_channels[i] = (src_channels[i] - mean_value[i]) / (0.00001 + std_value[i]);
    }
    cv::merge(src_channels, dst_image);
}

int main()
{
    std::string model_file = "/easy_data/one_class/classnet.onnx";
    std::string image_file = "/easy_data/one_class/000.png";
    std::string embedding_file = "/easy_data/one_class/embedding.bin";

    // int result = -1;
    // float *tempOutput[1] = {NULL};
    float* embedding_coreset = new float[LENGTH]();
    std::ifstream f1(embedding_file, std::ios::binary);
    if(f1)
    {
        f1.read(reinterpret_cast<char*>(&embedding_coreset), sizeof(float)*LENGTH);
    }
    f1.close();

    for (int i = 0; i < 10; i++)
    {
        std::cout << embedding_coreset[i] << std::endl;
    }

    // cv::Mat image = cv::imread(image_file);
    // cv::Mat dst_image;
    // preprocess(image, cv::Size(224,224), dst_image);
    // cv::dnn::Net net = cv::dnn::readNetFromONNX(model_file);
    // cv::Mat blob = cv::dnn::blobFromImage(dst_image);
    // net.setInput(blob);

    // cv::Mat out = net.forward();

    // float score;
    // score = postprocess(embedding_coreset, out);
    // std::cout << "out channel: " << out.channels() << " out height: " << out.rows \
    //           << " out width: " << out.cols << std::endl;
    // std::cout << out << std::endl;

    return 0;
}

float postprocess(const cv::Mat &embedding_coreset, const cv::Mat &embedding_test)
{
    const int K(KNEIGHBOURS);
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(K);
    knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    cv::Mat labels(embedding_coreset.rows, 1, CV_32FC1);

    knn->train(embedding_coreset, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result;
    knn->findNearest(embedding_test, K, result);

    float* max_neighbourts = (float*)result.data[0];
    std::cout << max_neighbourts << std::endl;
    

    // knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)
    // labels = [0 for _ in range(embedding_coreset.shape[0])]
    // labels = np.asarray(labels)
    // knn.train(embedding_coreset, cv2.ml.ROW_SAMPLE, labels)
    // result = knn.findNearest(embedding_test, k=self.neighbor_count)
    // score_patches = result[-1]

    // N_b = score_patches[np.argmax(score_patches[:, 0])]
    // w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
    // score = w * max(score_patches[:, 0])  # Image-level score
    float score = 0.0;

    return score;
}