#include "patchcore_postprocess.h"

#define KNEIGHBOURS (9)

unsigned long get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000 + tv.tv_usec);
}

// template<class Function>
// inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//     if (numThreads <= 0) {
//         numThreads = std::thread::hardware_concurrency();
//     }

//     if (numThreads == 1) {
//         for (size_t id = start; id < end; id++) {
//             fn(id, 0);
//         }
//     } else {
//         std::vector<std::thread> threads;
//         std::atomic<size_t> current(start);

//         // keep track of exceptions in threads
//         // https://stackoverflow.com/a/32428427/1713196
//         std::exception_ptr lastException = nullptr;
//         std::mutex lastExceptMutex;

//         for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//             threads.push_back(std::thread([&, threadId] {
//                 while (true) {
//                     size_t id = current.fetch_add(1);

//                     if ((id >= end)) {
//                         break;
//                     }

//                     try {
//                         fn(id, threadId);
//                     } catch (...) {
//                         std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                         lastException = std::current_exception();
//                         /*
//                          * This will work even when current is the largest value that
//                          * size_t can fit, because fetch_add returns the previous value
//                          * before the increment (what will result in overflow
//                          * and produce 0 instead of current + 1).
//                          */
//                         current = end;
//                         break;
//                     }
//                 }
//             }));
//         }
//         for (auto &thread : threads) {
//             thread.join();
//         }
//         if (lastException) {
//             std::rethrow_exception(lastException);
//         }
//     }
// }

cv::Mat train_embedding_process(const std::string &embedding_file,
                                const int output_channel)
{
    std::ifstream embedding(embedding_file, std::ios::binary|std::ios::in);
    embedding.seekg(0,std::ios::end);
    int embedding_length = embedding.tellg() / sizeof(float);
    embedding.seekg(0, std::ios::beg);
    float* embedding_coreset = new float[embedding_length];
    embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * embedding_length);
    embedding.close();

    cv::Mat embedding_train(embedding_length / output_channel, output_channel, CV_32FC1);

    memcpy(embedding_train.data, embedding_coreset, embedding_length * sizeof(float));

    return embedding_train;
}

cv::Mat reshape_embedding(const float *output,
                          const int out_channel, \ 
                          const int out_height, \
                          const int out_width)
{
    cv::Mat embedding_test(out_height*out_width, out_channel, CV_32FC1);
    for (int h = 0; h < out_height; h++)
    {
        for (int w = 0; w < out_width; w++)
        {
            for (int c = 0; c < out_channel; c++)
            {
                ((float*)embedding_test.data)[h*out_width*out_channel + w*out_channel + c] = 
                    output[c*out_height*out_height + h*out_width + w];
                // memcpy(embedding_test.data + h*out_width*out_channel + w*out_channel + c, \
                // embedding_array.data + c*out_height*out_height + h*out_width + w, sizeof(float));
            }
        }
    }
    return embedding_test;
}

void knn_process(const cv::Mat &embedding_train, const cv::Mat &embedding_test, float* distances)
{
    cv::Mat result, neighborResponses, distances_mat;
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(KNEIGHBOURS);
    knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    cv::Mat labels(embedding_train.rows, 1, CV_32FC1, cv::Scalar(0.0));

    knn->train(embedding_train, cv::ml::ROW_SAMPLE, labels);

    knn->findNearest(embedding_test, KNEIGHBOURS, result, neighborResponses, distances_mat);

    int distanceMatWidth = distances_mat.size[0];
    int distanceMatHeight = distances_mat.size[1];
    // std::cout << "result: " << distances_mat.size[0] << " " << distances_mat.size[1] << std::endl;

    // reshape 784 * 9 --> 9 * 784
    for (int d = 0; d < distanceMatHeight; d++)
    {
        for (int c = 0; c < distanceMatWidth; c++)
        {
            distances[d*distanceMatWidth + c] = ((float*)distances_mat.data)[c*distanceMatHeight + d];
            // memcpy(distances + d*784 + c, distances_mat.data + c*9 + d, sizeof(float));
        }
    }
}