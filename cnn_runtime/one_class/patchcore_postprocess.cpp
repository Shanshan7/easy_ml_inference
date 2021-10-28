#include "patchcore_postprocess.h"

#define KNEIGHBOURS (9)

unsigned long get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000 + tv.tv_usec);
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

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

void ann_train(const cv::Mat &embedding_train, const std::string save_path)
{
    int ef_construction = KNEIGHBOURS * 10;
    int M = 16;
    // Declaring index，声明索引类型，如：l2, cosine or ip
    auto space = std::unique_ptr<hnswlib::L2Space>(new hnswlib::L2Space(embedding_train.rows));
    // std::cout << embedding_train.rows << " " << embedding_test.rows << std::endl;
    // 初始化索引，元素的最大数需要是已知的
    auto appr_alg = std::unique_ptr<hnswlib::HierarchicalNSW<float>>(
            new hnswlib::HierarchicalNSW<float>(space.get(), embedding_train.rows, M,
                                                ef_construction));
    appr_alg->setEf(ef_construction);
    // std::priority_queue<std::pair<float, hnswlib::labeltype >> ret=appr_alg->searchKnn(embedding_test.data, K);//'1' is the nearest neg
    int count = 0;
    int vecdim = embedding_train.cols;
    unsigned long time_start_insert, time_end_insert, time_start_search, time_end_search;
    time_start_insert = get_current_time();
    ParallelFor(0, embedding_train.rows, 1, [&](size_t row, size_t threadId)
    {
        float *mass = new float[vecdim];
        memcpy(mass, embedding_train.data + row*vecdim, vecdim * sizeof(float));
        appr_alg->addPoint((void*)mass, row);
        delete[] mass;
    });
    time_end_insert = get_current_time();
    std::cout << "addpoint cost time: " <<  (time_end_insert - time_start_insert)/1000.0  << "ms" << std::endl;

    // save index
    appr_alg->saveIndex(save_path);
}

void ann_process(const cv::Mat &embedding_train, const cv::Mat &embedding_test, float* distances, const std::string save_path)
{
    // process
    int vecdim = embedding_train.cols;
    auto l2space = new hnswlib::L2Space(embedding_train.rows);
    auto appr_alg = new hnswlib::HierarchicalNSW<float>(l2space, save_path, false, embedding_train.rows);

    // time_start_search = get_current_time();

	ParallelFor(0, embedding_test.rows, 4, [&](size_t row, size_t threadId)
	{
		float *mass = new float[vecdim];
        memcpy(mass, embedding_test.data + row*vecdim, vecdim * sizeof(float));
		std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates = 
                                                        appr_alg->searchKnn((void*)mass, KNEIGHBOURS);
		delete[] mass;

        for (int k = 0; k < KNEIGHBOURS; k++)
        {
            // std::cout << "i: " << i << ", k: " << k << ", dist: " << candidates.top().first << std::endl;
            distances[row*KNEIGHBOURS + k] = candidates.top().first;
            candidates.pop();
        }
	});
    // time_end_search = get_current_time();
    // std::cout << "search cost time: " <<  (time_end_search - time_start_search)/1000.0  << "ms" << std::endl;
    //-----------------------------------------------------------------------------
}