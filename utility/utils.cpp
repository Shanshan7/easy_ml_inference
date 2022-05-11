#include "utils.h"

unsigned long get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000 + tv.tv_usec);
}

void ListPath(std::string const &path, std::vector<std::string> &paths) {
    paths.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
 /*   
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            cout << "name: "<<name<<endl;
            paths.push_back(name);
        }*/
        std::string name = entry->d_name;
        int type = (int)(entry->d_type);
       
       if(type != 8)
       {
         if((strcmp(name.c_str(), ".")!=0) && (strcmp(name.c_str(), "..")!=0) && (strcmp(name.c_str(), "results")!=0)) 
          {            
            #if(YUV420OPEN)
            if((strcmp(name.c_str(), "yuv")==0))
            {
                std::cout << "Dir name: "<<name<<std::endl;
                paths.push_back(name);  
            }
            #else
            if(strcmp(name.c_str(), "yuv")!=0)
            {
                std::cout << "Dir name: "<<name<<std::endl;
                paths.push_back(name);  
            }
            #endif
          }
        }
        
    }

    closedir(dir);
}
/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(std::string const &path, std::vector<std::string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            std::string name = entry->d_name;
            std::string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png") || (ext == "420") || (ext == "bin")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

std::vector<std::vector<float>> martrix_multiply_num(std::vector<std::vector<float>> matrix, float num) 
{
    // 矩阵除以一个数
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            matrix[i][j] = matrix[i][j] * num;
        }
    }
    return matrix;
}

void array_reverse(float* array, int array_length)
{
    float temp;
    for (int i = 0; i < array_length / 2; i++){
        temp = array[array_length-1-i];
        array[array_length-1-i] = array[i];
        array[i] = temp;
    }
} 

/* 复制文件
 * @参数 src - 源文件名
 * @参数 dest - 目标文件名，如果目标文件已存在，则覆盖
 * @返回 true | false 代表拷贝是否成功
*/
bool copy_file(std::string src, std::string dest) {
    std::ifstream is(src, std::ios::binary);
    if (is.fail()) {
        return false;
    }

    std::ofstream os(dest, std::ios::binary);
    if (os.fail()) {
        return false;
    }

    is.seekg(0, std::ios::end);
    long long length = is.tellg();  // C++ 支持的最大索引位置
    is.seekg(0);
    char buf[2048];
    while (length > 0)
    {
        int bufSize = length >= 2048 ? 2048 : length;
        is.read(buf, bufSize);
        os.write(buf, bufSize);
        length -= bufSize;
    }

    is.close();
    os.close();
    return true;
}

void conv1d(float *input, float *kernel, int input_length, int kernel_size, float *output)
{
	int output_length = input_length + kernel_size - 1;
    for(int k = 0; k < output_length; k++) {
        output[k] = 0;
    }
    for(int i = 0; i < output_length; i++) {
        for(int j = std::max(0, i + 1 - kernel_size); j <= std::min(i, input_length - 1); j++) 
        {
            output[i] += input[j] * kernel[i - j];
        }
    }
}

void conv1d(std::vector<float> &input, float *kernel, int input_length, int kernel_size, float *output)
{
	int output_length = input_length + kernel_size - 1;
    for(int k = 0; k < output_length; k++) {
        output[k] = 0;
    }
    for(int i = 0; i < output_length; i++) {
        for(int j = std::max(0, i + 1 - kernel_size); j <= std::min(i, input_length - 1); j++) 
        {
            output[i] += input[j] * kernel[i - j];
        }
    }
}