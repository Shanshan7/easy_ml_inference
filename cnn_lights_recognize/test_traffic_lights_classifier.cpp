#include "traffic_lights_classifier.h"


int main(int argc, char** argv)
{
    int rval = 0;

    if(argc != 2)
	{
        printf("usage: ./test_yolov5rt image_name\n");
        exit(0);
    }

    cv::Mat rgb_image;
    TrafficLightsClassifier traffic_lights_classifier;

    rgb_image = cv::imread(argv[1]);
    traffic_lights_classifier.red_green_yellow(rgb_image);

    return rval;
}