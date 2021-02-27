#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>

// compilation command:
// g++ -ggdb inference_example.cpp -ltensorflow -o inference_example `pkg-config --cflags --libs opencv`

void NoOpDeallocator(void* data, size_t a, void* b) {}

std::vector<std::vector<float>> getInspectionPoints(){

    float width = 2.0;
    float length = 2.0;
    int num_points_per_edge = 100;

    float dx = width/num_points_per_edge;
    float dy = length/num_points_per_edge;

    float xmin_ = 0.0;
    float ymin_ = 0.0;
    float xmax_ = width;
    float ymax_ = length;

    std::vector<float> xs;
    std::vector<float> ys;

    float x, y;
    for (int i = 0; i < num_points_per_edge; i++) {
        x = xmin_ + i*dx;
        y = ymin_ + i*dy;

        xs.push_back(x);
        xs.push_back(x);
        xs.push_back(xmin_);
        xs.push_back(xmax_);

        ys.push_back(ymin_);
        ys.push_back(ymax_);
        ys.push_back(y);
        ys.push_back(y);
    }

    std::vector<std::vector<float>> points = {xs, ys};

    return points;
}

float RandomNum(const float min, const float max) {
    return min + ((float)rand()/RAND_MAX) * (max - min);
}

std::vector<std::vector<float>> getRandomObstacles(const int num_rects=10, const float max_size=0.2) {

    float width = 2.0;
    float length = 2.0;

    float xmin_ = 0.0;
    float ymin_ = 0.0;
    float xmax_ = width;
    float ymax_ = length;

    std::vector<float> x1;
    std::vector<float> y1;
    std::vector<float> x2;
    std::vector<float> y2;

    while (x1.size() < num_rects) {
        float width = RandomNum(0, max_size);
        float length = RandomNum(0, max_size);
        float x = RandomNum(xmin_ + 0.5*width, xmax_ - 0.5*width);
        float y = RandomNum(ymin_ + 0.5*length, ymax_ - 0.5*length);

        x1.push_back(x);
        y1.push_back(y);
        x2.push_back(x+width);
        y2.push_back(y+length);
    }

    std::vector<std::vector<float>> obstacles = {x1, y1, x2, y2};

    return obstacles;
}

int main() {

    srand(2);

    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();


    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "saved_model/";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }    

    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
	    printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    //printf("%ld\n",sizeof(TF_Output));
    Input[0] = t0;

    
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    // get obstacles and inpection points
    std::vector<std::vector<float>> points = getInspectionPoints();
    std::vector<std::vector<float>> obstacles = getRandomObstacles();

    // create image
    int scale = 50;
    cv::Mat input_mat(cv::Size(101, 101), CV_8UC1);
    input_mat = 0;
    for (int i=0; i<10; i++) {
        cv::rectangle(input_mat, cv::Point((int)round(obstacles[0][i]*scale), (int)round(obstacles[1][i]*scale)), cv::Point((int)round(obstacles[2][i]*scale), (int)round(obstacles[3][i]*scale)), cv::Scalar(255), -1);
    }
    for (int i=0; i<400; i++) {
        cv::rectangle(input_mat, cv::Point((int)round(points[0][i]*scale), (int)round(points[1][i]*scale)), cv::Point((int)round(points[0][i]*scale), (int)round(points[1][i]*scale)), cv::Scalar(100));
    }
    //cv::imwrite("sample.png", input_mat);

    // flatten image
    uint totalElements = input_mat.total()*input_mat.channels(); // Note: image.total() == rows*cols.
    cv::Mat flat = input_mat.reshape(1, totalElements); // 1xN mat of 1 channel, O(1) operation

    // prepare input tensor
    int ndims = 2;
    int input_size = 10206;
    int64_t dims[] = {1,input_size};
    float data[input_size];
    for (int i=0;i<5;i++) {
        data[i] = 1.0;
    }    
    for (int i=5;i<input_size;i++) {
        data[i] = (int)(flat.at<uchar>(0, i-5)) / 255.0;
    }

	// open data file
    /*
	std::ofstream data_file;
	data_file.open("inputs.txt");
    for (int i=0; i<input_size; ++i) {
        data_file << std::fixed << std::setprecision(16) << data[i] << std::endl;
    }
    data_file.close();    
    */

    int ndata = sizeof(float)*input_size; // This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
        printf("TF_NewTensor is OK\n");
    }
    else
	printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;

    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);

    if(TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
    }

    // //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;
    printf("Result Tensor :\n");
    printf("%f\n",offsets[0]);
    printf("%f\n",offsets[1]);
    printf("%f\n",offsets[2]);
    printf("%f\n",offsets[3]);
    printf("%f\n",offsets[4]);

    return 0;
}
