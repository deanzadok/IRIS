#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <visilibity.hpp>

// compilation command:
// g++ -ggdb inference_example.cpp -ltensorflow -o inference_example `pkg-config --cflags --libs opencv`

// compilation command with visilibity:
// g++ -ggdb inference_example.cpp -I /home/deanz/Documents/Github/VisiLibity1/src /home/deanz/Documents/Github/VisiLibity1/src/visilibity.cpp /home/deanz/Documents/Github/VisiLibity1/src/visilibity.hpp -ltensorflow -o inference_example `pkg-config --cflags --libs opencv`


void NoOpDeallocator(void* data, size_t a, void* b) {}

float* ComputeEndEffector(float* data) {

    float origin[] = {1.0, 0.0};
    float link_lengths[] = {0.2, 0.1, 0.2, 0.3, 0.1};

	float x, y;
	x = origin[0];
	y = origin[1];
    int num_links = 5;
	for (int i = 0; i < num_links; ++i) {
		x += link_lengths[i] * std::cos(data[i]);
		y += link_lengths[i] * std::sin(data[i]);
	}
    
    float* ee_val = new float[2]();
    ee_val[0] = x;
    ee_val[1] = y;
    return ee_val;
}

std::vector<cv::Point> ComputeVisibilityTriangle(float* ee_val, float ee_orientation) {

    int maxWall = 200;
    double fov = M_PI_2;
    int scale = 50;
    float x1 = scale * ee_val[0] + maxWall * std::cos(ee_orientation + 0.5 * fov);
    float y1 = scale * ee_val[1] + maxWall * std::sin(ee_orientation + 0.5 * fov);
    float x2 = scale * ee_val[0] + maxWall * std::cos(ee_orientation - 0.5 * fov);
    float y2 = scale * ee_val[1] + maxWall * std::sin(ee_orientation - 0.5 * fov);

    std::vector<cv::Point> points;
    // cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale))
    // points = { cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)),  cv::Point((int)round(x1), (int)round(y1)), 
    //         cv::Point((int)round(x2), (int)round(y2)) };
    points.push_back(cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)));
    points.push_back(cv::Point((int)round(x2), (int)round(y2)));
    points.push_back(cv::Point((int)round(x1), (int)round(y1)));

    return points;
}

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

std::vector<std::vector<float>> getRandomObstacles(int num_rects=1, const float max_size=0.2) {

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

VisiLibity::Environment prepareEnvironment(std::vector<std::vector<float>> obstacles, int num_rects, double scale=50) {

    double end_env = 2.0 * scale;

    //std::vector<VisiLibity::Polygon> envPolygons;

    //define polygon for walls
    std::vector<VisiLibity::Point> wallsPoints;
    wallsPoints.push_back(VisiLibity::Point(0.0, 0.0));
    wallsPoints.push_back(VisiLibity::Point(end_env, 0.0));
    wallsPoints.push_back(VisiLibity::Point(end_env, end_env));
    wallsPoints.push_back(VisiLibity::Point(0.0, end_env));
    VisiLibity::Polygon wallsPolygon(wallsPoints);
    //envPolygons.push_back(wallsPolygon);
    
    VisiLibity::Environment environment(wallsPolygon);

    /*
    // define polygons for obstacles
    for (std::vector<std::vector<float>>::iterator it_obs = obstacles.begin(); it_obs != obstacles.end(); it_obs++ ) {
        std::vector<VisiLibity::Point> obstaclePoints;
        obstaclePoints.push_back(VisiLibity::Point((*it_obs)[0]*scale, (*it_obs)[1]*scale));
        obstaclePoints.push_back(VisiLibity::Point((*it_obs)[2]*scale, (*it_obs)[1]*scale));
        obstaclePoints.push_back(VisiLibity::Point((*it_obs)[2]*scale, (*it_obs)[3]*scale));
        obstaclePoints.push_back(VisiLibity::Point((*it_obs)[0]*scale, (*it_obs)[3]*scale));
        std::cout << "Obstacle Point 1: " << (*it_obs)[0]*scale << ", " << (*it_obs)[1]*scale << std::endl;
        std::cout << "Obstacle Point 2: " << (*it_obs)[2]*scale << ", " << (*it_obs)[1]*scale << std::endl;
        std::cout << "Obstacle Point 3: " << (*it_obs)[2]*scale << ", " << (*it_obs)[3]*scale << std::endl;
        std::cout << "Obstacle Point 4: " << (*it_obs)[0]*scale << ", " << (*it_obs)[3]*scale << std::endl;

        VisiLibity::Polygon obstaclePolygon(obstaclePoints);
        //envPolygons.push_back(obstaclePolygon);
        environment.add_hole(obstaclePolygon);
    }
    */


    // define polygons for obstacles
    for (int i=0; i<num_rects; i++ ) {
        std::vector<VisiLibity::Point> obstaclePoints;
        obstaclePoints.push_back(VisiLibity::Point(obstacles[0][i]*scale, obstacles[1][i]*scale));
        obstaclePoints.push_back(VisiLibity::Point(obstacles[2][i]*scale, obstacles[1][i]*scale));
        obstaclePoints.push_back(VisiLibity::Point(obstacles[2][i]*scale, obstacles[3][i]*scale));
        obstaclePoints.push_back(VisiLibity::Point(obstacles[0][i]*scale, obstacles[3][i]*scale));
        //std::cout << "Obstacle Point 1: " << (*it_obs)[0]*scale << ", " << (*it_obs)[1]*scale << std::endl;
        //std::cout << "Obstacle Point 2: " << (*it_obs)[2]*scale << ", " << (*it_obs)[1]*scale << std::endl;
        //std::cout << "Obstacle Point 3: " << (*it_obs)[2]*scale << ", " << (*it_obs)[3]*scale << std::endl;
        //std::cout << "Obstacle Point 4: " << (*it_obs)[0]*scale << ", " << (*it_obs)[3]*scale << std::endl;

        VisiLibity::Polygon obstaclePolygon(obstaclePoints);
        //envPolygons.push_back(obstaclePolygon);
        environment.add_hole(obstaclePolygon);
    }

    return environment;
}

std::vector<std::vector<cv::Point>> savePolygonPoints(VisiLibity::Visibility_Polygon visPolygon) {

    std::vector<cv::Point> end_points;

    for (int i=0; i<visPolygon.n(); i++) {
        cv::Point point_i((int)round(visPolygon[i].x()), (int)round(visPolygon[i].y()));
        std::cout << "Point " << i << ": " << (int)round(visPolygon[i].x()) << ", " << (int)round(visPolygon[i].y()) << std::endl;
        end_points.push_back(point_i);
    }

    std::vector<std::vector<cv::Point>> end_points_vec = {end_points};
    return end_points_vec;
}

int main() {

    srand(3);
    double epsilon = 0.000000001;
    int num_rects = 5;

    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();


    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "saved_model_decoder/";
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
    std::vector<std::vector<float>> obstacles = getRandomObstacles(num_rects);

    // create image
    int scale = 50;
    cv::Mat input_mat(cv::Size(101, 101), CV_8UC1);
    input_mat = 0;


    // prepare input tensor
    int ndims = 2;
    int n_z = 8;
    int c_dim = 5;
    int input_size = 10209;
    int64_t dims[] = {1,input_size};
    float data[input_size];


    float c_point[c_dim] = {0.812139, -0.083430, -0.847835, 0.427945, 1.25944};
    float* ee_val = ComputeEndEffector(c_point);
    float ee_orientation = 0.0;
    for (int i=0; i<c_dim; i++) {
        ee_orientation += c_point[i];
    }
    //std::cout << std::fixed << std::setprecision(16) << "ee_orientation: " << ee_orientation << std::endl;
    //cv::rectangle(input_mat, cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)), cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)), cv::Scalar(150));

    std::vector<cv::Point> visibilityTriangle = ComputeVisibilityTriangle(ee_val, ee_orientation);
    for (int i=0; i<3; i++) {
        std::cout << "Point "<< i << ": " << visibilityTriangle[i] << std::endl;
    }
    std::vector<std::vector<cv::Point>> visTriangles = {visibilityTriangle};
    //const cv::Point* visPoints = {visibilityTriangle[0], visibilityTriangle[1], visibilityTriangle[2]};
    //cv::fillConvexPoly(input_mat, visibilityTriangle, cv::Scalar(150));

    // get environment from obstacles
    VisiLibity::Environment environment = prepareEnvironment(obstacles, num_rects);
    std::vector<VisiLibity::Point> observerPoints = {VisiLibity::Point(ee_val[0]*scale, ee_val[1]*scale)};
    VisiLibity::Guards observer(observerPoints);
    observer.snap_to_boundary_of(environment, epsilon);
    observer.snap_to_vertices_of(environment, epsilon);
    VisiLibity::Visibility_Polygon visilibityPolygon(observer[0], environment, epsilon);
    std::vector<std::vector<cv::Point>> visilibityPoints = savePolygonPoints(visilibityPolygon);

    cv::fillPoly(input_mat, visilibityPoints, cv::Scalar(150));

    cv::Mat mask_visibility(cv::Size(101, 101), CV_8UC1);
    mask_visibility = 0;
    cv::fillPoly(mask_visibility, visTriangles, cv::Scalar(150));


    cv::bitwise_and(input_mat, mask_visibility, input_mat);


    for (int i=0; i<10; i++) {
        cv::rectangle(input_mat, cv::Point((int)round(obstacles[0][i]*scale), (int)round(obstacles[1][i]*scale)), cv::Point((int)round(obstacles[2][i]*scale), (int)round(obstacles[3][i]*scale)), cv::Scalar(255), -1);
    }
    for (int i=0; i<400; i++) {
        cv::rectangle(input_mat, cv::Point((int)round(points[0][i]*scale), (int)round(points[1][i]*scale)), cv::Point((int)round(points[0][i]*scale), (int)round(points[1][i]*scale)), cv::Scalar(100));
    }
    cv::rectangle(input_mat, cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)), cv::Point((int)round(ee_val[0]*scale), (int)round(ee_val[1]*scale)), cv::Scalar(0));

    cv::imwrite("sample.png", input_mat);


    // flatten image
    uint totalElements = input_mat.total()*input_mat.channels(); // Note: image.total() == rows*cols.
    cv::Mat flat = input_mat.reshape(1, totalElements); // 1xN mat of 1 channel, O(1) operation
    for (int i=0;i<n_z;i++) {
        data[i] = 1.0;
    }
    for (int i=n_z;i<input_size;i++) {
        data[i] = (int)(flat.at<uchar>(0, i-n_z)) / 255.0;
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