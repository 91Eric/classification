#include "classification.hpp"  
#include <vector>  
#include <iostream>
#include <ctime>

using namespace std;  
using namespace caffe;
int main(int argc, char** argv) {

  
  const string projectroot = "/home/eric/backup/cmake/classification/";
  const string model_file = "/home/eric/backup/cmake/classification/proto/deploy.prototxt";
  const string trained_file = "/home/eric/backup/cmake/classification/bvlc_reference_caffenet.caffemodel";
  const string mean_file    = "/home/eric/backup/cmake/classification/proto/mean.binaryproto";
  const string label_file   = "/home/eric/backup/cmake/classification/synset_words.txt";
  Classifier classifier(model_file, trained_file, mean_file, label_file);
 
  const string file = "../../data/cat.jpg";

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;
  clock_t start=clock();
  cv::Mat img = cv::imread(file, -1);
  if(img.empty())
  {
      cout<<"can't open imagine file!"<<endl;
      return 1;
  }
  cout<< "begin claffification"<<endl;
  std::vector<Prediction> predictions = classifier.Classify(img);
  clock_t end=clock();
  cout<<"end claffification,total timd is"<<static_cast<double>((end-start)/CLOCKS_PER_SEC)<<endl;
  // Print the top N predictions. 
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}

