#define	USE_OPENCV	1
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
using namespace caffe;
//using namespace std;
//using namespace cv;

#include"directory.h"


//加载均值文件  
cv::Mat SetMean(const string& mean_file)  
{  
  BlobProto blob_proto;  
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);  
  

  Blob<float> mean_blob;  
  mean_blob.FromProto(blob_proto);  
  //验证均值图片的通道个数是否与网络的输入图片的通道个数相同  
  
 //把三通道的图片分开存储，三张图片按顺序保存到channels中  
  std::vector<cv::Mat> channels;  
  float* data = mean_blob.mutable_cpu_data();  
  for (int i = 0; i < 3; ++i) {  
  
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);  
    channels.push_back(channel);  
    data += mean_blob.height() * mean_blob.width();  
  }  
  
//重新合成一张图片  
  cv::Mat mean;  
  cv::merge(channels, mean);  
 // cv::cvtColor(mean,mean,CV_BGR2RGB);

  return mean;
} 
cv::Mat Meanprocess(cv::Mat mean,cv::Mat image)
{
  cv::Mat image_float;
  image.convertTo(image_float, CV_32FC3); 
  image_float=image_float-mean;
 // cv::normalize(image_float, image, 0, 255, cv::NORM_MINMAX, CV_8UC3);
  /* for(int i=0;i<image.rows;i++)
    {
       for(int j=0;j<image.cols;j++)
       {
         for(int k=0;k<3;k++)
            std::cout<<"piex"<<(float)image.at<Vec3b>(i,j)[k]<<std::endl;
        }
    }*/

  return image_float;
  
}
//人脸裁剪,输入为人脸框+图片
cv::Mat Cropimage(cv::Mat img, int x1, int x2, int y1, int y2)
{

	float scale = 0.4f;
	int w = x2 - x1;
	int h = y2 - y1;


	int miny = max(0, int(y1 - scale*h));
	int minx = max(0, int(x1 - scale*w));
	int maxy = min(img.rows, int(y1 + (1 + scale)*h));
	int maxx = min(img.cols, int(x1 + (1 + scale)*w));
	cv::Mat roi = img(cv::Rect(minx, miny, maxx - minx, maxy - miny));

	int maxlenght = max(roi.rows, roi.cols);
	cv::Mat img0 = cv::Mat::zeros(maxlenght, maxlenght, CV_8UC3);


	int newx1 = maxlenght*.5 - roi.cols * .5;
	int newy1 = maxlenght*.5 - roi.rows * .5;
	int neww = (maxlenght*.5 + roi.cols * .5) - newx1;
	int newh = (maxlenght*.5 + roi.rows * .5) - newy1;
	cv::Rect temp(newx1, newy1, neww, newh);
	roi.copyTo(img0(temp));

	return img0;

}
int race_predict(const boost::shared_ptr< Net<float> > &net,cv::Mat img, cv::Rect facerect)
{
  
  /*if (img.channels()==4)
  {
    vector<cv::Mat> bgr;
    split(img,bgr);
    bgr.pop_back();
    cv::merge(bgr, img);     
  }*/
  
  
  
  
  
  if (img.channels() !=3)
  {
    std::cout <<"when race predict ,image must be three channels" << std::endl;
    return -1;
  }
  if(facerect.width<2||facerect.height<2||facerect.width>img.cols||facerect.height>img.rows||
  facerect.x<0||facerect.y<0||facerect.x>img.cols||facerect.y>img.rows)
  {
    std::cout <<"when race predict, face rect out of range" << std::endl;
    return -1;
  }
  
  
  
    //随机标签
  vector<int> labelVector;
  labelVector.push_back(0);
  
//第二步:裁剪出人脸
  cv::Mat image=Cropimage(img,facerect.x, facerect.x+facerect.width,facerect.y, facerect.y+facerect.height);
  cv::Size size(256,256);
  cv::resize(image,image,size);//缩放到指定大小
  //cv::Mat mean=SetMean("model/imagenet_meanv3.binaryproto");
  //image=Meanprocess(mean,image);


 //
 // cv::imwrite("crop.jpg",image);
  //image=cv::imread("cropfu.jpg");
  //输入图片
  vector<cv::Mat> imageVector;
  imageVector.push_back(image);




  // 载入图片数据到caffe中
  float loss = 0.0;
  boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
  memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name("data"));
  memory_data_layer->AddMatVector(imageVector,labelVector);

  //神经网络前向计算
  vector<Blob<float>*> results = net->ForwardPrefilled(&loss);

   // 获取结果
  const float* argmaxs = results[1]->cpu_data();
  float max = 0;
  float max_i = 0;
  for (int i = 0; i < 4; ++i) {
    float value = results[0]->cpu_data()[i];
    std::cout <<"classify "<<i<<" probability is :"<<value << std::endl;
    if (max < value){
      max = value;
      max_i = i;
    }
  }
  return max_i;
}
boost::shared_ptr< Net<float> > race_init(string proto="model/gender_train_val.prototxt",string model="model/caffenet_train_iter_v3.caffemodel")
{
  //第一步.加载模型
  // 设置计算只采用CPU
  Caffe::set_mode(Caffe::CPU);
  //加载网络模型
  boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto,caffe::TEST));
  //加载均值文件
  net->CopyTrainedLayersFrom(model);
  
  return net;
}

int main(int argc, char** argv)
{
// ::google::InitGoogleLogging("VR");
//  FLAGS_stderrthreshold =::google::ERROR;
  boost::shared_ptr< Net<float> > net;

  net=race_init();
   vector<string>labels;
  labels.push_back("black");
  labels.push_back("brown");
  labels.push_back("white");
  labels.push_back("yellow");
  
  
 /* cv::Mat image= cv::imread("2.jpg");

  cv::Rect facerect(108, 150, 372 ,371);
  // cv::Rect facerect(237,163,224,224);
  // cv::Rect facerect(0,0,image.cols,image.rows);
   int maxi=race_predict(net,image,facerect);




   //显示结果
  cv::imshow(labels[maxi],image);
  cv::moveWindow(labels[maxi], 100, 100);
  cv::waitKey(0);*/
  
  
  
    //读取图片数据
    vector<string>filepathname;
    filepathname=DirProcess("image");
    for(int	i=0;i<filepathname.size();i++)
    {
        clock_t	start = clock();//计算运行时间


  	    cv::Mat image = cv::imread(filepathname[i],-1);//读取图片
        std::cout<<filepathname[i]<<endl;
        cv::Rect facerect(0,0,image.cols,image.rows);
        int maxi=race_predict(net,image,facerect);

        if (maxi<0)
        {
            return 0;
        }

//打印计算时间
  	    clock_t	finish = clock();
  	    double	t=(double)(finish - start) / CLOCKS_PER_SEC;
  	    std::cout <<"compute time:"<<t << std::endl;

//显示结果
  	    cv::imshow(labels[maxi],image);
  	    cv::moveWindow(labels[maxi], 100, 100);
  	    cv::waitKey(0);
    }
  

 // ::google::ShutdownGoogleLogging();

  return 0;
}

