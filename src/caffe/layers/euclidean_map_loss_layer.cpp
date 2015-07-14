#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <fstream>
#include <sstream>

namespace caffe {

template <typename Dtype>
void EuclideanMapLossLayer<Dtype>::generate_gaussian_mask(Dtype* gmask, int height, int width, int x, int y, double sigma) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      int index = h * width + w;
      double distance = (w-x)*(w-x) + (h-y)*(h-y);
      gmask[index] = exp(-distance/(2*sigma*sigma));
    }
  }
}

template <typename Dtype>
void EuclideanMapLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // initialize parameters
  this->sigma_ = this->layer_param_.euclidean_map_loss_param().sigma();
  this->input_h_ = this->layer_param_.euclidean_map_loss_param().input_h();
  this->input_w_ = this->layer_param_.euclidean_map_loss_param().input_w();
  this->label_index_ = this->layer_param_.euclidean_map_loss_param().label_index();
  this->pred_index_ = this->layer_param_.euclidean_map_loss_param().pred_index();
	int num = bottom[pred_index_]->num();
	int channels = bottom[pred_index_]->channels();

	// Validation
  CHECK_EQ(bottom[pred_index_]->channels()*2, bottom[label_index_]->channels())
        << "Labels should have 2 times channels than the predicted maps.";
  CHECK_EQ(bottom[label_index_]->channels()%2, 0)
        << "Currently only support Gaussian maps in 2D (channels == 2n, n is the number of Gaussian maps).";


  /*// bottom shape
  LOG(INFO) << "bottom[" << pred_index_ << "]: " << bottom[pred_index_]->shape_string();
  LOG(INFO) << "bottom[" << label_index_ << "]: " << bottom[label_index_]->shape_string();

  LOG(INFO) << "sigma:" << sigma_;
  LOG(INFO) << "input_h_:" << input_h_;
  LOG(INFO) << "label_index_:" << label_index_;
  LOG(INFO) << "pred_index_:" << pred_index_;*/

  // copy labels from bottom to points_
  points_.ReshapeLike(*bottom[label_index_]);
  points_.CopyFrom(*bottom[label_index_]);

 /* const Dtype* tmp = points_.cpu_data();
  for (int i = 0; i < points_.count(); ++i) {
  	LOG(INFO) << tmp[i];
  }
  LOG(INFO) << "========================";*/


  // reshape bottom[label_index_] as Gaussian map
//  LOG(INFO) << "before reshape: " << bottom[label_index_]->shape_string ();
  bottom[label_index_]->ReshapeLike(*bottom[pred_index_]);

//  LOG(INFO) << "after reshape: " << bottom[label_index_]->shape_string ();

  int out_h = bottom[pred_index_]->height();
  int out_w = bottom[pred_index_]->width();


//  LOG(INFO) << "Out: " << out_h << " " << out_w;

  double scale_h = (double)out_h / input_h_;
  double scale_w = (double)out_w / input_w_;

//  LOG(INFO) << "scale: " << scale_h << " " << scale_w;
//  LOG(INFO) << points_.count();

  const Dtype* points_ptr = points_.cpu_data();
  Dtype* gmask = bottom[label_index_]->mutable_cpu_data();

  int x, y, offset;
  for (int n=0; n<num; ++n) {
    	offset = points_.offset(n);
    	x = points_ptr[offset]*scale_w;
    	y = points_ptr[offset+1]*scale_h;
//    	LOG(INFO) << x << " " << y;
    	int btmoffset = bottom[label_index_]->offset(n);
//    	LOG(INFO) << "btmoffset" << btmoffset;

    	generate_gaussian_mask(gmask+btmoffset, out_h, out_w, round(x), round(y), sigma_);

    	std::ofstream ofile;

			// Write gmask for debug
      std::stringstream ss;
      ss << "gmask-" << n << ".txt";
      string str = ss.str();

			ofile.open(str.c_str());

			for (int h = 0; h < out_h; ++h) {
				for (int w = 0; w < out_w; ++w) {
					int index = h * out_w + w;
					ofile << *(gmask+btmoffset + index) << ",";
				}
				ofile << "\n";
			}
			ofile.close();
  }


  /////

  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[pred_index_]);
}

template <typename Dtype>
void EuclideanMapLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanMapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanMapLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanMapLossLayer);
REGISTER_LAYER_CLASS(EuclideanMapLoss);

}  // namespace caffe
