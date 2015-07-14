#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

/**
 * layer to compute the logarithm of the input data elementwise
 * based on bnll_layer.cpp
 * Wei Yang | April 13, 2015
 * */

namespace caffe {

template <typename Dtype>
void LogarithmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();  
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = log(std::max(bottom_data[i], REAL_MEAN));
  }
}

template <typename Dtype>
void LogarithmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (1. / bottom_data[i] );
      // Would this be divided by zero?
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LogarithmLayer);
#endif

INSTANTIATE_CLASS(LogarithmLayer);
REGISTER_LAYER_CLASS(Logarithm);

}  // namespace caffe
