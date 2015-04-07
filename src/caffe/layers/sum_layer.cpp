#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	LOG(INFO) << "!!!!!!";
  const SumParameter& sum_param = this->layer_param_.sum_param();

	// read source file
  LOG(INFO) << sum_param.source().c_str();
	std::ifstream infile(sum_param.source().c_str());
	CHECK(infile.good()) << "Failed to open source file "
			<< sum_param.source()<< std::endl;
  string hashtag;
  int num_idx;
	if (!(infile >> hashtag >> num_idx)) {
		LOG(FATAL) << "source file is empty";
	}

	do {
	    CHECK_EQ(hashtag, "#");
	    // read each idx
	    vector<int> curidx;

	    for (int i = 0; i < num_idx; ++i) {
	      int cidx;
	      infile >> cidx;
	      curidx.push_back(cidx);
	    }
	    idx_.push_back(curidx);
	  } while (infile >> hashtag >> num_idx);

	infile.close();
}

template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SumParameter& sum_param = this->layer_param_.sum_param();

	// top shape
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int channels = idx_.size();

	top[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype*  bottom_data = bottom[0]->cpu_data();
  Dtype*        top_data = top[0]->mutable_cpu_data();

  // bottom statistics
  int bottom_num = bottom[0]->num();
  int bottom_channels = bottom[0]->channels(); 
  int bottom_height = bottom[0]->height(); 
  int bottom_width = bottom[0]->width(); 
  int bottom_count = bottom[0]->count();
  int bottom_K = bottom_count / bottom_num; 

  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels(); 
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();
  int top_K = top_count / top_num; 

  // Sum bottom maps
  caffe_set(top_count, Dtype(0), top_data);
  int bottom_offset;

  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < top_channels; ++c) {
      for (int cidx = 0; cidx < idx_[c].size(); ++cidx) {
        for (int h = 0; h < top_height; ++h) {
          for (int w = 0; w < top_width; ++w) {
            bottom_offset = (((n*bottom_channels + idx_[c][cidx])*top_height + h)*top_width + w);
            int top_offset = (((n*top_channels + c)*top_height + h)*top_width + w);
            *(top_data + top_offset) += *(bottom_data + bottom_offset);
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);

}  // namespace caffe
