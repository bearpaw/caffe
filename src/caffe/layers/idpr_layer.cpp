#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void IdprLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SumParameter& sum_param = this->layer_param_.sum_param();

  // read number of mixtures
  mix_num_ = sum_param.mix_num();

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
	    nhb_ids.push_back(curidx);
	  } while (infile >> hashtag >> num_idx);

	infile.close();

  // Create idpr_global_ids
  int cnt = 0;
  for (int i = 0; i < nhb_ids.size(); ++i) {
    vector<vector <int> > cur_nbh;
    for (int j = 0; j < nhb_ids[i].size(); ++j) {
      vector<int> cur_global_nbh_ids;
      for (int k = 0; k < mix_num_; ++k) {
        cur_global_nbh_ids.push_back(++cnt);
      }
      cur_nbh.push_back(cur_global_nbh_ids);
    }
    idpr_global_ids.push_back(cur_nbh);
  }

  // print idpr_global_ids
  for (int i = 0; i < idpr_global_ids.size(); ++i) {
    for (int j = 0; j < idpr_global_ids[i].size(); ++j) {
      for (int k = 0; k < idpr_global_ids[i][j].size(); ++k) {
        LOG(INFO) << idpr_global_ids[i][j][k];
      }
    }
  }
}

template <typename Dtype>
void IdprLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// top shape
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int channels = nhb_ids.size();

	top[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void IdprLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  // const Dtype*  bottom_data = bottom[0]->cpu_data();
  // Dtype*        top_data = top[0]->mutable_cpu_data();

  // // bottom statistics
  // int bottom_channels = bottom[0]->channels(); 

  // // top statistics
  // int top_num = top[0]->num();
  // int top_channels = top[0]->channels(); 
  // int top_height = top[0]->height(); 
  // int top_width = top[0]->width(); 
  // int top_count = top[0]->count();

}

template <typename Dtype>
void IdprLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();


  // // bottom statistics
  // int bottom_channels = bottom[0]->channels(); 

  // // top statistics
  // int top_num = top[0]->num();
  // int top_channels = top[0]->channels(); 
  // int top_height = top[0]->height(); 
  // int top_width = top[0]->width(); 
  // int top_count = top[0]->count();

}

#ifdef CPU_ONLY
STUB_GPU(IdprLayer);
#endif

INSTANTIATE_CLASS(IdprLayer);
REGISTER_LAYER_CLASS(Idpr);

}  // namespace caffe
