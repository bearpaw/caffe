#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/C3dmat.hpp"
#include "caffe/vision_layers.hpp"
#include <math.h>
#include <fstream>

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SumParameter& sum_param = this->layer_param_.sum_param();

  // read number of mixtures
  mix_num_ = sum_param.mix_num();

  //-------------------------------------
  //  read parents
  if ( sum_param.parents().size() != 0 ) {
    pa_.clear();
    std::copy(sum_param.parents().begin(),
          sum_param.parents().end(),
          std::back_inserter(pa_));
  }
  //------------- CHECKED ---------------

	// // read source file
 //  LOG(INFO) << sum_param.source().c_str();
	// std::ifstream infile(sum_param.source().c_str());
	// CHECK(infile.good()) << "Failed to open source file "
	// 		<< sum_param.source()<< std::endl;
 //  string hashtag;
 //  int num_idx;
	// if (!(infile >> hashtag >> num_idx)) {
	// 	LOG(FATAL) << "source file is empty";
	// }

	// do {
	//     CHECK_EQ(hashtag, "#");
	//     // read each idx
	//     vector<int> curidx;

	//     for (int i = 0; i < num_idx; ++i) {
	//       int cidx;
	//       infile >> cidx;
	//       curidx.push_back(cidx);
	//     }
	//     global_IDs.push_back(curidx);
	//   } while (infile >> hashtag >> num_idx);

	// infile.close();

  // get IDs
  get_IDs( pa_, mix_num_, nbh_IDs, global_IDs, target_IDs);
 
 // LOG(INFO) << "nbh size: " << nbh_IDs.size();
 //  std::ofstream outfile("nbh_ids.txt");
 //  for (int i = 0; i < nbh_IDs.size(); ++i) {
 //    for (int j = 0; j < nbh_IDs[i].size(); ++j) {
 //      outfile << nbh_IDs[i][j] << " ";
 //      LOG(INFO) << nbh_IDs[i][j];
 //    }
 //    outfile << std::endl;
 //  }
 //  outfile.close();

  //  LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!target size: " << target_IDs.size();
  // std::ofstream outfile("target_ids.txt");
  // for (int i = 0; i < target_IDs.size(); ++i) {
  //   for (int j = 0; j < target_IDs[i].size(); ++j) {
  //     outfile << target_IDs[i][j] << " ";
  //     LOG(INFO) << target_IDs[i][j];
  //   }
  //   outfile << std::endl;
  // }
  // outfile.close();

   LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!! global size: " << global_IDs.size();
  std::ofstream outfile("global_IDs.txt");
  for (int i = 0; i < global_IDs.size(); ++i) {
    for (int j = 0; j < global_IDs[i].size(); ++j) {
      outfile << global_IDs[i][j] << " ";
      //LOG(INFO) << global_IDs[i][j];
    }
    outfile << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Global ID saved!";
}

template <typename Dtype>
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// top shape
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int channels = global_IDs.size();

	top[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype*  bottom_data = bottom[0]->cpu_data();
  Dtype*        top_data = top[0]->mutable_cpu_data();

  // bottom statistics
  int bottom_channels = bottom[0]->channels(); 

  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels(); 
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();

  // Sum bottom maps
  caffe_set(top_count, Dtype(0), top_data);
  int bottom_offset;

  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < top_channels; ++c) {
      for (int cidx = 0; cidx < global_IDs[c].size(); ++cidx) {
        for (int h = 0; h < top_height; ++h) {
          for (int w = 0; w < top_width; ++w) {
            bottom_offset = (((n*bottom_channels + global_IDs[c][cidx])*top_height + h)*top_width + w);
            int top_offset = (((n*top_channels + c)*top_height + h)*top_width + w);
            *(top_data + top_offset) += *(bottom_data + bottom_offset);
          }
        }
      }
    }
  } // end of sum bottom maps
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();


  // bottom statistics
  int bottom_channels = bottom[0]->channels(); 

  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels(); 
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

      caffe_set(top_count, Dtype(0), bottom_diff);
      int bottom_offset;

      for (int n = 0; n < top_num; ++n) {
        for (int c = 0; c < top_channels; ++c) {
          for (int cidx = 0; cidx < global_IDs[c].size(); ++cidx) {
            for (int h = 0; h < top_height; ++h) {
              for (int w = 0; w < top_width; ++w) {
                bottom_offset = (((n*bottom_channels + global_IDs[c][cidx])*top_height + h)*top_width + w);
                int top_offset = (((n*top_channels + c)*top_height + h)*top_width + w);
                *(bottom_diff + bottom_offset) = *(top_diff + top_offset);
              }
            }
          }
        } // end of propagate gradients
      } 
      
    }
  }
}

template <typename Dtype>
void SumLayer<Dtype>::get_IDs(const vector<int>& pa, const int K, 
      vector<vector<int> >& nbh_IDs, vector<vector <int> >& global_IDs, vector<vector<int> >& target_IDs) {
  int p_no = pa.size();
  int t_cnt = 0;
  int g_cnt = 0;
  for (int i = 0; i < p_no; ++i) {
    // compute nbh_IDs
    vector<int> cur_nbh_IDs;
    for (int j = 0; j < p_no; ++j) {
      if (pa[i] == (j+1) || pa[j] == (i+1)) {
        cur_nbh_IDs.push_back(j+1);
      }
    }
    nbh_IDs.push_back(cur_nbh_IDs);

    // compute target IDs
    vector<int> cur_target_IDs;
    for (int k = 0; k < cur_nbh_IDs.size(); ++k) {
      cur_target_IDs.push_back(++t_cnt);
    }
    target_IDs.push_back(cur_target_IDs);

    // compute global ID
    vector<int> cur_global_IDs;
    int max_g = pow(mix_num_, cur_nbh_IDs.size());
    for (int g = 0; g < max_g; ++g) {
      cur_global_IDs.push_back(++g_cnt);
      LOG(INFO) << "g_cnt: " << g_cnt;
    }
    global_IDs.push_back(cur_global_IDs);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);

}  // namespace caffe
