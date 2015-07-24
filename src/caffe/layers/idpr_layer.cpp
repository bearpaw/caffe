#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <math.h>
#include <fstream>

# define Vec3DElem vector<platero::C3dmat<int>* > 
# define C3DMAT platero::C3dmat<int>

namespace caffe {

template <typename Dtype>
void IdprLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const SumParameter& sum_param = this->layer_param_.sum_param();
  // read number of mixtures
  mix_num_ = sum_param.mix_num();
  //  read parents
  if ( sum_param.parents().size() != 0 ) {
    pa_.clear();
    std::copy(sum_param.parents().begin(),
          sum_param.parents().end(),
          std::back_inserter(pa_));
  }
  // get IDs
  get_IDs();
}

template <typename Dtype>
void IdprLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// top shape
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int channels = 0;
  for (int p = 0; p < idpr_global_ids.size(); ++p) {
    for (int n = 0; n < idpr_global_ids[p].size(); ++n) {
      channels += mix_num_;
    }
  }

  top[0]->Reshape(num, channels, height, width);
  normfactor.Reshape(1, 1, height, width);
}

template <typename Dtype>
void IdprLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype*        top_data = top[0]->mutable_cpu_data();

  // top statistics
  int top_num = top[0]->num();
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();

  // compute IDPR maps
  caffe_set(top_count, Dtype(0), top_data);

  for (int n = 0; n < top_num; ++n) {
    int  tmapid = 0;
    for (int p = 0; p < idpr_global_ids.size(); ++p) { // for each part
      for (int s = 0; s < idpr_global_ids[p].size(); ++s) { // for each neighbor
        platero::C3dmat<int>* idpr_mix = idpr_global_ids[p][s];  

        for (int m = 0; m < mix_num_; ++m) {
          for (int d = 0; d < idpr_mix->get_dims(); ++d) {
            for (int c = 0; c < idpr_mix->get_cols(); ++c) { 
              // print the m-th row of each dimension
              int bmapid = idpr_mix->at(m, c, d);

              // ----------------------------------------
              // sum a map
              for (int h = 0; h < top_height; ++h) {
                for (int w = 0; w < top_width; ++w) {
                   *(top_data + top[0]->offset(n, tmapid, h, w)) += bottom[0]->data_at(n, bmapid, h, w);
                } // end w
              } // end h
              // ----------------------------------------

            } // end col
          } // end dim
          tmapid++;
        } // end mix

        // Normalize the summed IDPR map
        for (int h = 0; h < top_height; ++h) {
          for (int w = 0; w < top_width; ++w) {
            Dtype normfactor = 0;
            for (int tid = tmapid - 1; tid >= tmapid - mix_num_; --tid) {
              normfactor += top[0]->data_at(n, tid, h, w);
            }
            for (int tid = tmapid - 1; tid >= tmapid - mix_num_; --tid) {
              *(top_data + top[0]->offset(n, tid, h, w)) = top[0]->data_at(n, tid, h, w) / normfactor;
            }
          } // end w
        } // end h

      } // end neighbor
    } // end parts
  } // end sample
}

template <typename Dtype>
void IdprLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype*        bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	// top statistics
	int top_num = top[0]->num();
	int top_height = top[0]->height();
	int top_width = top[0]->width();
	int top_count = top[0]->count();

	Dtype* normfactor_ = normfactor.mutable_cpu_data();

	// compute IDPR maps
	for (int n = 0; n < top_num; ++n) {
		int  tmapid = 0;
		for (int p = 0; p < idpr_global_ids.size(); ++p) { // for each part
			for (int s = 0; s < idpr_global_ids[p].size(); ++s) { // for each neighbor
				platero::C3dmat<int>* idpr_mix = idpr_global_ids[p][s];

				// Normalize the summed IDPR map
				caffe_set(normfactor.count(), Dtype(0), normfactor_);

				for (int h = 0; h < top_height; ++h) {
					for (int w = 0; w < top_width; ++w) {
						for (int tid = tmapid - 1; tid >= tmapid - mix_num_; --tid) {
							*(normfactor_+normfactor.offset(0, 0, h, w)) += top[0]->data_at(n, tid, h, w);
						}
					} // end w
				} // end h

				for (int m = 0; m < mix_num_; ++m) {
					for (int d = 0; d < idpr_mix->get_dims(); ++d) {
						for (int c = 0; c < idpr_mix->get_cols(); ++c) {
							// print the m-th row of each dimension
							int bmapid = idpr_mix->at(m, c, d);

							// ----------------------------------------
							// sum a map
							for (int h = 0; h < top_height; ++h) {
								for (int w = 0; w < top_width; ++w) {
									*(bottom_diff + bottom[0]->offset(n, bmapid, h, w)) += top[0]->diff_at(n, tmapid, h, w)/normfactor.data_at(0, 0, h, w);
								} // end w
							} // end h
							// ----------------------------------------

						} // end col
					} // end dim
					tmapid++;
				} // end mix


			} // end neighbor
		} // end parts
	} // end sample
}


template <typename Dtype>
void IdprLayer<Dtype>::get_IDs() {
  using namespace std;
  using namespace platero;

  int p_no = pa_.size();
  int t_cnt = 0;
  int g_cnt = 0;

  for (int i = 0; i < p_no; ++i) {
    //-----------------------------------------
    // compute nbh_IDs
    vector<int> cur_nbh_IDs;
    for (int j = 0; j < p_no; ++j) {
      if (pa_[i] == (j+1) || pa_[j] == (i+1)) {
        cur_nbh_IDs.push_back(j+1);
      }
    }
    nbh_IDs.push_back(cur_nbh_IDs);

    //-----------------------------------------
    // compute target IDs
    vector<int> cur_target_IDs;
    for (int k = 0; k < cur_nbh_IDs.size(); ++k) {
      cur_target_IDs.push_back(++t_cnt);
    }
    target_IDs.push_back(cur_target_IDs);

    //-----------------------------------------
    // compute global ID
    Vec3DElem cur_global_IDs;
    C3DMAT* mat;

    if (cur_nbh_IDs.size() == 1) {
      mat = new C3DMAT(mix_num_, 1, 1);
      for (int row = 0; row < mix_num_; ++row) {
        mat->set(row, 0, 0, ++g_cnt);
      }
    } else if (cur_nbh_IDs.size() == 2) {
      mat = new C3DMAT(mix_num_, mix_num_, 1);
      for (int col = 0; col < mix_num_; ++col) {
        for (int row = 0; row < mix_num_; ++row) {
          mat->set(row, col, 0, ++g_cnt);
        }
      }

    } else if (cur_nbh_IDs.size() == 3) {
      mat = new C3DMAT(mix_num_, mix_num_, mix_num_);
      for (int dim = 0; dim < mix_num_; ++dim) {
        for (int col = 0; col < mix_num_; ++col) {
          for (int row = 0; row < mix_num_; ++row) {
            mat->set(row, col, dim, ++g_cnt);
          }
        }
      }
    } else {
      LOG(INFO) << "Number of neighborhood must between 1-3.";
    }
    cur_global_IDs.push_back(mat);
    global_IDs.push_back(cur_global_IDs);

    //-----------------------------------------
    // compute global_idpr_ids
    Vec3DElem idpr_part;

    for (int n = 0; n < cur_nbh_IDs.size(); ++n) {
      C3DMAT* idpr_neighbor = new C3dmat<int>(1, 1, 1);
      idpr_neighbor->clone(*mat);
      switch (n) {
        case 0: break;
        case 1:
          idpr_neighbor->permute(2, 1, 3);
          break;
        case 2:
          idpr_neighbor->permute(3, 1, 2);
          break;
        default:
          LOG(ERROR) << "Number of neighbor cannot be greater than 3";
      }
      idpr_part.push_back(idpr_neighbor);
    } // end Neighbor
    idpr_global_ids.push_back(idpr_part);
  }
}


#ifdef CPU_ONLY
STUB_GPU(IdprLayer);
#endif

INSTANTIATE_CLASS(IdprLayer);
REGISTER_LAYER_CLASS(Idpr);

}  // namespace caffe
