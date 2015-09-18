#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <math.h>
#include <fstream>
#include <sstream>

# define Vec3DElem vector<platero::C3dmat<int>* > 
# define C3DMAT platero::C3dmat<int>

namespace caffe {

template <typename Dtype>
void SumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LOG(INFO) << "Sum setup";
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
void SumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// top shape
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int channels = global_IDs.size();

	top[0]->Reshape(num, channels, height, width);
	caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
}

// global_ids starts from 1 (skip 0 which is the bg channel)
template <typename Dtype>
void SumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LOG(INFO) << "sum forward cpu";
	//	LOG(INFO) << "bottom shape: " << bottom[0]->shape_string();
	//	LOG(INFO) << "top shape: " << top[0]->shape_string();
	clock_t start, end;
	start = clock();
	Dtype*        top_data = top[0]->mutable_cpu_data();
	const Dtype*	bottom_data = bottom[0]->cpu_data();

	// top statistics
	int top_num = top[0]->num();
	int top_channels = top[0]->channels();
	int top_height = top[0]->height();
	int top_width = top[0]->width();
	//int top_count = top[0]->count();
	int map_size = top_width*top_height;
	//int top_index, bottom_index;
	// compute global ids
	int cnt = 0;
	for (int c = 0; c < top_channels; ++c) {
		vector<int> global_ids_vec = global_IDs[c][0]->vectorize();
		// ---- global_ids_vec stores all the bottom channels should be summed together
		for (int nbhid = 1; nbhid < global_IDs[c].size(); ++nbhid) { // number of neighbor
			vector<int> tmp =  global_IDs[c][nbhid]->vectorize();
			global_ids_vec.insert(global_ids_vec.end(), tmp.begin(), tmp.end());
		} // end nbhid
		global_ids_vecs.push_back(global_ids_vec);
		cnt += global_ids_vec.size();
	}
	LOG(INFO) << "CNT:" << cnt;
	// sum over maps
	for (int n = 0; n < top_num; ++n) {
		for (int c = 0; c < top_channels; ++c) {
			for (int vid = 0; vid < global_ids_vecs[c].size(); ++vid) {
				caffe_axpy<Dtype> (map_size,
						Dtype(1),
						bottom_data + bottom[0]->offset(n, global_ids_vecs[c][vid]),
						top_data + top[0]->offset(n, c));
			} // end vid
		} // end c
	} // end of sum bottom maps

	end = clock();

	LOG(INFO) << "sum forword CPU: "
			<< (double)(end-start)/CLOCKS_PER_SEC
			<< " seconds.";
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype*        bottom_diff = bottom[0]->mutable_cpu_diff();
//	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

	// top statistics
	int top_num = top[0]->num();
	int top_channels = top[0]->channels();
	int top_height = top[0]->height();
	int top_width = top[0]->width();
	int map_size = top_width*top_height;

	for (int n = 0; n < top_num; ++n) {
		for (int c = 0; c < top_channels; ++c) {
			for (int vid = 0; vid < global_ids_vecs[c].size(); ++vid) {
				if (propagate_down[0]) {
					// compute: *(bottom_diff + bottom[0]->offset(n, global_ids_vec[vid], h, w)) += top[0]->diff_at(n, c, h, w);
					caffe_copy (map_size,
							top[0]->cpu_diff() + top[0]->offset(n, c),
							bottom_diff+bottom[0]->offset(n, global_ids_vecs[c][vid]));
				}
			} // end vid

		} // end c
	} // end of sum bottom maps
}

template <typename Dtype>
void SumLayer<Dtype>::get_IDs() {
	int p_no = pa_.size();
	int t_cnt = 0;
	int g_cnt = 0;
	for (int i = 0; i < p_no; ++i) {
		// compute nbh_IDs
		vector<int> cur_nbh_IDs;
		for (int j = 0; j < p_no; ++j) {
			if (pa_[i] == (j+1) || pa_[j] == (i+1)) {
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
			LOG(INFO) << "Numbor of neighborhood must between 1-3.";
		}
		cur_global_IDs.push_back(mat);
		global_IDs.push_back(cur_global_IDs);
	}
}

#ifdef CPU_ONLY
STUB_GPU(SumLayer);
#endif

INSTANTIATE_CLASS(SumLayer);
REGISTER_LAYER_CLASS(Sum);

}  // namespace caffe
