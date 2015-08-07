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
void SumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  Dtype*        top_data = top[0]->mutable_gpu_data();
  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels(); 
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();
  int map_size = top_width*top_height;

  // Sum bottom maps
  caffe_gpu_set(top_count, Dtype(0), top_data);

  for (int n = 0; n < top_num; ++n) {
    for (int c = 0; c < top_channels; ++c) {
      vector<int> global_ids_vec = global_IDs[c][0]->vectorize();

      // ---- global_ids_vec stores all the bottom channels should be summed together
      for (int nbhid = 1; nbhid < global_IDs[c].size(); ++nbhid) { // number of neighbor
        vector<int> tmp =  global_IDs[c][nbhid]->vectorize();
        global_ids_vec.insert(global_ids_vec.end(), tmp.begin(), tmp.end());
      } // end nbhid

      for (int vid = 0; vid < global_ids_vec.size(); ++vid) {
      	caffe_gpu_axpy (map_size,
      			Dtype(1),
      			bottom[0]->gpu_data()+bottom[0]->offset(n, global_ids_vec[vid]),
      			top_data + top[0]->offset(n, c));
      } // end vid

    } // end c
  } // end of sum bottom maps
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype*        bottom_diff = bottom[0]->mutable_gpu_diff();
	caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

	// top statistics
	int top_num = top[0]->num();
	int top_channels = top[0]->channels();
	int top_height = top[0]->height();
	int top_width = top[0]->width();
  int map_size = top_width*top_height;

	for (int n = 0; n < top_num; ++n) {
		for (int c = 0; c < top_channels; ++c) {
			vector<int> global_ids_vec = global_IDs[c][0]->vectorize();

			// ---- global_ids_vec stores all the bottom channels should be summed together
			// ! Note: no overlap in global_ids_vec (all different). BP is easy
			for (int nbhid = 1; nbhid < global_IDs[c].size(); ++nbhid) { // number of neighbor
				vector<int> tmp =  global_IDs[c][nbhid]->vectorize();
				global_ids_vec.insert(global_ids_vec.end(), tmp.begin(), tmp.end());
			} // end nbhid

			for (int vid = 0; vid < global_ids_vec.size(); ++vid) {
				if (propagate_down[global_ids_vec[vid]]) {
					// compute: *(bottom_diff + bottom[0]->offset(n, global_ids_vec[vid], h, w)) += top[0]->diff_at(n, c, h, w);
					caffe_gpu_axpy (map_size,
	      			Dtype(1),
	      			top[0]->gpu_data() + top[0]->offset(n, c),
	      			bottom_diff+bottom[0]->offset(n, global_ids_vec[vid]));
				}
			} // end vid

		} // end c
	} // end of sum bottom maps
}


INSTANTIATE_LAYER_GPU_FUNCS(SumLayer);

}  // namespace caffe
