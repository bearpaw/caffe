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
__global__ void IdprForward(const int nthreads,
    const Dtype* const top_data,
    const int n, const int channels, const int height, const int width,
    const int tmapid, const int mix_num_,
    Dtype* normfactor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;


    for (int tid = tmapid - 1; tid >= tmapid - mix_num_; --tid) {
      int offset = ((n * channels + tid) * height + h) * width + w;
    	normfactor[index] += *(top_data + offset);
    }
  }
}

template <typename Dtype>
void IdprLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype*        top_data = top[0]->mutable_gpu_data();

  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels();
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();
  int map_size = top_width*top_height;

	Dtype* normfactor_ = normfactor.mutable_gpu_data();

  // compute IDPR maps
  caffe_gpu_set(top_count, Dtype(0), top_data);


  for (int n = 0; n < top_num; ++n) {
    int normcnt = 0;
    int  tmapid = 0;
    for (int p = 0; p < idpr_global_ids.size(); ++p) { // for each part
      for (int s = 0; s < idpr_global_ids[p].size(); ++s) { // for each neighbor
        platero::C3dmat<int>* idpr_mix = idpr_global_ids[p][s];  

        for (int m = 0; m < mix_num_; ++m) {
          for (int d = 0; d < idpr_mix->get_dims(); ++d) {
            for (int c = 0; c < idpr_mix->get_cols(); ++c) { 
              // print the m-th row of each dimension
              int bmapid = idpr_mix->at(m, c, d);

              // -------sum a map
            	caffe_gpu_axpy (map_size,
            			Dtype(1),
            			bottom[0]->gpu_data()+bottom[0]->offset(n, bmapid),
            			top_data + top[0]->offset(n, tmapid));

            } // end col
          } // end dim
          tmapid++;
        } // end mix

				// compute the normalization term

        IdprForward<Dtype><<<CAFFE_GET_BLOCKS(top_height*top_width), CAFFE_CUDA_NUM_THREADS>>>(
        		top_height*top_width,
        		top[0]->gpu_data(),
            n, top_channels, top_height, top_width,
            tmapid, mix_num_,
            normfactor_ + normfactor.offset(n, normcnt));

				// Normalize the summed IDPR map (mixture_num of channels)
				for (int tid = tmapid - 1; tid >= tmapid - mix_num_; --tid) {
					caffe_gpu_div(map_size, top_data + top[0]->offset(n, tid), normfactor_+normfactor.offset(n, normcnt), top_data + top[0]->offset(n, tid));
				}
				normcnt++;
      } // end neighbor
    } // end parts
  } // end sample
}

template <typename Dtype>
void IdprLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	  // top statistics
	  int top_num = top[0]->num();
	  int top_height = top[0]->height();
	  int top_width = top[0]->width();
	  int map_size = top_width*top_height;

		Dtype* normfactor_ = normfactor.mutable_gpu_data();

	  // compute IDPR maps
	  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
		Dtype*        bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype*        top_diff = top[0]->gpu_diff();

	  for (int n = 0; n < top_num; ++n) {
	    int  tmapid = 0;
	    int normcnt = 0;
	    for (int p = 0; p < idpr_global_ids.size(); ++p) { // for each part
	      for (int s = 0; s < idpr_global_ids[p].size(); ++s) { // for each neighbor
	        platero::C3dmat<int>* idpr_mix = idpr_global_ids[p][s];

	        for (int m = 0; m < mix_num_; ++m) {
	          for (int d = 0; d < idpr_mix->get_dims(); ++d) {
	            for (int c = 0; c < idpr_mix->get_cols(); ++c) {
	              // print the m-th row of each dimension
	              int bmapid = idpr_mix->at(m, c, d);

	              // -------sum a diff map
	            	caffe_gpu_axpy (map_size,
	            			Dtype(1),
	            			top_diff + top[0]->offset(n, tmapid),
	            			bottom_diff+bottom[0]->offset(n, bmapid)
	            			);
	            	caffe_gpu_div(map_size,
	            			bottom_diff + bottom[0]->offset(n, bmapid),
	            			normfactor_+normfactor.offset(n, normcnt),
	            			bottom_diff + bottom[0]->offset(n, bmapid));
	            } // end col
	          } // end dim
	          tmapid++;
	        } // end mix

        	normcnt ++;
	      } // end neighbor
	    } // end parts
	  } // end sample
}

INSTANTIATE_LAYER_GPU_FUNCS(IdprLayer);

}  // namespace caffe
