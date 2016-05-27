#include <vector>

#include "caffe/layers/gated_conv_layer.hpp"
#include <iostream>

using namespace std;

#define out LOG(INFO)
namespace caffe {

template <typename Dtype>
void GatedConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom.size(), top.size() + 1) << "bottom.size() == top.size()+1";
  bottom_num_ = bottom.size();
  gate_num_ = this->blobs_[0]->shape(0)*this->blobs_[0]->shape(1);
}

template <typename Dtype>
void GatedConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  gates_ = bottom[bottom_num_-1];
  gated_blobs_.resize(this->num_);  // batchsize
  top_no_gate_.resize(top.size());  // batchsize

  for (int n = 0; n < top.size(); ++n) {
    top_no_gate_[n].reset(new Blob<Dtype>());
    top_no_gate_[n]->Reshape(top[n]->shape());
  }
  // check gate size
//  LOG(INFO) << "Gate shape: "  << gates_->shape_string();
  CHECK(bottom[0]->shape(0) == gates_->shape(0));  // check batch number
  CHECK(gate_num_ == gates_->shape(1));  // check gate number
//LOG(INFO) << "gate channel : " << gates_->shape(this->channel_axis_);


  // precompute gated_blobs for later use
  for (int n = 0; n < this->num_; ++n) {
    // copy weights to gated_weights (ref: layer.hpp)
    gated_blobs_[n].reset(new Blob<Dtype>());
    gated_blobs_[n]->Reshape(this->blobs_[0]->shape());
//    LOG(INFO) << gated_blobs_[n]->diff_at(n, 0, 0, 0);
//    gated_blobs_[n]->CopyFrom((*this->blobs_[0]));
    Dtype*        gated_map = gated_blobs_[n]->mutable_cpu_data();
    const int     weight_size = this->blobs_[0]->count(this->channel_axis_+1);
    caffe_set(gated_blobs_[n]->count(), Dtype(1), gated_map);
    // augment scalar gates to gate maps
    for (int i = 0; i < this->blobs_[0]->shape(0); ++i) {
      for (int j = 0; j < this->blobs_[0]->shape(1); ++j) {
        int offset = gated_blobs_[n]->offset(i, j);
        caffe_scal(weight_size, gates_->data_at(n, i*j, 0, 0), gated_map + offset);
      }
    }
  }

  vector<int> sz;
  int spatial_dim = gated_blobs_[0]->count()/(this->channels_*gated_blobs_[0]->shape(0));
//  LOG(INFO) << "spatial_dim: " << spatial_dim;

//  sz.push_back(gates_->shape(this->channel_axis_));
//  spatial_sum_multiplier_.Reshape(sz);
//  Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
//  caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
//
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz.push_back(spatial_dim);
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }
}


template <typename Dtype>
void GatedConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // forward convolution with gated_weight
  for (int i = 0; i < bottom.size()-1; ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    Dtype* top_no_gate_data = top_no_gate_[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      // compute gated_weight
      Blob<Dtype> cur_gate_blob;
      cur_gate_blob.ReshapeLike(*gated_blobs_[n]);
      Dtype* gated_weight = cur_gate_blob.mutable_cpu_data();
      caffe_mul(cur_gate_blob.count(), gated_blobs_[n]->cpu_data(), this->blobs_[0]->cpu_data(), gated_weight);
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, gated_weight,  // weight -> gated_weight
          top_data + n * this->top_dim_);
//      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,  // convolution without gates
//          top_no_gate_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void GatedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int weight_size = this->blobs_[0]->count(this->channel_axis_+1);
  const int weight_count = this->blobs_[0]->count();

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      Blob<Dtype> cum_gate_blob;
      cum_gate_blob.ReshapeLike(*this->blobs_[0]);
      for (int n = 0; n < this->num_; ++n) {
        Dtype* cum_gate_blob_diff = cum_gate_blob.mutable_cpu_diff();
        caffe_set(weight_count, Dtype(0), cum_gate_blob_diff);
//        const Dtype* gated_weight = gated_blobs_[n]->cpu_data();
        // compute gated_weight
        Blob<Dtype> cur_gate_blob;
        cur_gate_blob.ReshapeLike(*gated_blobs_[n]);
        Dtype* gated_weight = cur_gate_blob.mutable_cpu_data();
        caffe_mul(weight_count, gated_blobs_[n]->cpu_data(), this->blobs_[0]->cpu_data(), gated_weight);
        // gradient w.r.t. weight. Note that we will accumulate diffs (delta * x . gate).
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          caffe_mul(weight_count, weight_diff, gated_weight, weight_diff);
          // gradient w.r.t. gates
          Dtype* gated_blob_diff = gated_blobs_[n]->mutable_cpu_diff();
          caffe_mul(weight_count, weight_diff, this->blobs_[0]->cpu_data(), cum_gate_blob_diff);
//          caffe_axpy(weight_count, 1, gated_blob_diff, top_data);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, gated_weight,  // weight -> gated_weight
              bottom_diff + n * this->bottom_dim_);
        }
        // Marginalize gate diff
        if (propagate_down[bottom_num_-1]) {

//          LOG(INFO) << "weight_size: " << weight_size << " weight_count/weight_size: " << weight_count/weight_size;
          caffe_cpu_gemv<Dtype>(CblasNoTrans, weight_count/weight_size, weight_size,
              1, cum_gate_blob.cpu_diff(),
              spatial_sum_multiplier_.cpu_data(), 0.,
              bottom[bottom_num_-1]->mutable_cpu_diff()+ bottom[bottom_num_-1]->offset(n));
        }
      }  // end of n
    }

//    const Dtype* tmp_diff = bottom[bottom_num_-1]->cpu_diff();
//    for (int ii = 0; ii < bottom[bottom_num_-1]->count(); ++ii) {
//      cout << tmp_diff[ii] << " ";
//    }
//    cout << endl;
//    Dtype* bias_diff = this->blobs_[0]->mutable_cpu_diff();
//        for (int ii = 0; ii < this->blobs_[0]->count(); ++ii) {
//          cout << bias_diff[ii] << " ";
//        }
//        cout << endl;
  }
}

//template <typename Dtype>
//void GatedConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
//    const Dtype* output, Dtype* weights) {
//  BaseConvolutionLayer<Dtype>::weight_cpu_gemm(input, output, weights);
//}

#ifdef CPU_ONLY
STUB_GPU(GatedConvolutionLayer);
#endif

INSTANTIATE_CLASS(GatedConvolutionLayer);
REGISTER_LAYER_CLASS(GatedConvolution);

}  // namespace caffe
