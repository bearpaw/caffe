#ifndef CAFFE_LABEL_DROPOUT_LAYER_HPP_
#define CAFFE_LABEL_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute masked predictions.
 * Mask = 1 means the predictions should be keeped (compute loss and diff here).
 * Mask = 0 means the predictions should be masked (do not compute loss and diff here).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class LabelDropoutLayer : public Layer<Dtype> {
 public:
  explicit LabelDropoutLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelDropout"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void set_mask(const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> mask_;
  Dtype drop_neg_ratio;
  Dtype hard_ratio;
  Dtype rand_ratio;
  Dtype hard_ratio_zero;
  Dtype rand_ratio_zero;
};

}  // namespace caffe

#endif  // CAFFE_LABEL_DROPOUT_LAYER_HPP_
