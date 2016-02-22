#ifndef CAFFE_LABEL_MAP_LAYER_HPP_
#define CAFFE_LABEL_MAP_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
struct tuple {
	int x;
	int y;
	int z;
	tuple(int x, int y, int z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class LabelMapLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit LabelMapLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~LabelMapLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelMap"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleData();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<vector<tuple > > lines_;
  int lines_id_;
  int num_output_;
  int new_height_;
  int new_width_;
  int batch_size_;
  float sigma_;
  float float_min_;
};


}  // namespace caffe

#endif  // CAFFE_LABEL_MAP_LAYER_HPP_
