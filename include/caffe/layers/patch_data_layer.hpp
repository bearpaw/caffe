#ifndef CAFFE_PATCH_DATA_LAYER_HPP_
#define CAFFE_PATCH_DATA_LAYER_HPP_


#include <vector>
#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Prepare cropped patches for pose data
 *    This data layer shares parameters with CPMDataLayer
 */
template <typename Dtype>
class PatchDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit PatchDataLayer(const LayerParameter& param);
    virtual ~PatchDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual inline bool ShareInParallel() const { return false; }
    virtual inline const char* type() const { return "CPMData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    virtual void load_batch(Batch<Dtype>* batch);
    Mat crop_roi(const Mat& img_aug, Point2f patch_centers, int half_psize);

    DataReader reader_;
//    Blob<Dtype> transformed_label_; // add another blob
    vector< Clusters > clusters_;  // mixture information
    int   np_in_lmdb_;
    int   np_;
    int   num_mixtures_;
    int   use_mixture_;
    int   patch_size_;
    Dtype fg_fraction_;
    //Blob<Dtype> transformed_label_all_; // all peaks, including others
};

}  // namespace caffe

#endif  // CAFFE_PATCH_DATA_LAYER_HPP_
