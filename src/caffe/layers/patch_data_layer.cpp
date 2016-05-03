#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>


#include "caffe/common.hpp"
#include "caffe/layers/patch_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
PatchDataLayer<Dtype>::PatchDataLayer(const LayerParameter& param)
: BasePrefetchingDataLayer<Dtype>(param),
  reader_(param, true) {
}

template <typename Dtype>
PatchDataLayer<Dtype>::~PatchDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void PatchDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  np_in_lmdb_ = this->layer_param_.transform_param().np_in_lmdb();
  np_ = this->layer_param_.transform_param().num_parts();
  num_mixtures_ = this->layer_param_.transform_param().num_mixtures();
  use_mixture_ = this->layer_param_.transform_param().use_mixture();
  patch_size_ = this->layer_param_.cpmdata_param().patch_size();
  fg_fraction_ = this->layer_param_.cpmdata_param().fg_fraction();


  LOG(INFO) << "np_in_lmdb_: " << np_in_lmdb_;
  LOG(INFO) << "np_: " << np_;

  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  LOG(INFO) << datum.height() << " " << datum.width() << " " << 3;

  bool force_color = this->layer_param_.cpmdata_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();

  const int patch_num = batch_size*(np_ + round(np_*(1-fg_fraction_)/fg_fraction_));

  if (patch_size_ > 0) {
    top[0]->Reshape(patch_num, 3, patch_size_, patch_size_);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(patch_num, 3, patch_size_, patch_size_);
    }
    this->transformed_data_.Reshape(1, 3, patch_size_, patch_size_);
  } 
  else {
    const int patch_height = this->phase_ != TRAIN ? datum.height() :
        this->layer_param_.cpmdata_param().patch_size();
    const int patch_width = this->phase_ != TRAIN ? datum.width() :
        this->layer_param_.cpmdata_param().patch_size();
    LOG(INFO) << "PREFETCH_COUNT is " << this->PREFETCH_COUNT;
    top[0]->Reshape(patch_num, 3, patch_height, patch_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(patch_num, 3, patch_height, patch_width);
    }
    this->transformed_data_.Reshape(1, 3, patch_height, patch_width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  if (this->output_labels_) {
    int num_parts = np_;
    if (use_mixture_) {
      num_parts = np_*num_mixtures_;
      LOG(INFO) << "NUM of PARTS!!! : " << num_parts;
    }
    vector<int> label_shape(1, patch_num);

    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
//    this->transformed_label_.Reshape(label_shape);
  }

  // read mixture information
  if (this->layer_param_.transform_param().use_mixture()) {
    std::ifstream infile(this->layer_param_.cpmdata_param().mixture_source().c_str());
    CHECK(infile.good()) << "Failed to open mixture file "
        << this->layer_param_.cpmdata_param().mixture_source() << std::endl;

    string hashtag;
    int joint_index, num_mixtures;
    if (!(infile >> hashtag >> joint_index)) {
      LOG(FATAL) << "Mixture file is empty";
    }
    do {
      CHECK_EQ(hashtag, "#");
      // read number of mixtures
      infile >> num_mixtures;
      Clusters cur_cluster;
      cur_cluster.number = num_mixtures;
      for (int i = 0; i < num_mixtures; ++i) {
        Dtype x, y;
        infile >> x >> y;
        cur_cluster.centers.push_back(Point2f(x,y));
      }
      clusters_.push_back(cur_cluster);
    } while (infile >> hashtag >> joint_index);
    // reorder clusters
    if(np_ == 14){
      LOG(ERROR) << "NP_: " << np_;
      vector< Clusters > tmp_clusters;
      int MPI_to_ours[14] = {9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5};

      for(int i=0; i<np_; i++){
        tmp_clusters.push_back(clusters_[MPI_to_ours[i]]);
      }
      clusters_ = tmp_clusters;
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void PatchDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double deque_time = 0;
  double decod_time = 0;
  double trans_time = 0;
  static int cnt = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());


  int half_psize = round(patch_size_/2);

  // Reshape on single input batches for inputs of varying dimension.
  const int batch_size = this->layer_param_.cpmdata_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
//  const int patch_num = batch_size*(np_ + round(np_*(1-fg_fraction_)/fg_fraction_));

  //  LOG(INFO) << "Patch num" << patch_num;

  bool force_color = this->layer_param_.cpmdata_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    Datum& datum = *(reader_.full().peek());
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    batch->data_.Reshape(1, 3,
        datum.height(), datum.width());
    this->transformed_data_.Reshape(1, 3,
        datum.height(), datum.width());
  }

//  LOG(INFO) << "batch->data_: " << batch->data_.shape_string();

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  //  LOG(INFO) << "top_data shape: " << batch->data_.shape_string();

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    deque_time += timer.MicroSeconds();

    timer.Start();
    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
            << "channel sizes. Consider adding a 'force_color' flag to the "
            << "model definition, or rebuild your dataset using "
            << "convert_imageset.";
      }
    }
    decod_time += timer.MicroSeconds();

    // Apply data transformations (mirror, scale, crop...)
    vector<Point2f> patch_centers;
    vector<int> patch_labels;

    timer.Start();
    const int offset_data = batch->data_.offset(item_id);
//    const int offset_label = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset_data);
//    this->transformed_label_.set_cpu_data(top_label + offset_label);

    Mat img_aug = Mat::zeros(patch_size_, patch_size_, CV_8UC3);

    if (datum.encoded()) {
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {  // actually use this one
      this->data_transformer_->Transform_patch(datum, img_aug,
          patch_centers, patch_labels,
          clusters_,
          patch_size_,
          fg_fraction_,
          cnt);
      ++cnt;
    }

    // wyang: Compute crops and labels
    const int channels = img_aug.channels();
    CHECK_EQ(patch_centers.size(), patch_labels.size());

    for (int pid = 0; pid < patch_centers.size(); ++pid) {
      Mat cv_cropped_img = crop_roi(img_aug, patch_centers[pid], half_psize);
      int patch_index = item_id*(np_ + round(np_*(1-fg_fraction_)/fg_fraction_)) + pid;

      // copy the cropped patch into top_data
      for (int h = 0; h < cv_cropped_img.rows; ++h) {
        for (int w = 0; w < cv_cropped_img.cols; ++w) {
          Vec3b& cv_cropped_img_rgb = cv_cropped_img.at<Vec3b>(h, w);

          for (int c = 0; c < channels; ++c) {
            int top_index = ((patch_index * channels + c) * patch_size_+ h) * patch_size_ + w;
            CHECK_LT(top_index, batch->data_.count());
            Dtype pixel = cv_cropped_img_rgb[c];
            top_data[top_index] = (pixel - 128) / 256;
//            LOG(INFO) << top_data[top_index];
          }
        }
      }
      // copy label to top_data
      if (this->output_labels_) {
        top_label[patch_index] = patch_labels[pid] - 1; // index from 0
//         LOG(INFO) << "top_label["<< patch_index << "]: " << top_label[patch_index] ;
      }

      // visualize the cropped patch
      if (this->layer_param_.transform_param().visualize()) {
        char imagename [100];
        // sprintf(imagename, "cache_patch/pid_%02d_%04d.jpg", pid, cnt);  // start by part
        // sprintf(imagename, "cache_patch/%04d_pid_%02d.jpg", cnt, pid);  // start by person
        sprintf(imagename, "cache_patch/mix_%04d_%04d_pid_%02d.jpg", patch_labels[pid], cnt, pid);  // start by mix id
        imwrite(imagename, cv_cropped_img);
      }
    }

    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  batch_timer.Stop();
  //  LOG(INFO) << "prefetch done";

#ifdef BENCHMARK_DATA
  LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms.";
  LOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
}

template<typename Dtype>
Mat PatchDataLayer<Dtype>::crop_roi(const Mat& img, Point2f patch_center, int half_psize) {
  Mat cropped_image(2*half_psize, 2*half_psize, CV_8UC3, Scalar(128,128,128));
  int height = img.rows;
  int width = img.cols;
  int x1 = round(patch_center.x) - half_psize;
  int y1 = round(patch_center.y) - half_psize;

  int xstart = min(max(0, x1), width);
  int ystart = min(max(0, y1), height);
  int xend = max(min(width, x1+2*half_psize), 0);
  int yend = max(min(height, y1+2*half_psize), 0);
  int patch_off_x = (x1 < 0) ? -x1: 0;
  int patch_off_y = (y1 < 0) ? -y1: 0;

  for (int i = ystart, patch_i=0; i < yend; ++i, ++patch_i) {  // rows
    for (int j = xstart, patch_j = 0; j < xend; ++j, ++patch_j) {  // cols
      // For debug we can use:
      // CHECK_LT(patch_off_x+patch_j, 72);
      // CHECK_LT(patch_off_y+patch_i, 72);
      cropped_image.at<Vec3b>(patch_off_x+patch_j, patch_off_y+patch_i) = img.at<Vec3b>(i,j);
    }
  }

  if ((patch_off_x != 0 || patch_off_y != 0) && this->layer_param_.transform_param().visualize()) {
    char imagename [100];
    sprintf(imagename, "cache_patch/crop_roi.jpg");  // start by mix id
    imwrite(imagename, cropped_image);
  }

  return cropped_image;
}
INSTANTIATE_CLASS(PatchDataLayer);
REGISTER_LAYER_CLASS(PatchData);

}  // namespace caffe
