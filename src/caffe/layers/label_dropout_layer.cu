#include <cfloat>
#include <vector>

#include "caffe/layers/label_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


int my_gpu_random(int i) { return caffe_rng_rand() % i; }

template <typename Dtype>
void LabelDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // set mask
  // TODO: please implement the GPU version
  set_mask(bottom);
  // copy data
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype * bottom_data = bottom[0]->gpu_data();
  caffe_copy(top[0]->count(), bottom_data, top_data);
//  LOG(INFO) << "forward GPU";
}


template <typename Dtype>
void LabelDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_mul(bottom[0]->count(), top[0]->gpu_diff(), mask_.gpu_data(), bottom_diff);
//  LOG(INFO) << "backward GPU";
}


/**
 * TODO: not implemented
 */
template<typename Dtype>
void LabelDropoutLayer<Dtype>::set_gpu_mask(const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  int dim = bottom[0]->count() / bottom[0]->num();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int out_channel = bottom[0]->channels();
  int out_height = bottom[0]->height();
  int out_width = bottom[0]->width();
  int mapsize = out_height * out_width;

  Dtype* mask_data = mask_.mutable_gpu_data();

  vector<vector<pair<float, int> > > negpairs(out_channel);
  vector<vector<int> > sid1(out_channel);
  vector<vector<int> > sid2(out_channel);
  vector<int> pos_num(out_channel);
  vector<int> neg_num(out_channel);

  LOG(INFO) << "negpairs: " << negpairs.size();
  caffe_gpu_set(count, Dtype(0), mask_data);

//  for (int i = 0; i < num; i++) {
//    for (int j = 0; j < out_channel; j++) {
//      negpairs.clear();
//      sid1.clear();
//      sid2.clear();
//      int pos_num = 0;
//      int neg_num = 0;
//      for (int k = 0; k < mapsize; k++) {
//        int nowid = i * dim + j * mapsize + k;
//        if (label[nowid] > 0) {
//          mask_data[nowid] = 1;
//          pos_num++;
//        }
//        if (label[nowid] == 0) {
//          float ts = fabs(bottom_data[nowid]);
//          negpairs.push_back(make_pair(ts, nowid));
//          neg_num++;
//        }
//      }
//      int use_neg_num = pos_num * drop_neg_ratio;
//      // for all zero maps
//      if (pos_num == 0 && drop_on_zero) {
//        use_neg_num = neg_num * drop_on_zero_ratio;
//      }
//
//      if (use_neg_num >= neg_num) {
//        for (int k = 0; k < negpairs.size(); k++) {
//          mask_data[negpairs[k].second] = 1;
//        }
//        continue;
//      }
//
//      sort(negpairs.begin(), negpairs.end());
//      for (int k = 0; k < use_neg_num; k++) {
//        sid1.push_back(negpairs[neg_num - k - 1].second);
//      }
//      for (int k = 0; k < neg_num - use_neg_num; k++) {
//        sid2.push_back(negpairs[k].second);
//      }
//      std::random_shuffle(sid1.begin(), sid1.end(), my_gpu_random);
//      int hardNum = use_neg_num * hard_ratio;
//      int randNum = use_neg_num * rand_ratio;
//
//      for (int k = 0; k < hardNum; k++) {
//        mask_data[sid1[k]] = 1;
//      }
//      for (int k = 0; k < randNum; k++) {
//        sid2.push_back(sid1[hardNum + k]);
//      }
//      std::random_shuffle(sid2.begin(), sid2.end(), my_gpu_random);
//      for (int k = 0; k < randNum; k++) {
//        mask_data[sid2[k]] = 1;
//      }
//
//    }
//
//  }

}


INSTANTIATE_LAYER_GPU_FUNCS(LabelDropoutLayer);

}  // namespace caffe
