#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

struct Clusters {
  vector<Point2f> centers;
  int number;
  int size() {
    return number;
  }
};

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
  void Transform_nv(const Datum& datum, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob,
                    vector< Clusters >& clusters_, int cnt); //image and label
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  struct Joints {
    vector<Point2f> joints;
    vector<int> isVisible;
  };

  struct MetaData {
    string dataset;
    Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)

    vector<Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople
  };


  void Transform_patch(const Datum& datum, Mat& img_aug, vector<Point2f>& patch_centers, vector<int>& patch_labels,
        vector< Clusters >& clusters_, int patch_size, Dtype fg_fraction, int cnt); // wyang for patch

  void generateLabelMap(Dtype*, Mat&, MetaData meta);
  void generateLabelMap(Dtype*, Mat&, MetaData meta, vector< Clusters >& clusters_);
  void generateLabel(vector<Point2f>& patch_centers, vector<int>& patch_labels,
      int patch_size, Dtype fg_fraction, Mat& img_aug, MetaData meta, vector< Clusters >& clusters_);
  void visualize(Mat& img, MetaData meta, AugmentSelection as);

  bool augmentation_flip(Mat& img, Mat& img_aug, MetaData& meta);
  float augmentation_rotate(Mat& img_src, Mat& img_aug, MetaData& meta);
  float augmentation_scale(Mat& img, Mat& img_temp, MetaData& meta);
  Size augmentation_croppad(Mat& img_temp, Mat& img_aug, MetaData& meta);
  void RotatePoint(Point2f& p, Mat R);
  bool onPlane(Point p, Size img_size);
  void swapLeftRight(Joints& j);
  void SetAugTable(int numData);

  int np_in_lmdb;
  int np;
  bool is_table_set;
  bool use_mixture_;
  bool aug_midway_;
  int  num_mixtures_;
  vector<vector<float> > aug_degs;
  vector<vector<int> > aug_flips;

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  void Transform(const Datum& datum, Dtype* transformed_data);
  void Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, vector< Clusters >& clusters_, int cnt);
  void ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1);
  void TransformMetaJoints(MetaData& meta, vector< Clusters >& clusters);
  void TransformJoints(Joints& joints);
  void clahe(Mat& img, int, int);
  void putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData);
  int  assign_cluster_label(Point2f point, const Clusters& cluster);

  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  vector<pair<Point2f, Point2f> > trans;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
