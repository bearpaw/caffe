#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/label_map_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
LabelMapLayer<Dtype>::~LabelMapLayer<Dtype>() {
	this->StopInternalThread();
}

template <typename Dtype>
void LabelMapLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	new_height_ = this->layer_param_.label_map_param().new_height();
	new_width_  = this->layer_param_.label_map_param().new_width();
	num_output_  = this->layer_param_.label_map_param().num_output();
	sigma_ = this->layer_param_.label_map_param().sigma();
	float_min_ = this->layer_param_.label_map_param().float_min();

	CHECK((new_height_ > 0 && new_width_ > 0)) << "Current implementation requires "
			"new_height and new_width to be set.";
	// Read the file with filenames and labels
	const string& source = this->layer_param_.label_map_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	vector<tuple > points;
	std::string line_id;
	int pts_num, map_id, x, y;
	while (infile >> line_id >> pts_num) {
		//  	LOG(INFO) << line_id << " " << pts_num;
		points.clear();
		for (int p = 0; p < pts_num; ++p) {
			infile >> map_id >> x >> y;
			points.push_back(tuple(map_id, x, y));
		}
		lines_.push_back(points);
	}
	if (this->layer_param_.label_map_param().shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleData();
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.label_map_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand() %
				this->layer_param_.label_map_param().rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
		lines_id_ = skip;
	}
	// Initialize the top shape
	vector<int> top_shape (4, 0);
	top_shape[0] = 1;
	top_shape[1] = num_output_;
	top_shape[2] = new_height_;
	top_shape[3] = new_width_;
	this->transformed_data_.Reshape(top_shape);
	// Reshape prefetch_data and top[0] according to the batch_size.
	batch_size_ = this->layer_param_.label_map_param().batch_size();
	CHECK_GT(batch_size_, 0) << "Positive batch size required";
	top_shape[0] = batch_size_;

	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}
	top[0]->Reshape(top_shape);

	LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
	//  // label
	//  vector<int> label_shape(1, batch_size_);
	//  top[1]->Reshape(label_shape);
	//  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	//    this->prefetch_[i].label_.Reshape(label_shape);
	//  }
	////  LOG(INFO) << "Layer Set up done";
}

template <typename Dtype>
void LabelMapLayer<Dtype>::ShuffleData() {
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void LabelMapLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	//	LOG(INFO) << "load_batch";
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
//	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	// Initialize the top shape
	vector<int> top_shape (4, 0);
	top_shape[0] = 1;
	top_shape[1] = num_output_;
	top_shape[2] = new_height_;
	top_shape[3] = new_width_;
	this->transformed_data_.Reshape(top_shape);
	// Reshape batch according to the batch_size.
	top_shape[0] = batch_size_;
	batch->data_.Reshape(top_shape);


	//	LOG(INFO) << batch->data_.num() << ", "
	//			<< batch->data_.channels() << ", "
	//			<< batch->data_.height() << ", "
	//			<< batch->data_.width();

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	caffe_set(batch->data_.count(), Dtype(0), prefetch_data);
	//  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
	vector<tuple > points;

	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size_; ++item_id) {
		// get a blob
		timer.Start();
		CHECK_GT(lines_size, lines_id_);
		// create gaussian maps
		points.clear();
		points = lines_[lines_id_];
		const int points_size = points.size();
//		LOG(INFO) << "Point size: " << points_size;

		for (int p = 0; p < points_size; ++p) {
			int map_idx = points[p].x;
			int x = points[p].y;
			int y = points[p].z;
			//    	LOG(INFO) << points[p].x << " " << points[p].y << " " << points[p].z;
			Dtype* gmask = prefetch_data + batch->data_.offset(item_id, map_idx);
			for (int h = 0; h < new_height_; ++h) {
				for (int w = 0; w < new_width_; ++w) {
					int index = h * new_width_ + w;
					double distance = (w-x)*(w-x) + (h-y)*(h-y);
					Dtype label = exp(-distance/(2*sigma_*sigma_));
					if (label > float_min_) gmask[index] = label;
				}
			}

			// debug
			if(0)
			{
				char ss1[1010];
				char datass[1010];
				cv::Mat imgout(cv::Size(new_width_, new_height_), CV_8UC3);
				for(int h = 0; h < new_height_; h ++ )
					for(int w = 0; w < new_width_; w ++)
					{
						for(int ch = 0; ch < 3; ch ++) {
							imgout.at<cv::Vec3b>(h, w)[ch] =(uchar)( gmask[h * new_width_ + w] * 255 );
//							LOG(INFO) << gmask[h * new_width_ + w];
						}
					}

				sprintf(ss1,"/home/wyang/tmp/label_map_layer/%d_labelmap.jpg", p);
				imwrite(ss1, imgout);

			}

		}
		read_time += timer.MicroSeconds();
		// go to the next iter
		lines_id_++;
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.label_map_param().shuffle()) {
				ShuffleData();
			}
		}
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch:    " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "Create label time: " << read_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(LabelMapLayer);
REGISTER_LAYER_CLASS(LabelMap);

}  // namespace caffe
#endif  // USE_OPENCV
