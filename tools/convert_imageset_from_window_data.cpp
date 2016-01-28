// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_layers.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
		"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
		"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
		"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
		"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
		"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
		"Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_double(fg_threshold, 0.5, "foreground threshold of windows");
DEFINE_double(bg_threshold, 0.5, "background threshold of windows");
DEFINE_string(root_folder, "",
		"Optional: root folder of images.");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
			"format used as input for Caffe.\n"
			"Usage:\n"
			"    convert_imageset [FLAGS] LISTFILE DB_NAME\n"
			"The ImageNet dataset for the training demo is at\n"
			"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	LOG(INFO) <<  argc ;

	if (argc < 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset_from_window_data");
		return 1;
	}

	// ---- Window data file params
	vector<std::pair<std::string, vector<int> > > image_database_;
	vector<vector<float> > 												fg_windows_;
	vector<vector<float> > 												bg_windows_;
	enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const double fg_threshold = FLAGS_fg_threshold;
	const double bg_threshold = FLAGS_bg_threshold;
	const string root_folder = FLAGS_root_folder;

	// ---- Read Window data file
	std::ifstream infile(argv[1]);
	map<int, int> label_hist;
	label_hist.insert(std::make_pair(0, 0));

	string hashtag;
	int image_index, channels;
	if (!(infile >> hashtag >> image_index)) {
		LOG(FATAL) << "Window file is empty";
	}
	do {
		CHECK_EQ(hashtag, "#");
		// read image path
		string image_path;
		infile >> image_path;
		image_path = root_folder + image_path;
		// read image dimensions
		vector<int> image_size(3);
		infile >> image_size[0] >> image_size[1] >> image_size[2];
		channels = image_size[0];
		image_database_.push_back(std::make_pair(image_path, image_size));

		// read each box
		int num_windows;
		infile >> num_windows;

		for (int i = 0; i < num_windows; ++i) {
			int label, x1, y1, x2, y2;
			float overlap;
			infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;

			vector<float> window(NUM);
			window[IMAGE_INDEX] = image_index;
			window[LABEL] = label;
			window[OVERLAP] = overlap;
			window[X1] = x1;
			window[Y1] = y1;
			window[X2] = x2;
			window[Y2] = y2;

			// add window to foreground list or background list
			if (overlap >= fg_threshold) {
				int label = window[LABEL];
				CHECK_GT(label, 0);
				fg_windows_.push_back(window);
				label_hist.insert(std::make_pair(label, 0));
				label_hist[label]++;
			} else if (overlap < bg_threshold) {
				// background window, force label and overlap to 0
				window[LABEL] = 0;
				window[OVERLAP] = 0;
				bg_windows_.push_back(window);
				label_hist[0]++;
			}
		}

		if (image_index % 100 == 0) {
			LOG(INFO) << "num: " << image_index << " "
					<< image_path << " "
					<< image_size[0] << " "
					<< image_size[1] << " "
					<< image_size[2] << " "
					<< "windows to process: " << num_windows;
		}
	} while (infile >> hashtag >> image_index);


  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }


  LOG(INFO) << "Number of images: " << image_index+1;

//	// ----
//	if (FLAGS_shuffle) {
//		// randomly shuffle data
//		LOG(INFO) << "Shuffling data";
//		shuffle(image_database_.begin(), image_database_.end());
//	}
//
//	if (encode_type.size() && !encoded)
//		LOG(INFO) << "encode_type specified, assuming encoded=true.";
//
//	int resize_height = std::max<int>(0, FLAGS_resize_height);
//	int resize_width = std::max<int>(0, FLAGS_resize_width);
//
//	// Create new DB
//	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
//	db->Open(argv[3], db::NEW);
//	scoped_ptr<db::Transaction> txn(db->NewTransaction());
//
//	// Storing to db
//	Datum datum;
//	int count = 0;
//	const int kMaxKeyLength = 256;
//	char key_cstr[kMaxKeyLength];
//	int data_size = 0;
//	bool data_size_initialized = false;
//
//	for (int image_id = 0; image_id < image_index; ++image_id) {
//		bool status;
//		std::string enc = encode_type;
//		// load the image containing the window
//		pair<std::string, vector<int> > image = image_database_[image_id];
//
//		if (encoded && !enc.size()) {
//			// Guess the encoding type from the file name
//			string fn = image.first;
//			size_t p = fn.rfind('.');
//			if ( p == fn.npos )
//				LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
//			enc = fn.substr(p);
//			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
//		}
//
//		cv::Mat cv_img;
//
//		        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
//		        if (!cv_img.data) {
//		          LOG(ERROR) << "Could not open or find file " << image.first;
//		          return;
//		        }
//
//		      const int channels = cv_img.channels();
//
//		      // crop window out of image and warp it
//		      int x1 = window[WindowDataLayer<Dtype>::X1];
//		      int y1 = window[WindowDataLayer<Dtype>::Y1];
//		      int x2 = window[WindowDataLayer<Dtype>::X2];
//		      int y2 = window[WindowDataLayer<Dtype>::Y2];
//
//		      int pad_w = 0;
//		      int pad_h = 0;
//		      if (context_pad > 0 || use_square) {
//		        // scale factor by which to expand the original region
//		        // such that after warping the expanded region to crop_size x crop_size
//		        // there's exactly context_pad amount of padding on each side
//		        Dtype context_scale = static_cast<Dtype>(crop_size) /
//		            static_cast<Dtype>(crop_size - 2*context_pad);
//
//		        // compute the expanded region
//		        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
//		        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
//		        Dtype center_x = static_cast<Dtype>(x1) + half_width;
//		        Dtype center_y = static_cast<Dtype>(y1) + half_height;
//		        if (use_square) {
//		          if (half_height > half_width) {
//		            half_width = half_height;
//		          } else {
//		            half_height = half_width;
//		          }
//		        }
//		        x1 = static_cast<int>(round(center_x - half_width*context_scale));
//		        x2 = static_cast<int>(round(center_x + half_width*context_scale));
//		        y1 = static_cast<int>(round(center_y - half_height*context_scale));
//		        y2 = static_cast<int>(round(center_y + half_height*context_scale));
//
//		        // the expanded region may go outside of the image
//		        // so we compute the clipped (expanded) region and keep track of
//		        // the extent beyond the image
//		        int unclipped_height = y2-y1+1;
//		        int unclipped_width = x2-x1+1;
//		        int pad_x1 = std::max(0, -x1);
//		        int pad_y1 = std::max(0, -y1);
//		        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
//		        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
//		        // clip bounds
//		        x1 = x1 + pad_x1;
//		        x2 = x2 - pad_x2;
//		        y1 = y1 + pad_y1;
//		        y2 = y2 - pad_y2;
//		        CHECK_GT(x1, -1);
//		        CHECK_GT(y1, -1);
//		        CHECK_LT(x2, cv_img.cols);
//		        CHECK_LT(y2, cv_img.rows);
//
//		        int clipped_height = y2-y1+1;
//		        int clipped_width = x2-x1+1;
//
//		        // scale factors that would be used to warp the unclipped
//		        // expanded region
//		        Dtype scale_x =
//		            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
//		        Dtype scale_y =
//		            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);
//
//		        // size to warp the clipped expanded region to
//		        cv_crop_size.width =
//		            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
//		        cv_crop_size.height =
//		            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
//		        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
//		        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
//		        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
//		        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));
//
//		        pad_h = pad_y1;
//		        // if we're mirroring, we mirror the padding too (to be pedantic)
//		        if (do_mirror) {
//		          pad_w = pad_x2;
//		        } else {
//		          pad_w = pad_x1;
//		        }
//
//		        // ensure that the warped, clipped region plus the padding fits in the
//		        // crop_size x crop_size image (it might not due to rounding)
//		        if (pad_h + cv_crop_size.height > crop_size) {
//		          cv_crop_size.height = crop_size - pad_h;
//		        }
//		        if (pad_w + cv_crop_size.width > crop_size) {
//		          cv_crop_size.width = crop_size - pad_w;
//		        }
//		      }
//
//		      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
//		//      //---------------------------
//		//      LOG(INFO) << "src: "<< image.first << "( " << cv_img.rows << "," << cv_img.cols << " )";
//		//      LOG(INFO) << "roi:" << x1 << ", " << y1 << ", " << x2 << ", " << y2;
//
//		      cv::Mat cv_cropped_img = cv_img(roi);
//		      cv::resize(cv_cropped_img, cv_cropped_img,
//		          cv_crop_size, 0, 0, cv::INTER_LINEAR);
//
//		      // horizontal flip at random
//		      if (do_mirror) {
//		        cv::flip(cv_cropped_img, cv_cropped_img, 1);
//		      }
//
//		////////////////////
//		status = ReadImageToDatum(root_folder + image.first,
//				image.second, resize_height, resize_width, is_color,
//				enc, &datum);
//		if (status == false) continue;
//		if (check_size) {
//			if (!data_size_initialized) {
//				data_size = datum.channels() * datum.height() * datum.width();
//				data_size_initialized = true;
//			} else {
//				const std::string& data = datum.data();
//				CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
//						<< data.size();
//			}
//		}
//		// sequential
//		int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
//				lines[line_id].first.c_str());
//
//		// Put in db
//		string out;
//		CHECK(datum.SerializeToString(&out));
//		txn->Put(string(key_cstr, length), out);
//
//		if (++count % 1000 == 0) {
//			// Commit db
//			txn->Commit();
//			txn.reset(db->NewTransaction());
//			LOG(INFO) << "Processed " << count << " files.";
//		}
//	}
//	// write the last batch
//	if (count % 1000 != 0) {
//		txn->Commit();
//		LOG(INFO) << "Processed " << count << " files.";
//	}
	return 0;
}
