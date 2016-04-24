// This program converts a set of *.mats to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_matset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the mats, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.mat 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of mats and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(varname, "",
        "The variable name to be read in the mat file");
DEFINE_int32(channels, 2, "Array channels.");
DEFINE_int32(height, -1, "Array height.");
DEFINE_int32(width, -1, "Array width.");
//DEFINE_string(sizes, "", "Array of integers specifying an n-dimensional array shape (seperate by ,).");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");

//// Parse sizes
//static void get_sizes(vector<int>* sizes) {
//	// sizes and ndims should match
//  if (FLAGS_sizes.size()) {
//    vector<string> strings;
//    boost::split(strings, FLAGS_sizes, boost::is_any_of(","));
//    for (int i = 0; i < strings.size(); ++i) {
//    	sizes->push_back(boost::lexical_cast<int>(strings[i]));
//    }
//  }
//  CHECK_EQ(sizes->size(), FLAGS_ndims);
//}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of mats(type: double) to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_matset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_matset");
    return 1;
  }

  const bool check_size = FLAGS_check_size;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " mats.";

//  int ndims = FLAGS_ndims;
//  vector<int> sizes;
//  get_sizes(&sizes);

//  LOG(INFO) << "Sizes: " << sizes.size();

//  for(int i = 0; i < sizes.size(); ++i) {
//  	LOG(INFO) << "\t " << i << ": " << sizes[i];
//  }
  int channels = FLAGS_channels;
  int height = FLAGS_height;
  int width = FLAGS_width;

  const string varname = FLAGS_varname;

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    status = ReadMatlabToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, channels, height, width, varname, &datum);
    if (status == false) {
        LOG(INFO) << "STATUS FAIL: " << lines[line_id].first;
        continue;
    }
    if (check_size) {
    	LOG(INFO) << datum.channels() <<" " <<  datum.height() << " " << datum.width();
//      if (!data_size_initialized) {
//        data_size = datum.channels() * datum.height() * datum.width();
//        data_size_initialized = true;
//      } else {
//        const std::string& data = datum.data();
//        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
//            << data.size();
//      }
    }
    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return 0;
}
