#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <math.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <sstream>

# define Vec3DElem vector<platero::C3dmat<int>* > 
# define C3DMAT platero::C3dmat<int>

namespace caffe {

#define INF 1E20

template <typename Dtype>
static inline Dtype square(Dtype x) { return x*x; }


template <typename Dtype>
void MessagePassingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top.size(), 1)
	      << "There should be only 1 tops for this layer";
  const MessagePassingParameter& params = this->layer_param_.message_passing_param();
  // read number of mixtures
  mix_num_ = params.mix_num();
  //  read parents
  if ( params.parents().size() != 0 ) {
    pa_.clear();
    std::copy(params.parents().begin(),
    		params.parents().end(),
          std::back_inserter(pa_));
  }
  // read current child
  child_ = params.child();
	parent_ = pa_[child_-1];
  // Read mean value of each neighbor
	const string& source = this->layer_param_.message_passing_param().mean_file();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	Dtype mean_x, mean_y;
	while (infile >> mean_x >> mean_y) {
		 meanvals_.push_back(std::make_pair(mean_x, mean_y));
	}

  // get IDs
  get_IDs();
  // Initialize and fill the weights
  // Only four weights for quadratic form distance feature
  this->blobs_.resize(1); // only weight no bias
  vector<int> weight_shape(2);
  weight_shape[0] = bottom[1]->channels();
	weight_shape[1] = 4;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
      this->layer_param_.message_passing_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
}

/**
 * top[0] = bottom[0]: score
 */
template <typename Dtype>
void MessagePassingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//---------- top shape
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int channels = mix_num_*mix_num_;
  //---------- full dt score map
  score_.Reshape(num, channels, height, width);
  caffe_set(score_.count(), Dtype(0), score_.mutable_cpu_data());
  //---------- full Ix
  Ix_.Reshape(num, channels, height, width);
  caffe_set(Ix_.count(), Dtype(0), Ix_.mutable_cpu_data());
  //---------- full Iy
  Iy_.Reshape(num, channels, height, width);
  caffe_set(Iy_.count(), Dtype(0), Iy_.mutable_cpu_data());
  //---------- set maxout index
  max_idx_.Reshape(num, 1, height, width);
  int* mask = max_idx_.mutable_cpu_data();
  caffe_set(max_idx_.count(), -1, mask);
  //---------- max dt score map
  max_score_.Reshape(num, 1, height, width);
  caffe_set(max_score_.count(), Dtype(-10000), max_score_.mutable_cpu_data());
  //---------- Iy
  max_Iy_.Reshape(num, 1, height, width);
  caffe_set(max_Iy_.count(), Dtype(0), max_Iy_.mutable_cpu_data());
  //---------- Ix
  max_Ix_.Reshape(num, 1, height, width);
  caffe_set(max_Ix_.count(), Dtype(0), max_Ix_.mutable_cpu_data());
  //---------- middle diff
  mid_diff_.Reshape(num, channels, height, width);
  caffe_set(mid_diff_.count(), Dtype(0), mid_diff_.mutable_cpu_data());
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


	LOG(INFO) << "Weight shape :" << this->blobs_[0]->shape_string();
	std::stringstream outstream;
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  // get input maps
  const Dtype*  app_map = bottom[0]->cpu_data();
  const Dtype*  def_map = bottom[1]->cpu_data();
  // get dt maps
  Dtype*        score_ptr = score_.mutable_cpu_data(); // dt map
  Dtype*        Ix_ptr = Ix_.mutable_cpu_data();  // Ix
  Dtype*        Iy_ptr = Iy_.mutable_cpu_data();  // Iy
  // get output maps
  Dtype* 				max_score_ptr = max_score_.mutable_cpu_data(); // max score
  Dtype* 				max_Ix_ptr = max_Ix_.mutable_cpu_data(); // max Ix
  Dtype* 				max_Iy_ptr = max_Iy_.mutable_cpu_data(); // max Iy

  int*					mask = max_idx_.mutable_cpu_data(); //maxscore mask
  const Dtype* 	defw = this->blobs_[0]->cpu_data();


	cbid_ = std::find(nbh_IDs[child_-1].begin(), nbh_IDs[child_-1].end(), parent_) - nbh_IDs[child_-1].begin();
	pbid_ = std::find(nbh_IDs[parent_-1].begin(), nbh_IDs[parent_-1].end(), child_) - nbh_IDs[parent_-1].begin();

/*	LOG(INFO) << "cbid: " << cbid << " | pbid: " << pbid;*/

	ptarget_ = target_IDs[child_-1][cbid_];
	ctarget_ = target_IDs[parent_-1][pbid_];

/*	LOG(INFO) << "ctarget : " << ctarget << " | ptarget: " << ptarget;

	LOG(INFO) << "btm 0 shape: " << bottom[0]->shape_string();
	LOG(INFO) << "btm 1 shape: " << bottom[1]->shape_string();*/

	// message passing
	int Ny = bottom[0]->height();
	int Nx = bottom[0]->width();

  Dtype* 		val_data = new Dtype[height*width];
  int32_t* 	tmp_Ix_data = new int32_t[height*width];
  int32_t* 	tmp_Iy_data = new int32_t[height*width];

  outstream.clear();
  for (int n = 0; n < num; ++n) {
  	const Dtype* 	app_map_offset = app_map + bottom[0]->offset(n, child_-1);

		for (int mc = 0; mc < mix_num_; ++mc) {
			for (int mp = 0; mp < mix_num_; ++mp) {
				// ----  top offset
		  	Dtype* cur_score_ptr = score_ptr + score_.offset(n, mc*mix_num_+mp); 	// dt map
		  	Dtype* cur_Ix_ptr = Ix_ptr + Ix_.offset(n, mc*mix_num_+mp);		// Ix
		  	Dtype* cur_Iy_ptr = Iy_ptr + Iy_.offset(n, mc*mix_num_+mp);		// Iy

		  	// ----  clean val_data, Ix_data, Iy_data
		  	caffe_set(height*width, Dtype(0), val_data);
		  	caffe_set(height*width, 0, tmp_Ix_data);
		  	caffe_set(height*width, 0, tmp_Iy_data);

		  	// ----  current defmap
		  	const Dtype* def_map_offset = def_map + bottom[1]->offset(n, (ptarget_-1)*mix_num_+mc);
		  	//  ---- child.score + child.defmap{cbid}(:, :, mc)
		  	caffe_copy(Ny*Nx, app_map_offset, cur_score_ptr);
				caffe_axpy(Ny*Nx, Dtype(1), def_map_offset, cur_score_ptr);

				// ---- Transpose to column first (for distance transform)
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						val_data[w*height + h] = cur_score_ptr[h*width + w];
					}
				}

			  // ---- deformation weight child->parent
			  const Dtype*	defw_c = defw + this->blobs_[0]->offset((ptarget_-1)*mix_num_+mc);
			  // ---- deformation weight parent->child
			  const Dtype*	defw_p = defw + this->blobs_[0]->offset((ctarget_-1)*mix_num_+mp);
			  // ---- child->parent var
			  Dtype var_c[2] = {1, 1};
			  // ---- parent->child var
			  Dtype var_p[2] = {1, 1};
			  // ---- child->parent mean (in caffe should be (mean_y, mean_x))
			  Dtype mean_c[2] = {meanvals_[(ptarget_-1)*mix_num_+mc].second, meanvals_[(ptarget_-1)*mix_num_+mc].first};
			  // ---- parent->child mean
			  Dtype mean_p[2] = {meanvals_[(ctarget_-1)*mix_num_+mp].second, meanvals_[(ctarget_-1)*mix_num_+mp].first};

			  // Distance Transform
			  distance_transform(val_data, Nx, Ny, defw_c, defw_p,
			                    mean_c, var_c, mean_p, var_p, Nx, Ny,
			                    val_data, tmp_Ix_data, tmp_Iy_data);

				// ---- Transpose to width first (for Caffe blobs)
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						cur_score_ptr[h*width + w] = val_data[w*height + h];
						cur_Ix_ptr[h*width + w] = tmp_Ix_data[w*height + h];
						cur_Iy_ptr[h*width + w] = tmp_Iy_data[w*height + h];
					}
				}
			} // end MP
		} // end MC

		// max out score maps
  	Dtype* 	max_score_ptr_offset = max_score_ptr + max_score_.offset(n);
		int* 		mask_offset = mask +max_idx_.offset(n);
		Dtype* 	max_Ix_ptr_offset = max_Ix_ptr+max_Ix_.offset(n);
		Dtype* 	max_Iy_ptr_offset = max_Iy_ptr+max_Iy_.offset(n);

		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				for (int c = 0; c < score_.channels(); ++c) {
					if ( score_.data_at(n, c, h, w) > max_score_ptr_offset[h*width + w]) {
						max_score_ptr_offset[h*width + w] = score_.data_at(n, c, h, w);
						mask_offset[h*width+w] = c;
					}
				}

				// set Ix Iy
				max_Ix_ptr_offset[h*width+w] = Ix_.data_at(n, mask_offset[h*width+w], h, w);
				max_Iy_ptr_offset[h*width+w] = Iy_.data_at(n, mask_offset[h*width+w], h, w);
		//		LOG(INFO) << "(" << h+1 << ", " << w+1 << ")" <<max_idx_.data_at(n, 0, h, w) ;
			}
		}

		// ---- parts(par).score = parts(par).score + msg;
  	Dtype* 	par_app_map_offset = top[0]->mutable_cpu_data() + top[0]->offset(n, parent_-1);
  	LOG(INFO) << "score shape: " << max_score_.shape_string();
  	LOG(INFO) << "par app shape: " << bottom[0]->shape_string();
  	caffe_axpy(height*width, Dtype(1), max_score_ptr_offset, par_app_map_offset);
  } // end n

  // delete temp maps
  delete[] val_data;
  delete[] tmp_Ix_data;
  delete[] tmp_Iy_data;
/*

  LOG(INFO) << outstream.str();


	LOG(INFO) << "Ny:  " << Ny << " Nx: " << Nx;
	LOG(INFO)  << "CBID: " << cbid << " pbid: " << pbid;

	outstream.clear();
	outstream << "\nnbh_IDs\n";
	for (int p=0; p < nbh_IDs.size(); ++p) {
		for (int n=0; n < nbh_IDs[p].size(); ++n) {
			outstream << nbh_IDs[p][n] << " ";
		}
		outstream << "\n";
	}

	LOG(INFO) << outstream.str();
*/

  LOG(INFO) << "end forward";

}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  LOG(INFO) << "Start BP" ;
  const int* mask = max_idx_.cpu_data();
	const int count = top[0]->count();
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	int num = top[0]->num();
	int height = top[0]->height();
  int width = top[0]->width();

	Dtype* unary_diff = bottom[0]->mutable_cpu_diff();
	Dtype* idpr_diff = bottom[1]->mutable_cpu_diff();
	Dtype* mid_diff = mid_diff_.mutable_cpu_diff();
	Dtype* defw_diff = this->blobs_[0]->mutable_cpu_diff();

	// set diff = 0 (mid_diff has been set as 0 in Reshape ())
  caffe_set(bottom[0]->count(), Dtype(0), unary_diff);
  caffe_set(bottom[1]->count(), Dtype(0), unary_diff);

  for(int n=0; n<num; ++n) {
  	// ---- compute gradient for max operation
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				int max_channel_ = max_idx_.data_at(n, 0, h, w);
				*(mid_diff + mid_diff_.offset(n, max_channel_, h, w)) += top[0]->diff_at(n, parent_-1, h, w);
			}
		}

		for (int mc = 0; mc < mix_num_; ++mc) {
			for (int mp = 0; mp < mix_num_; ++mp) {
				int mid_diff_idx = mc*mix_num_ + mp;
				int idpr_idx = (ptarget_-1)*mix_num_+mc;

				// ----  top offset
				Dtype* cur_score_ptr = score_ptr + score_.offset(n, mc*mix_num_+mp); 	// dt map
				Dtype* cur_Ix_ptr = Ix_ptr + Ix_.offset(n, mc*mix_num_+mp);		// Ix
				Dtype* cur_Iy_ptr = Iy_ptr + Iy_.offset(n, mc*mix_num_+mp);		// Iy

				// ----  clean val_data, Ix_data, Iy_data
				caffe_set(height*width, Dtype(0), val_data);
				caffe_set(height*width, 0, tmp_Ix_data);
				caffe_set(height*width, 0, tmp_Iy_data);

				// ----  current defmap
				const Dtype* def_map_offset = def_map + bottom[1]->offset(n, (ptarget_-1)*mix_num_+mc);
				//  ---- child.score + child.defmap{cbid}(:, :, mc)
				caffe_copy(Ny*Nx, app_map_offset, cur_score_ptr);
				caffe_axpy(Ny*Nx, Dtype(1), def_map_offset, cur_score_ptr);

				// ---- Transpose to column first (for distance transform)
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						val_data[w*height + h] = cur_score_ptr[h*width + w];
					}
				}

				// ---- deformation weight child->parent
				const Dtype*	defw_c = defw + this->blobs_[0]->offset((ptarget_-1)*mix_num_+mc);
				// ---- deformation weight parent->child
				const Dtype*	defw_p = defw + this->blobs_[0]->offset((ctarget_-1)*mix_num_+mp);
				// ---- child->parent var
				Dtype var_c[2] = {1, 1};
				// ---- parent->child var
				Dtype var_p[2] = {1, 1};
				// ---- child->parent mean (in caffe should be (mean_y, mean_x))
				Dtype mean_c[2] = {meanvals_[(ptarget_-1)*mix_num_+mc].second, meanvals_[(ptarget_-1)*mix_num_+mc].first};
				// ---- parent->child mean
				Dtype mean_p[2] = {meanvals_[(ctarget_-1)*mix_num_+mp].second, meanvals_[(ctarget_-1)*mix_num_+mp].first};

				// Distance Transform
				distance_transform(val_data, Nx, Ny, defw_c, defw_p,
													mean_c, var_c, mean_p, var_p, Nx, Ny,
													val_data, tmp_Ix_data, tmp_Iy_data);

				// ---- Transpose to width first (for Caffe blobs)
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						cur_score_ptr[h*width + w] = val_data[w*height + h];
						cur_Ix_ptr[h*width + w] = tmp_Ix_data[w*height + h];
						cur_Iy_ptr[h*width + w] = tmp_Iy_data[w*height + h];
					}
				}
			} // end MP
		} // end MC
  }



  LOG(INFO) << "End BP" ;

}

// start from 0
template <typename Dtype>
int MessagePassingLayer<Dtype>::query_target_id(int selfid, int nbid) {
	int id = std::find(nbh_IDs[selfid].begin(), nbh_IDs[selfid].end(), nbid) - nbh_IDs[selfid].begin();
	return target_IDs[selfid][id];
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::get_IDs() {
  using namespace std;
  using namespace platero;

  int p_no = pa_.size();
  int t_cnt = 0;
  int g_cnt = 0;

  for (int i = 0; i < p_no; ++i) {
    //-----------------------------------------
    // compute nbh_IDs
    vector<int> cur_nbh_IDs;
    for (int j = 0; j < p_no; ++j) {
      if (pa_[i] == (j+1) || pa_[j] == (i+1)) {
        cur_nbh_IDs.push_back(j+1);
      }
    }
    nbh_IDs.push_back(cur_nbh_IDs);

    //-----------------------------------------
    // compute target IDs
    vector<int> cur_target_IDs;
    for (int k = 0; k < cur_nbh_IDs.size(); ++k) {
      cur_target_IDs.push_back(++t_cnt);
    }
    target_IDs.push_back(cur_target_IDs);

    //-----------------------------------------
    // compute global ID
    Vec3DElem cur_global_IDs;
    C3DMAT* mat;

    if (cur_nbh_IDs.size() == 1) {
      mat = new C3DMAT(mix_num_, 1, 1);
      for (int row = 0; row < mix_num_; ++row) {
        mat->set(row, 0, 0, ++g_cnt);
      }
    } else if (cur_nbh_IDs.size() == 2) {
      mat = new C3DMAT(mix_num_, mix_num_, 1);
      for (int col = 0; col < mix_num_; ++col) {
        for (int row = 0; row < mix_num_; ++row) {
          mat->set(row, col, 0, ++g_cnt);
        }
      }

    } else if (cur_nbh_IDs.size() == 3) {
      mat = new C3DMAT(mix_num_, mix_num_, mix_num_);
      for (int dim = 0; dim < mix_num_; ++dim) {
        for (int col = 0; col < mix_num_; ++col) {
          for (int row = 0; row < mix_num_; ++row) {
            mat->set(row, col, dim, ++g_cnt);
          }
        }
      }
    } else {
      LOG(INFO) << "Number of neighborhood must between 1-3.";
    }
    cur_global_IDs.push_back(mat);
    global_IDs.push_back(cur_global_IDs);

    //-----------------------------------------
    // compute global_idpr_ids
    Vec3DElem idpr_part;

    for (int n = 0; n < cur_nbh_IDs.size(); ++n) {
      C3DMAT* idpr_neighbor = new C3dmat<int>(1, 1, 1);
      idpr_neighbor->clone(*mat);
      switch (n) {
        case 0: break;
        case 1:
          idpr_neighbor->permute(2, 1, 3);
          break;
        case 2:
          idpr_neighbor->permute(3, 1, 2);
          break;
        default:
          LOG(ERROR) << "Number of neighbor cannot be greater than 3";
      }
      idpr_part.push_back(idpr_neighbor);
    } // end Neighbor
    idpr_global_ids.push_back(idpr_part);
  }
}


/*-------------------------------------------------------------------------------------
 * 									dt 1-d
 * -----------------------------------------------------------------------------------*/
template <typename Dtype>
void MessagePassingLayer<Dtype>::dt1d(const Dtype *src, Dtype *dst, int *ptr, int step, int len,
        Dtype a_c, Dtype b_c, Dtype a_p, Dtype b_p, Dtype dshift_c, Dtype dshift_p, int dlen) {
  int   *v = new int[len];
  float *z = new float[len+1];
  int k = 0;
  int q = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;

  for (q = 1; q <= len-1; q++) {
    float s = ( (src[q*step] - src[v[k]*step])
    - b_c * -(q - v[k]) + a_c * (square(q) - square(v[k]))
    - b_p * (q - v[k]) + a_p * (square(q) - square(v[k]))
    + 2*a_c * (q-v[k])*(-dshift_c) + 2*a_p * (q-v[k])*(dshift_p) )
    / ( 2*a_c*(q-v[k]) + 2*a_p*(q-v[k]) );
    while (s <= z[k]) {
      k--;
      s = ( (src[q*step] - src[v[k]*step])
      - b_c * -(q - v[k]) + a_c * (square(q) - square(v[k]))
      - b_p * (q - v[k]) + a_p * (square(q) - square(v[k]))
      + 2*a_c * (q-v[k])*(-dshift_c) + 2*a_p * (q-v[k])*(dshift_p) )
      / ( 2*a_c*(q-v[k]) + 2*a_p*(q-v[k]) );
    }
    k++;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = +INF;
  }

  k = 0;
  for (q = 0; q <= dlen-1; q++) {
    while (z[k+1] < q)
      k++;
    dst[q*step] = src[v[k]*step] + a_c * square(q + dshift_c - v[k]) + b_c * -(q + dshift_c - v[k])
    + a_p * square(q - dshift_p - v[k]) + b_p * (q - dshift_p - v[k]);
    ptr[q*step] = v[k];
  }

  delete [] v;
  delete [] z;
}


/**
 * Distance Transform
 * --------------------------------------
 * Compute distance transform of map vals
 * Input params
 * - vals:  input map
 * - sizx, sizy: size of the inputmap vals
 * - defw_c, defw_p: deformation weight child->parent and parent->child
 * - mean_c, mean_p: child->parent mean / parent->child mean
 * - var_c, var_p: child->parent var / parent->child var (could set as [1, 1])
 * - lenx, leny: ????
 *
 * Output Params
 * - M
 * - Ix, Iy
 */
template <typename Dtype>
void MessagePassingLayer<Dtype>::distance_transform(const Dtype* vals, int sizx, int sizy,
                        const Dtype* defw_c, const Dtype* defw_p,
                        Dtype* mean_c, Dtype* var_c,
                        Dtype* mean_p, Dtype* var_p,
                        int32_t lenx, int32_t leny,
                        Dtype *M, int32_t *Ix, int32_t *Iy
                        ) {
  // ---- deformation weight child->parent
  Dtype ax_c = -defw_c[0] ;            // 2nd order
  Dtype bx_c = -defw_c[1] ;            // 1st order
  Dtype ay_c = -defw_c[2];
  Dtype by_c = -defw_c[3];

  // ---- deformation weight parent->child
  Dtype ax_p = -defw_p[0];
  Dtype bx_p = -defw_p[1];
  Dtype ay_p = -defw_p[2];
  Dtype by_p = -defw_p[3];

  Dtype   *tmpM =  new Dtype[leny*sizx];
  int32_t *tmpIy = new int32_t[leny*sizx];

  for (int x = 0; x < sizx; x++)
    dt1d(vals+x*sizy, tmpM+x*leny, tmpIy+x*leny, 1, sizy,
            ay_c/square(var_c[1]), by_c/var_c[1],
            ay_p/square(var_p[1]), by_p/var_p[1],
            mean_c[1], mean_p[1], leny);

  for (int y = 0; y < leny; y++)
    dt1d(tmpM+y, M+y, Ix+y, leny, sizx,
            ax_c/square(var_c[0]), bx_c/var_c[0],
            ax_p/square(var_p[0]), bx_p/var_p[0],
            mean_c[0], mean_p[0], lenx);

  // get argmins and adjust for matlab indexing from 1
  for (int x = 0; x < lenx; x++) {
    for (int y = 0; y < leny; y++) {
      int p = x*leny+y;
      Iy[p] = tmpIy[Ix[p]*leny+y] + 1;
      Ix[p] = Ix[p] + 1;
    }
  }
}




#ifdef CPU_ONLY
STUB_GPU(IdprLayer);
#endif

INSTANTIATE_CLASS(MessagePassingLayer);
REGISTER_LAYER_CLASS(MessagePassing);

}  // namespace caffe
