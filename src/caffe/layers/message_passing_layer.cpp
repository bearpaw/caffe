#include <vector>

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
  // Read mean value of each neighbor
	const string& source = this->layer_param_.message_passing_param().mean_file();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	Dtype mean_x, mean_y;
	while (infile >> mean_x >> mean_y) {
		meanvals_.push_back(std::make_pair(mean_x, mean_y));
		//meanvals_.push_back(std::make_pair(-0.1, 0.5));
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

template <typename Dtype>
void MessagePassingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// top shape
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int channels = mix_num_*mix_num_;
  top[0]->Reshape(num, channels, height, width);
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  // dtvals shape
  dtvals_.Reshape(num, channels, width, height);
  Dtype* dtvals_data = dtvals_.mutable_cpu_data();
  caffe_set(dtvals_.count(), Dtype(0), dtvals_data);
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

/*	for (int i = 0; i < meanvals_.size(); ++i)
	{
		LOG(INFO) << i << ": " << meanvals_[i].first << ", " << meanvals_[i].second ;
	}*/


	LOG(INFO) << "Weight shape :" << this->blobs_[0]->shape_string();
	std::stringstream outstream;
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  const Dtype*  score_map = bottom[0]->cpu_data();
  const Dtype*  def_map = bottom[1]->cpu_data();
  Dtype*        top_data = top[0]->mutable_cpu_data();
  Dtype*				val_data = dtvals_.mutable_cpu_data();
  const Dtype* 	weight = this->blobs_[0]->cpu_data();

	int parent_ = pa_[child_-1];

	int cbid = std::find(nbh_IDs[child_-1].begin(), nbh_IDs[child_-1].end(), parent_) - nbh_IDs[child_-1].begin();
	int pbid = std::find(nbh_IDs[parent_-1].begin(), nbh_IDs[parent_-1].end(), child_) - nbh_IDs[parent_-1].begin();

	int ptarget = target_IDs[child_-1][cbid];
	int ctarget = target_IDs[parent_-1][pbid];

	LOG(INFO) << "1: " << ctarget;
	LOG(INFO) << "2: " << ptarget;

	LOG(INFO) << "btm 0 shape: " << bottom[0]->shape_string();
	LOG(INFO) << "btm 1 shape: " << bottom[1]->shape_string();

	// message passing
	int Ny = bottom[0]->height();
	int Nx = bottom[0]->width();

  int32_t *Iy = new int32_t[Ny * Nx];
  int32_t *Ix = new int32_t[Ny * Nx];

  outstream.clear();
  for (int n = 0; n < num; ++n) {
  	const Dtype* score_map_offset = score_map + bottom[0]->offset(n, child_-1);
		for (int mc = 0; mc < mix_num_; ++mc) {
			for (int mp = 0; mp < mix_num_; ++mp) {
				// top offset
		  	Dtype* top_data_offset = top_data + top[0]->offset(n, mc*mix_num_+mp);
		  	Dtype* val_data_offset = val_data + dtvals_.offset(n, mc*mix_num_+mp);
		  	// current defmap
		  	const Dtype* def_map_offset = def_map + bottom[1]->offset(n, (ptarget-1)*mix_num_+mc);
		  	// child.score + child.defmap{cbid}(:, :, mc)
		  	caffe_copy(Ny*Nx, score_map_offset, top_data_offset);
				caffe_axpy(Ny*Nx, Dtype(1), def_map_offset, top_data_offset);

				//----------------------------------------------
				// Transpose
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						val_data_offset[w*height + h] = top_data_offset[h*width + w];
					}
				}


			  // ---- deformation weight child->parent
			  const Dtype*	defw_c = weight + this->blobs_[0]->offset((ptarget-1)*mix_num_+mc);
			  // ---- deformation weight parent->child
			  const Dtype*	defw_p = weight + this->blobs_[0]->offset((ctarget-1)*mix_num_+mp);
			  // ---- child->parent var
			  Dtype var_c[2] = {1, 1};
			  // ---- parent->child var
			  Dtype var_p[2] = {1, 1};
			  // ---- child->parent mean
			  Dtype mean_c[2] = {meanvals_[(ptarget-1)*mix_num_+mc].first, meanvals_[(ptarget-1)*mix_num_+mc].second};
			  // ---- parent->child mean
			  Dtype mean_p[2] = {meanvals_[(ctarget-1)*mix_num_+mp].first, meanvals_[(ctarget-1)*mix_num_+mp].second};

			  // Distance Transform
			  distance_transform(val_data_offset, Nx, Ny, defw_c, defw_p,
			                    mean_c, var_c, mean_p, var_p, Nx, Ny,
			                    val_data_offset, Ix, Iy);

				//----------------------------------------------
				// Transpose
				for (int h = 0; h < height; ++h) {
					for (int w = 0; w < width; ++w) {
						top_data_offset[h*width + w] = val_data_offset[w*height + h];
					}
				}

			  outstream << mc*mix_num_ + mp + 1 << ": "
			  		<< mean_c[0] << ", " << mean_c[1] << ", " <<mean_p[0] << ", " << mean_p[1] << "\n";
			}
		}
  }

  LOG(INFO) << outstream.str();


	LOG(INFO) << "Ny:  " << Ny << " Nx: " << Nx;
	LOG(INFO)  << "CBID: " << cbid << " pbid: " << pbid;

	outstream << "\nnbh_IDs\n";
	for (int p=0; p < nbh_IDs.size(); ++p) {
		for (int n=0; n < nbh_IDs[p].size(); ++n) {
			outstream << nbh_IDs[p][n] << " ";
		}
		outstream << "\n";
	}

	LOG(INFO) << outstream.str();

}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  LOG(INFO) << "Start BP" ;
  const Dtype* top_diff = top[0]->cpu_diff();

  // bottom statistics
  int bottom_channels = bottom[0]->channels(); 

  // top statistics
  int top_num = top[0]->num();
  int top_channels = top[0]->channels(); 
  int top_height = top[0]->height(); 
  int top_width = top[0]->width(); 
  int top_count = top[0]->count();

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

      caffe_set(top_count, Dtype(0), bottom_diff);
      
    }
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
