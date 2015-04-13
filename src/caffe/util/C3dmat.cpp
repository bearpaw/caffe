#include "caffe/util/C3dmat.hpp"
#include <sstream>
#include <iomanip>  

namespace platero {


template<typename Dtype>
Dtype*** C3dmat<Dtype>::create_private_array(int r, int c, int d, Dtype val)
{
  Dtype***arr =  new Dtype**[r];

  for (int i = 0; i < r; ++i) {
    arr[i] = new Dtype*[c];
    for (int j = 0; j < c; ++j) {
      arr[i][j] = new Dtype[d];
      // memset(arr[i][j], Dtype(val), sizeof(Dtype)*d);
      // memset send the val as int
      for (int k = 0; k < d; ++k) {
        arr[i][j][k] = val;
      }
    }
  }
  return arr;
}

template<typename Dtype>
void C3dmat<Dtype>::destroy_private_array(int r, int c, Dtype*** arr)
{
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      delete[] arr[i][j];
    }
    delete[] arr[i];
  }
  delete[] arr;
  arr = NULL;
}

template<typename Dtype>
C3dmat<Dtype>::C3dmat()
{
    mat = NULL;
    cols = 0;
    rows = 0;
    dims = 0;
}

template<typename Dtype>
C3dmat<Dtype>::C3dmat(int r, int c, int d, Dtype val)
{
  C3dmat();
  create(r, c, d, val);
}


// deep copy
template<typename Dtype>
C3dmat<Dtype>::C3dmat(const C3dmat& obj)
{
  C3dmat();
  clone(obj);
}

template<typename Dtype>
void C3dmat<Dtype>::clone(const C3dmat& obj)
{
  destroy();
  this->rows = obj.rows;
  this->cols = obj.cols;
  this->dims = obj.dims;
  // create new mat
  this->mat = create_private_array(rows, cols, dims, Dtype(0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      for (int k = 0; k < dims; ++k) {
        this->mat[i][j][k] = obj.mat[i][j][k];
      }
    }
  }

}

template<typename Dtype>
C3dmat<Dtype>::~C3dmat()
{
  //dtor
  this->destroy();
}

template<typename Dtype>
void C3dmat<Dtype>::destroy() {
  destroy_private_array(rows, cols, mat);

  rows = 0;
  cols = 0;
  dims = 0;
}


template<typename Dtype>
Dtype*** C3dmat<Dtype>::create(int r, int c, int d, Dtype val) {
  rows = r;
  cols = c;
  dims = d;

  mat = create_private_array(rows, cols, dims, val);

  return mat;
}


template<typename Dtype>
std::string C3dmat<Dtype>::print() {
  std::ostringstream oss;
  oss << "\nrows: " << rows << " | cols: " << cols << " | dims: " << dims << std::endl;
  for (int k = 0; k < dims; ++k) {
    oss << "Channel " << k << " of " << dims << "\n ==================" << std::endl;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        oss << mat[i][j][k] << " ";
      }
      oss << std::endl;
    }
  }
  return oss.str();
}

template<typename Dtype>
void C3dmat<Dtype>::set(int row, int col, int dim, Dtype val)
{
  mat[row][col][dim] = val;
}

template<typename Dtype>
const Dtype& C3dmat<Dtype>::at(int row, int col, int dim)
{
  return mat[row][col][dim];
}

template<typename Dtype>
const int& C3dmat<Dtype>::get_rows()
{
  return rows;
}

template<typename Dtype>
const int& C3dmat<Dtype>::get_cols()
{
  return cols;
}


template<typename Dtype>
const int& C3dmat<Dtype>::get_dims()
{
  return dims;
}

template<typename Dtype>
int C3dmat<Dtype>::counts()
{
  return cols*rows*dims;
}



// vectorize
template<typename Dtype>
const vector<Dtype>& C3dmat<Dtype>::vectorize()
{
  vec.clear();
  if (cols == 0 || rows == 0 || dims == 0) return vec;

  for (int k = 0; k < dims; ++k) {
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        vec.push_back(mat[i][j][k]);
      }
    }
  }
  return vec;
}


template<typename Dtype>
void C3dmat<Dtype>::permute(int row, int col, int dim)
{
  // compute new cols rows and dims
  int nrows = rows;
  int ncols = cols;
  int ndims = dims;
  Dtype*** nmat = NULL;

  if (row == 1 && col == 2 && dim == 3) {
    return;
  }
  else if (row == 1 && col == 3 && dim == 2) {
    ncols = dims;
    ndims = cols;
    // create new mat
    nmat = create_private_array(nrows, ncols, ndims, Dtype(0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[i][k][j];
        }
      }
    }
    destroy_private_array(rows, cols, mat);
    mat = nmat;

  }
  else if (row == 2 && col == 1 && dim == 3) {
    ncols = rows;
    nrows = cols;
    // create new mat
    nmat = create_private_array(nrows, ncols, ndims, Dtype(0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[j][i][k];
        }
      }
    }
    destroy_private_array(rows, cols, mat);
    mat = nmat;
  }
  else if (row == 2 && col == 3 && dim == 1) {
    nrows = cols;
    ncols = dims;
    ndims = rows;
    // create new mat
    nmat = create_private_array(nrows, ncols, ndims, Dtype(0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[k][i][j];
        }
      }
    }
    destroy_private_array(rows, cols, mat);
    mat = nmat;
  }
  else if (row == 3 && col == 1 && dim == 2) {
    nrows = dims;
    ncols = rows;
    ndims = cols;
    // create new mat
    nmat = create_private_array(nrows, ncols, ndims, Dtype(0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[j][k][i];
        }
      }
    }
    destroy_private_array(rows, cols, mat);
    mat = nmat;
  }
  else if (row == 3 && col == 2 && dim == 1) {
    nrows = dims;
    ndims = rows;
    // create new mat
    nmat = create_private_array(nrows, ncols, ndims, Dtype(0));
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[k][j][i];
        }
      }
    }
    destroy_private_array(rows, cols, mat);
    mat = nmat;
  } else {
    std::cout << "Wrong number" << std::endl;
  }


  rows = nrows;
  cols = ncols;
  dims = ndims;

}

template class C3dmat<float>;
template class C3dmat<double>;
template class C3dmat<int>;

} // end namespace
