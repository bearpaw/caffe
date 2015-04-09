#include "caffe/util/C3dmat.hpp"
namespace platero {

template<typename Dtype>
Dtype*** C3dmat<Dtype>::create_private_array(int x, int y, int z, Dtype val)
{
  Dtype***arr =  new Dtype**[x];

  for (int i = 0; i < x; ++i) {
    arr[i] = new Dtype*[y];
    for (int j = 0; j < y; ++j) {
       arr[i][j] = new Dtype[z];
       memset(arr[i][j], Dtype(val), sizeof(Dtype)*z);
    }
  }
  return arr;
}

template<typename Dtype>
void C3dmat<Dtype>::destroy_private_array(int x, int y, Dtype*** arr)
{
    for (int i = 0; i < x; ++i) {
    for (int j = 0; j < y; ++j) {
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
C3dmat<Dtype>::C3dmat(int x, int y, int z, Dtype val)
{
  create(x, y, z, val);
}

template<typename Dtype>
C3dmat<Dtype>::~C3dmat()
{
  //dtor
  this->destroy();
}


template<typename Dtype>
Dtype*** C3dmat<Dtype>::create(int x, int y, int z, Dtype val) {
  cols = x;
  rows = y;
  dims = z;

  mat = create_private_array(cols, rows, dims, val);

  return mat;
}

template<typename Dtype>
void C3dmat<Dtype>::destroy() {
  destroy_private_array(cols, rows, mat);

  cols = 0;
  rows = 0;
  dims = 0;
}

template<typename Dtype>
void C3dmat<Dtype>::print() {
  cout << "cols: " << cols << " | rows: " << rows << " | dims: " << dims << endl;
  for (int k = 0; k < dims; ++k) {
    std::cout << "Channel " << k << " of " << dims << "\n ==================" << std::endl;
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        std::cout << mat[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
  }
}

template<typename Dtype>
void C3dmat<Dtype>::set(int col, int row, int dim, Dtype val)
{
  mat[col][row][dim] = val;
}

template<typename Dtype>
const Dtype& C3dmat<Dtype>::at(int col, int row, int dim)
{
  return mat[col][row][dim];
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
const int& C3dmat<Dtype>:: get_dims()
{
  return dims;
}


template<typename Dtype>
void C3dmat<Dtype>::permute(int col, int row, int dim)
{
  // compute new cols rows and dims
  int ncols = cols;
  int nrows = rows;
  int ndims = dims;
  Dtype*** nmat = NULL;

  if (col == 1 && row == 2 && dim == 3) {
    return;
  }
  else if (col == 1 && row == 3 && dim == 2) {
    nrows = dims;
    ndims = rows;
    // create new mat
    nmat = create_private_array(ncols, nrows, ndims, Dtype(0));
    for (int i = 0; i < ncols; ++i) {
      for (int j = 0; j < nrows; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[i][k][j];
        }
      }
    }
    destroy_private_array(cols, rows, mat);
    mat = nmat;

  }
  else if (col == 2 && row == 1 && dim == 3) {
    ncols = rows;
    nrows = cols;
    // create new mat
    nmat = create_private_array(ncols, nrows, ndims, Dtype(0));
    for (int i = 0; i < ncols; ++i) {
      for (int j = 0; j < nrows; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[j][i][k];
        }
      }
    }
    destroy_private_array(cols, rows, mat);
    mat = nmat;
  }
  else if (col == 2 && row == 3 && dim == 1) {
    ncols = rows;
    nrows = dims;
    ndims = cols;
    // create new mat
    nmat = create_private_array(ncols, nrows, ndims, Dtype(0));
    for (int i = 0; i < ncols; ++i) {
      for (int j = 0; j < nrows; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[k][i][j];
        }
      }
    }
    destroy_private_array(cols, rows, mat);
    mat = nmat;
  }
  else if (col == 3 && row == 1 && dim == 2) {
    ncols = dims;
    nrows = cols;
    ndims = rows;
    // create new mat
    nmat = create_private_array(ncols, nrows, ndims, Dtype(0));
    for (int i = 0; i < ncols; ++i) {
      for (int j = 0; j < nrows; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[j][k][i];
        }
      }
    }
    destroy_private_array(cols, rows, mat);
    mat = nmat;
  }
  else if (col == 3 && row == 2 && dim == 1) {
    ncols = dims;
    ndims = cols;
    // create new mat
    nmat = create_private_array(ncols, nrows, ndims, Dtype(0));
    for (int i = 0; i < ncols; ++i) {
      for (int j = 0; j < nrows; ++j) {
        for (int k = 0; k < ndims; ++k) {
          nmat[i][j][k] = mat[k][j][i];
        }
      }
    }
    destroy_private_array(cols, rows, mat);
    mat = nmat;
  } else {
    std::cout << "Wrong number" << std::endl;
  }


  cols = ncols;
  rows = nrows;
  dims = ndims;

}

template class C3dmat<float>;
template class C3dmat<double>;
template class C3dmat<int>;

} // end namespace
