#ifndef C3DMAT_H
#define C3DMAT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

using namespace std;

namespace platero {
  template<typename Dtype>
  class C3dmat
  {
    public:

      // initiate arguments
      C3dmat();
      // initiate arguments
      C3dmat(int x, int y, int z, Dtype val = Dtype(0));
      // create matrix
      Dtype*** create(int x, int y, int z, Dtype val);
      // set matrix
      void set(int col, int row, int dim, Dtype val);
      // permute matrix
      void permute(int col, int row, int dim);
      // get matrix
      const Dtype& at(int col, int row, int dim);
      // destroy matrix
      void destroy();
      // print matrix
      void print();
      // get cols
      const int& get_cols();
      // get rows
      const int& get_rows();
      // get dims
      const int& get_dims();
      // destrop matrix
      virtual ~C3dmat();
    private:
      Dtype*** create_private_array(int x, int y, int z, Dtype val);
      void destroy_private_array(int x, int y, Dtype*** arr);


      Dtype*** mat;
      int cols;
      int rows;
      int dims;
  };
} // name space

#endif // C3DMAT_H
