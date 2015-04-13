#ifndef C3DMAT_H
#define C3DMAT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>

namespace platero {

using namespace std;


  template<typename Dtype>
  class C3dmat
  {
    public:

      // initiate arguments
      C3dmat();
      // deep copy
      C3dmat(const C3dmat& obj);
      // initiate arguments
      C3dmat(int r, int c, int d, Dtype val = Dtype(0));
      // initiate arguments
      void clone(const C3dmat& obj);
      // create matrix
      Dtype*** create(int r, int c, int d, Dtype val);
      // set matrix
      void set(int row, int col, int dim, Dtype val);
      // permute matrix
      void permute(int row, int col, int dim);
      // get matrix
      const Dtype& at(int row, int col, int dim);
      // destroy matrix
      void destroy();
      // print matrix
      std::string print();
      // get cols
      const int& get_cols();
      // get rows
      const int& get_rows();
      // get dims
      const int& get_dims();
      // get dims
      int counts();
      // vectorize
      const vector<Dtype>& vectorize();
      // destrop matrix
      virtual ~C3dmat();
    private:
      Dtype*** create_private_array(int r, int c, int d, Dtype val);
      void destroy_private_array(int r, int c, Dtype*** arr);


      Dtype*** mat;
      vector<Dtype> vec;
      int cols;
      int rows;
      int dims;
  };
} // name space

#endif // C3DMAT_H
