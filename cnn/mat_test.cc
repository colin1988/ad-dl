#include <iostream>

#include "armadillo"
using namespace arma;
using namespace std;

int main(void) {
  rowvec a(5);
  a.fill(5.0);

  colvec b = trans(a);


  cout << a;
  cout << b;
  
  b.set_size(10);
  b.fill(1.0);

  cout << b;
  cout << "subvec:" << b.subvec(1, 5);

  mat matrix_ins;
  matrix_ins.set_size(3, 4);
  matrix_ins << 1 << 2 << 4 << 2 << endr  <<
                2 << 4 << 1 << 3 << endr  <<
                4 << 3 << 2 << 1 << endr  <<
                5 << 1 << 3 << 1 << endr;
  
  cout << matrix_ins << endl;

  for (int i = 0; i < matrix_ins.n_cols; i++) {
    double max_val;
    uword max_index;

    max_val = matrix_ins.col(i).max(max_index);
    cout << "col " << i << " max_value:" << max_val <<
            " max index:" << max_index << endl;
  }
  
  return 0;
}
