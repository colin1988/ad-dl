#ifndef _CONV_LAYER_H_
#define _CONV_LAYER_H_


#include "armadillo"
using namespace armadillo;

struct ConvLayer : public Layer {
  ConvLayer(int win_size, int output_size, int layer_indice);
 public:
  virtual void FeedForward(Sample* sample);
  virtual void BackPropagation(Sample* sample);

 private:
  mat user_query_weight_;
  int query_win_size_;
  int query_conv_size_;
  struct ad_weight {
    mat ader_weight;
    mat ad_weight;
    mat weight;
    int win_size;
  } keyword_weight_, title_weight_, desc_weight_;

  mat user_vec_;
  mat ader_vec_;
  mat ad_vec_;
  mat word_vec_;
};
#endif // _CONV_LAYER_H_

