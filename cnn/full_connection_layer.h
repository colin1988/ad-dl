#ifndef _FULL_CONNECTION_LAYER_H_
#define _FULL_CONNECTION_LAYER_H_

struct FullConnectionLayer : public Layer {
  FullConnectionLayer(int input_size, int output_size, int layer_indice);
 public:
  virtual void FeedForward(Sample* sample);
  virtual void BackPropagation(Sample* sample);
  virtual void Update(Sample* sample);

 private:
  mat hidden_weight_;
  rowvec hidden_bias_;
  mat output_weight_;
  rowvec output_bias_;
  int input_size_;
  int hidder_size_;
  int output_size_;
};
#endif // _FULL_CONNECTION_LAYER_H_

