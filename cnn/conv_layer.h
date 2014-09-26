#ifndef _CONV_LAYER_H_
#define _CONV_LAYER_H_

struct ConvLayer : public Layer {
  ConvLayer(int win_size, int output_size, int layer_indice);
 public:
  virtual void FeedForward(Sample* sample);
  virtual void BackPropagation(Sample* sample);
};
#endif // _CONV_LAYER_H_

