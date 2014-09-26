#ifndef _FULL_CONNECTION_LAYER_H_
#define _FULL_CONNECTION_LAYER_H_

struct FullConnectionLayer : public Layer {
  FullConnectionLayer(int input_size, int output_size, int layer_indice);
 public:
  virtual void FeedForward(Sample* sample);
  virtual void BackPropagation(Sample* sample);
};
#endif // _FULL_CONNECTION_LAYER_H_

