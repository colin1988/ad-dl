#ifndef _LAYER_H_
#define _LAYER_H_

struct Sample;

struct Layer {
  int input_size_;
  int output_size_;
  float** weight_;
  float* bias_;
 public:
  virtual void FeedForward(Sample* sample) = 0;
  virtual void BackPropagation(Sample* sample) = 0;
};

#endif // _LAYER_H_
