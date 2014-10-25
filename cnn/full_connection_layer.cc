#include "full_connection_layer.h"

#include "sample.h"

FullConnectionLayer::FullConnectionLayer(
    int input_size, int output_size, int indice)
    : Layer(input_size, output_size, indice) {
  hidden_weight_.set_size(input_size, hidden_size_);
  output_weight_.set_size(hidden_size_, output_size_);

void FullConnectionLayer::FeedForward(Sample* sample) {
  sample->hidden_output = sample->full_input *
                          hidden_weight_;
  sample->hidden_output += hidden_bias_;
  sample->hidden_output = tanh(sample->hidden_output);
  
  sample->final_output = sample->hidden_output *
                         output_weight_;
  sample->final_output += output_bias_;
  sample->final_output = sigmoid(sample->final_output);

}

// compute gradient for next layer
void FullConnectionLayer::BackPropagation(Sample* sample) {
  vector<float>& cur_gradient = sample->gradient[layer_indice_];
  util::TanhGradient(&cur_gradient);
  vector<float>& next_gradient = sample->gradient[layer_indice_ - 1];
  float **weight = last_layer_weight;

  util::MatrixMultiply(last_gradient, )
}
