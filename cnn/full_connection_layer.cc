#include "full_connection_layer.h"

#include "sample.h"

FullConnectionLayer::FullConnectionLayer(
    int input_size, int output_size, int indice)
    : Layer(input_size, output_size, indice) {
  
}

void FullConnectionLayer::FeedForward(Sample* sample) {
  vector<float>& data = sample->layer_data[layer_indice_ - 1];
  vector<float>& cur_layer_data = sample->layer_data[layer_indice];
  util::MatrixMultiply(weight_, data, &cur_layer_data);
  util::VectorAdd(&cur_layer_data, bias);
  util::TanhVector(&cur_layer);
  sample->layer_data.push_back(cur_layer_data);
}

// compute gradient for next layer
void FullConnectionLayer::BackPropagation(Sample* sample) {
  vector<float>& cur_gradient = sample->gradient[layer_indice_];
  util::TanhGradient(&cur_gradient);
  vector<float>& next_gradient = sample->gradient[layer_indice_ - 1];
  float **weight = last_layer_weight;

  util::MatrixMultiply(last_gradient, )
}
