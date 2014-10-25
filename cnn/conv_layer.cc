#include "conv_layer.h"

ConvLayer::ConvLayer(int intput_size,
                     int output_size,
                     int layer_indice)
    : Layer(input_size, output_size, layer_indice) {
}

void ConvLayer::InitSentence(vector<vector<int>>& token_ids,
                             float* token_vector,
                             size_t vector_size,
                             int id, 
                             int win_size,
                             Sentence* sen) {
  vector<int>& tokens = token_ids[id];
  sen->sen.resize(tokens.size());
  std::copy(tokens.begin(), tokens.end(), sen->sen.begin());
  sen->sen_vector.set_size(tokens.size() * vector_size);
  //to do :copy
  
  sen->max_indice.assign()
}

void ConvLayer::ConvQuery(Sample* sample) {
  sample->query.conv.resize(query_conv_size_, 0.0);
  vector<float> tmp;
  int sen_size = sample->query.sen.size();
  if (sen_size > query_win_size) {
    mat conv;
    int len = sen - query_win_size + 1;
    conv.set_size(query_conv_size, len);

    for (size_t i = 0; i < len; i++) {
      conv.col(len) = sample->query.sen_vector.subvec(i * vector_size,
                                                      (i + query_win_size) * vector_size - 1) * 
                      query_conv_weight;
      // vector_mutlply_matrix(sample->query.sen_vector, i, i + query_win_size, query_conv_weight, &conv[i]);
    }
    
    for (int i = 0; i < query_conv_size_; i++) {
      float max_val;
      uword max_index;

      max_val = conv.row(i).max(max_index);

      max_indice[i] = max_index;
      sample->query.conv(i) = max_val;
    }

    //MaxPool(conv, sen);    
  }

  sample->query.conv += trans(sample->user_vector * user_query_weight_);
  sample->query.conv += query_conv_bias_;
  sample->query.conv = tanh(sample->query.conv);

  // vector_multiply_matrix(sample->user_vector, user_query_weight_, &sample->query.conv);
  // vector_tanh(&sample->sample.conv);
  
}

void ConvLayer::ConvAdSen(Sample* sample, int win_size, int conv_size, int vector_size,
                          vector<float>& ader_weight, vector<float>&ad_weight, vector<float>& weight, vector<float>& bias
                          Sentence* sen) {
  sen->conv.resize(conv_size);
  int sen_size = sen->sen.size();
  if (sen_size > win_size) {
    // vector<vector<float>> conv;
    mat conv;
    int len = sen_size - win_size + 1;
    // conv.resize(len);
    conv.set_size(conv_size, len);
    // for (size_t i = 0; i < conv.size(); i++) {
    //    vector_multiply_matrix(sen.sen_vector,i * vector_size, (i + query_win_size) * vector_size, weight, &conv[i]); 
    // }
    
    for (size_t i = 0 ; i < len; i++) {
      conv.col(i) = sen.sen_vector(i * vector_size,
                                   (i + query_win_size) * vector_size - 1)
                    * weight;
    }
    // MaxPool(conv, sen);

    for (int i = 0; i < conv_size; i++) {
      float max_val;
      uword max_index;

      max_val = conv.row(i).max(max_index);

      max_indice[i] = max_index;
      sen->conv(i) = max_val;
    }

    
  } else {
      // vector_mutltiply_matrix(sen.sen_vector, 0, sen_size * vector_size, weight, &sen.conv);
      sen.conv.fill(0.0);
      sen.conv = trans(sen.sen_vector * weight.submat(0, 0,
                                                sen_size * vector_size - 1, weight.n_cols - 1));
      
  }

  sen.conv += trans(sample->ader_vector * ader_weight);
  sen.conv += trans(sample->ad_vector * ad_weight);
  sen.conv += bias;
  sen.conv = tanh(sen.conv);

  // vector_multiply_matrix(sample->ader_vector, ader_weight, &sen.conv);
  // vector_multiply_matrix(sample->ad_vector, ad_weight, &sen.conv);
  // vector_add_vector(&sen.conv, bias);
  // vector_tanh(&sen.conv);
}

void ConvLayer::ConvKeyword(Sample* sample) {
  sample->keyword.conv.resize(keyword_conv_size_, 0.0);
  vector<float> tmp;

  int sen_size = sample->keyword.sen.size();
  if (sen_size > query_win_)
}

void ConvLayer::ConvKeyword(Sample* sample) {

}

void ConvLayer::ConvTitle(Sample* sample) {

}

void ConvLayer::ConvDesc(Sample* sample) {

}

void ConvLayer::BpQuery(Sample* sample) {
  rowvec deri = sample->query.output_delta % (1.0 - (sample->query.conv % sample->query.conv));
  sample->query.input_delta = deri * trans(query_conv_weight_);
  sample->user_query_delta = deri * trans(user_query_weight_);
}  

void ConvLayer::BpAd(Sample* sample, int win_size, int conv_size, int vector_size,
                     vector<float>& ader_weight, vector<float>&ad_weight, vector<float>& weight, vector<float>& bias
                     Sentence* sen) {
  rowvec deri = sen->output_delta % (1.0 - sen->conv % sen->conv);
  sen.input_delta = deri * trans(weight);
  
  // ader_delta in sample or in sen ??
  sample->ader_delta += deri * trans(ader_weight);
  sample->ad_delta += deri * trans(ad_weight);
}

void ConvLayer::FeedForward(Sample* sample) {
  vector<int> sentence;
  GetQueryToken(sample->train_input.query_id, &sentence);
  GetSentenceVector()
}

void ConvLayer::UpdateQuery(Sample* sample) {
  sentence& sent = sample->query;
  rowvec deri = sen->output_delta % (1.0 - sen->conv % sen->conv);
  
  for (int i=0; i < max_indice.n_rows(); i++) {
    int index = max_indice(i);
    for (int j=0; j<query_win_size_; i++) {
      int word_index = index+j;
      if (sent.sen[i] ==0) {
        continue;
      } 
      
      weight.col(i).(j*vec_size, (j+1)*vec_size-1) += deri(i).word_vec_.col(word_index); 
      //input_delta 已经是对query词向量节点的输出的倒数，而输出等于输入
      word_vec_.col(word_index) += trans(sent.input_delta(j*vec_size, (j+1)*vec_size-1);
    }
  }
}

void ConvLayer::Update(Sample* sample) {
  UpdateQuery();
  UpdateAd(keyword);
  UpdateAd(desc);
  UpdateAd(title);
}
