#ifndef _SAMPLE_H
#define _SAMPLE_H

#include <vector>
#include <string>

struct TrainRecord {
  int click_cnt;
  int imp_cnt;
  string url;
  int ad_id;
  int ader_id;
  int depth;
  int pos;
  int query_id;
  int keyword_id;
  int title_id;
  int desc_id;
  int user_id; 
};

struct Sentence {
  vector<int> sen;
  rowvec sen_vector;
  vector<float> max_indice;
  rowvec conv;
};

struct Sample {
  TrainRecord train_input;
  Sentence query;
  Sentence keyword;
  Sentence title;
  Sentence desc;
  rowvec user_vector;
  rowvec ad_vector;
  rowvec ader_vector;
  vector<vector<float>> layer_data;
};
#endif //  _SAMPLE_H
