//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

using std::vector;
using std::string;
using std::ifstream;
using std::unordered_map;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

struct vocab_word_cnt {
    long long cnt;
};

char train_file[MAX_STRING], output_file[MAX_STRING], word_vec_path[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100,
          word_vec_size = 50, ad_id_vec_size = 30, ader_id_vec_size = 20,
          user_id_vec_size = 30;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
long long ad_table_size = 0, ader_table_size = 0, user_table_size = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *word_vec, *syn1, *weight, *expTable;
real *ad_id_vec, *ader_id_vec, *user_id_vec;
clock_t start;

struct vocab_word_cnt* word_table;
vector<vector<int> > title_tokens, desc_tokens, keyword_tokens, query_tokens;
char keyword_token_file[MAX_STRING], title_token_file[MAX_STRING],
     desc_token_file[MAX_STRING], query_token_file[MAX_STRING];
char ader_id_map_path[MAX_STRING], ad_id_map_path[MAX_STRING], user_id_map_path[MAX_STRING];
unordered_map<int, int> ader_id_map, ad_id_map, user_id_map;
long max_token_id;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
    int a, i;
    long long train_words_pow = 0;
    real d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(word_table[a].cnt, power);
    i = 0;
    d1 = pow(word_table[i].cnt, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(word_table[i].cnt, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}


// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if (vocab[a].cn < min_count) {
            vocab_size--;
            free(vocab[vocab_size].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void ReadVocab2() {
    printf("read vocab2 %s\n", read_vocab_file);
    int max_cnt, max_id, word_count;
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    fscanf(fin, "%d%d%d", &max_cnt, &max_id, &word_count);
    max_token_id = max_id;
    vocab_size = max_id + 2;
  

    word_table = new vocab_word_cnt[vocab_size];
    word_table[0].cnt = 10000;
    for (int i = 0; i < word_count; i++) {
      int cnt, id;
      fscanf(fin, "%d%d", &cnt, &id);
      word_table[id+1].cnt = cnt; 
      if (cnt >= min_count) {
        train_words += cnt;
      }
    }
    
    fclose(fin);
    printf("finish reading vocab2\n");
}

void ReadIdMap(const char* file_path, unordered_map<int, int>* id_map) {
  printf("start read id map:%s\n", file_path);
  FILE* map_file = fopen(file_path, "rb");
  int id, idx;
  while (fscanf(map_file, "%d%d", &id, &idx) != EOF) {
    (*id_map)[id] = idx;
  }
  printf("finish reading id map\n");
}

void SplitStringToVector(const string& full, const char* delimiters,
                                                  bool omit_empty_strings,
                                                  vector<string>* out) {
  // CHECK(out != NULL);
  out->clear();

  size_t start = 0, end = full.size();
  size_t found = 0;
  while (found != string::npos) {
    found = full.find_first_of(delimiters, start);

    // start != end condition is for when the delimiter is at the end.
    if (!omit_empty_strings || (found != start && start != end))
      out->push_back(full.substr(start, found - start));
    start = found + 1;
  }
}


void ReadTokenList(const char* token_file, vector<vector<int>>* token_vector) {
  printf("read token file %s\n", token_file);
  ifstream fin(token_file);
  if (!fin) {
    printf("Vocabulary file not found\n");
    exit(1);
  }

  string token_list;
  int id;

  while (fin) {
    fin >> id >> token_list;
    if (id >= static_cast<int>(token_vector->size())) {
      token_vector->resize(id+1);
    }
    
    vector<string> tokens;
    SplitStringToVector(token_list, "|", true, &tokens);
    
    for (auto itr = tokens.begin(); itr != tokens.end(); ++itr) {
      int token = stoi(*itr, NULL);
      (*token_vector)[id].push_back(token);
    
    }   
  }

}

void InitNet() {
    long long a, b;
    // a = posix_memalign((void **)&word_vec, 128, (long long)vocab_size * word_vec_size * sizeof(real));
    // if (word_vec == NULL) {printf("Memory allocation failed for word_vec\n"); exit(1);}
        

    // a = posix_memalign((void **)&ad_id_vec, 128, (long long)ad_table_size * ad_id_vec_size * sizeof(real));
    // if (ad_id_vec == NULL) {printf("Memory allocation failed for ad_id_vec\n"); exit(1);}

    a = posix_memalign((void **)&user_id_vec, 128, (long long)user_table_size * user_id_vec_size * sizeof(real));
    if (user_id_vec == NULL) {printf("Memory allocation failed for ader_id_vec\n"); exit(1);}

    if (hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) {
        a = posix_memalign((void **)&weight, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (weight == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            weight[a * layer1_size + b] = 0;
    }
    // for (b = 0; b < word_vec_size; b++) for (a = 0; a < vocab_size; a++)
    //    word_vec[a * word_vec_size + b] = (rand() / (real)RAND_MAX - 0.5) / word_vec_size;
    // for (b = 0; b < ad_id_vec_size; b++) for (a = 0; a < ad_table_size; a++)
    //     ad_id_vec[a * ad_id_vec_size + b] = (rand() / (real)RAND_MAX - 0.5) / ad_id_vec_size;
    for (b = 0; b < user_id_vec_size; b++) for (a = 0; a < user_table_size; a++)
        user_id_vec[a * user_id_vec_size + b] = (rand() / (real)RAND_MAX - 0.5) / user_id_vec_size;
    // CreateBinaryTree();
}


struct train_record {
int query_id;
int user_id;
};

bool ReadTrainRecord(FILE* fin, train_record* record) {
  char url[100];
  int query_id, user_id;
  if (fscanf(fin, "%d%d", 
         &record->query_id,
         &user_id) == EOF){
    return false;
  }
  
  record->user_id = user_id_map[user_id];
  return true;
}
  

enum STATUS {
KEY_WORD,
TITLE,
DESC
};

void *TrainModelThread(void *id) {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    long long ad_id_start, ader_id_start, user_id_start;
    unsigned long long next_random = (long long)id;
    train_record cur_record;    
    // STATUS last_status = DESC;
    // int ader_id_vector_end = ader_id_vec_size;
    // int ad_id_vector_end = ader_id_vec_size + ad_id_vec_size;
    int user_id_vector_end = user_id_vec_size;
    int word_vector_end = user_id_vec_size + word_vec_size;
    

    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                    word_count_actual / (real)(train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            if (!ReadTrainRecord(fi, &cur_record)){
              break;
            }
            // ad_id_start = cur_record.ad_id * ad_id_vec_size;
            // ader_id_start = cur_record.ader_id * ader_id_vec_size;
            user_id_start = cur_record.user_id * user_id_vec_size;
            
            vector<int>* token_list = &query_tokens[cur_record.query_id];

            size_t i = 0;
            while (1) {
                int id;                

                if (i > token_list->size()) {
                  break;
                } else if (i == token_list->size()) {
                  id = 0; // add </s>
                } else {
                  id = (*token_list)[i++] + 1;
                }
                if (id >= vocab_size || word_table[id].cnt < min_count) {
                  continue;
                } 
                word_count++;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(word_table[id].cnt / (sample * train_words)) + 1) * (sample * train_words) / word_table[id].cnt;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = id;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi)) break;
        if (word_count > train_words / num_threads) break;
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        if (cbow) {  //train the cbow architecture
            // in -> hidden
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                for (c = 0; c < layer1_size; c++) neu1[c] += word_vec[c + last_word * layer1_size];
            }
            if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                f = 0;
                l2 = vocab[word].point[d] * layer1_size;
                // Propagate hidden -> output
                for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                if (f <= -MAX_EXP) continue;
                else if (f >= MAX_EXP) continue;
                else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                // 'g' is the gradient multiplied by the learning rate
                g = (1 - vocab[word].code[d] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                // Learn weights hidden -> output
                for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
            }
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += neu1[c] * weight[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * weight[c + l2];
                for (c = 0; c < layer1_size; c++) weight[c + l2] += g * neu1[c];
            }
            // hidden -> in
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                for (c = 0; c < layer1_size; c++) word_vec[c + last_word * layer1_size] += neu1e[c];
            }
        } else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                // l1 = last_word * layer1_size;
                l1 = last_word * word_vec_size;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                    f = 0;
                    l2 = vocab[word].point[d] * layer1_size;
                    // Propagate hidden -> output
                    for (c = 0; c < layer1_size; c++) f += word_vec[c + l1] * syn1[c + l2];
                    if (f <= -MAX_EXP) continue;
                    else if (f >= MAX_EXP) continue;
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    // 'g' is the gradient multiplied by the learning rate
                    g = (1 - vocab[word].code[d] - f) * alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                    // Learn weights hidden -> output
                    for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * word_vec[c + l1];
                }
                // NEGATIVE SAMPLING
                if (negative > 0) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    // for (c = 0; c < layer1_size; c++) f += word_vec[c + l1] * weight[c + l2];
                    // for (c = 0; c < ader_id_vec_size; c++) f += ader_id_vec[c + ader_id_start] * weight[c + l2];
                    // for (c = 0; c < ader_id_vec_size; c++) f += ader_id_vec[c + ader_id_start] * weight[c + l2];
                    for (c = 0; c < user_id_vec_size; c++) f += user_id_vec[c + user_id_start] * weight[c + l2];
                    for (c = 0; c < word_vec_size; c++) f += word_vec[c + l1] * weight[c + l2 + user_id_vector_end];
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * weight[c + l2];
                    // for (c = 0; c < layer1_size; c++) weight[c + l2] += g * word_vec[c + l1];
                    // for (c = 0; c < ader_id_vec_size; c++) weight[c + l2] += g * ader_id_vec[c + ader_id_start];
                    // for (; c < ad_id_vec_size; c++) weight[c + l2] += g * ad_id_vec[c + ad_id_start];
                    for (c = 0; c < user_id_vec_size; c++) weight[c + l2] += g * user_id_vec[c + user_id_start];
                    for (c = 0; c < word_vec_size ; c++) weight[c + l2 + user_id_vector_end] += g * word_vec[c + l1];
                }
                // Learn weights input -> hidden
                // for (c = 0; c < layer1_size; c++) word_vec[c + l1] += neu1e[c];
                // for (c = 0; c < ader_id_vec_size; c++) ader_id_vec[c + ader_id_start] += neu1e[c];
                // for (c = 0; c < ad_id_vec_size; c++) ad_id_vec[c + ad_id_start] += neu1e[c + ader_id_vector_end];
                for (c = 0; c < user_id_vec_size; c++) user_id_vec[c + user_id_start] += neu1e[c];
                // fix word vec for (c = 0 ; c < word_vec_size; c++) word_vec[c + l1] += neu1e[c + ad_id_vector_end];
            }
        }
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void output(const char * output_path) {
  printf("output vector to %s\n", output_path);
  
  FILE* output_file = fopen(output_path, "wb");

  printf("user");
  
  fprintf(output_file,"%lld %lld\n", user_table_size, user_id_vec_size);
  for (int i = 0; i < user_table_size; i++) {
    for (int j = 0; j < user_id_vec_size; j++) {
      fprintf(output_file, "%f ", user_id_vec[i * user_id_vec_size + j]);
    }
    fprintf(output_file, "\n");
  }
}

void LoadWordVec() {
  FILE* f_in = fopen(word_vec_path, "rb");
  if (!f_in) {
    printf("failed to open word_vec_path:%s\n", word_vec_path);
  } 

  int tmp;
  fscanf(f_in, "%d%lld", &tmp, &word_vec_size);
  
  long long a, b;
  a = posix_memalign((void **)&word_vec, 128, (long long)vocab_size * word_vec_size * sizeof(real));
  if (word_vec == NULL) {printf("Memory allocation failed for word_vec\n"); exit(1);}

  for (int i = 0; i < tmp ; i++) {
    for (int j = 0; j < word_vec_size; j++) {
      float input;
      fscanf(f_in, "%f", &input);
      word_vec[i * word_vec_size + j] = input;
    }
  }
  fclose(f_in);
}

void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    if (read_vocab_file[0] != 0) ReadVocab2(); else LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    LoadWordVec();
  

    layer1_size = word_vec_size + user_id_vec_size;

    ReadIdMap(user_id_map_path, &user_id_map);
    user_table_size = user_id_map.size();

    //Read token
    ReadTokenList(query_token_file, &query_tokens);

    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    //fo = fopen(output_file, "wb");
    if (classes == 0) {
        // printf("class == 0;save word vector; ");
        output(output_file);
        return;  
        printf("class == 0;save word vector; ");
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&word_vec[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", word_vec[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) {
                    cent[layer1_size * cl[c] + d] += word_vec[c * layer1_size + d];
                    centcn[cl[c]]++;
                }
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * word_vec[c * layer1_size + b];
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-keyword <file>\n");
        printf("\t\tkeyword token<file>\n");
        printf("\t-title <file>\n");
        printf("\t\ttitle token<file>\n");
        printf("\t-desc<file>\n");
        printf("\t\tdesc token<file>\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
        printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (0 = not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
        printf("\t-ader-size <int>\n");
        printf("\t\tsize of ader vector\n");
        printf("\t-ad-size <int>\n");
        printf("\t\tsize of ad size\n");
        printf("\t-ader-map <int>\n");
        printf("\t\tnum of ader\n");
        printf("\t-ad-map <int>\n");
        printf("\t\tnum of ad \n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
        return 0;
    }
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-ader-size", argc, argv)) > 0) ader_id_vec_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-ad-size", argc, argv)) > 0) ad_id_vec_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-user-size", argc, argv)) > 0) user_id_vec_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-ader-map", argc, argv)) > 0) strcpy(ader_id_map_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-ad-map", argc, argv)) > 0) strcpy(ad_id_map_path, argv[i+1]);
    if ((i = ArgPos((char *)"-user-map", argc, argv)) > 0) strcpy(user_id_map_path, argv[i+1]);
    if ((i = ArgPos((char *)"-word-vec", argc, argv)) > 0) strcpy(word_vec_path, argv[i+1]);
    if ((i = ArgPos((char *)"-keyword", argc, argv)) > 0) strcpy(keyword_token_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-title", argc, argv)) > 0) strcpy(title_token_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-desc", argc, argv)) > 0) strcpy(desc_token_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-query", argc, argv)) > 0) strcpy(query_token_file, argv[i + 1]);
  
    layer1_size = word_vec_size + ader_id_vec_size + ad_id_vec_size;

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}
