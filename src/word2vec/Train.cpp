#include "Train.h"

bool CMP(const int &a,const int &b){
  return a<b;
}
Train::Train(){
};
Train::~Train(){};

int Train::parse_line(const string &line, Record &rec){
	string::size_type last_pos;
	last_pos = 0;
	int i=0,last=0,j=0;
	int len=line.size();
	char ch=9,ch1;
	
	stringstream sstr;
	string tmp;
	while(i<len){
		while(i<len && (ch1=line[i]) != ch)i++;
		//cout<<" i am hereeeee"<<endl;
		tmp=line.substr(last,i-last);
		//cout<<tmp<<endl;
		//break;
		//system("pause");
		//if(i>10)break;
		last= ++i;
		sstr<<tmp;
		switch(j){
			case 0:{
				sstr>>rec.click;
		    break;
			}
		  case 1:{
				sstr>>rec.impres;
		    break;
			}
		  case 2:{
				sstr>>rec.disURL;
		    break;
			}
			case 3:{
				sstr>>rec.adID;
		    break;
			}
			case 4:{
				sstr>>rec.advID;
		    break;
			}
			
			case 5:{
				sstr>>rec.depth;
		    break;
			}
			case 6:{
				sstr>>rec.position;
		    break;
			}
			case 7:{
				sstr>>rec.queryID;
		    break;
			}
			
			case 8:{
				sstr>>rec.keywordID;
		    break;
			}
			case 9:{
				sstr>>rec.titleID;
		    break;
			}
		  case 10:{
				sstr>>rec.descriptionID;
		    break;
			}
			case 11:{
				sstr>>rec.userID;
		    break;
			}
			default:
				break;
		}
		sstr.clear();
		j++;
	}
	return 1;
}

	
	
void Train::maxID(){
	ifstream fin("./data/training.txt");
	
	if(!fin){
		cerr << "can not open the training file "<<endl;
		return ;
	}
	
	string line,tmp;
	//string::size_type pos,pos_1,pos_2;
	//char ch=9;
	stringstream sstr;
	//int id,tokenid,len,i;
	
	/*int cnt=0;
	while(getline(fin,line)){
		
		cnt++;
	}
	cout<<" queryid_tokenid.txt has line= "<<cnt<<endl;
	fin.close();
	return ;
*/	
	//unsigned int max=0;
	unsigned int max_adID,max_advID,max_userID,max_click,max_click_im,max_im,max_im_cl;
	max_adID=max_advID=max_userID=max_click=max_click_im=max_im=max_im_cl=0;
	Record rec;
	while(getline(fin,line)){
		parse_line(line,rec);
		if(rec.click>max_click){
			max_click=rec.click;
			max_click_im=rec.impres;
		}
		if(rec.impres>max_im){
			max_im=rec.impres;
			max_im_cl=rec.click;
		}
		if(rec.adID>max_adID){
			max_adID=rec.adID;
		}
		
		if(rec.advID>max_advID){
			max_advID=rec.advID;
		}
		if(rec.userID>max_userID){
			max_userID=rec.userID;
		}
		
	}
	cout<<"max_adID = "<<max_adID
	    <<"\nmax_advID    = "<<max_advID
			<<"\nmax_userID   = "<<max_userID
			<<"\nmax_click    = "<<max_click
			<<"\nmax_click_im = "<<max_click_im
			<<"\nmax_im       = "<<max_im
			<<"\nmax_im_cl    = "<<max_im_cl<<endl;	
		
	fin.close();
	return ;
}

void Train::ToVector(string &filename,vector< vector<int> > &vec){
  ifstream fin(filename.c_str());
	if(!fin){
		cerr << "can not open the file "<<filename<<endl;
		_exit(-1);
	}
	
	string line,tmp;
	string::size_type pos;
	char ch=9;
	
	int id,tokenid,len,i;
	
	int max=-1;
	while(getline(fin,line)){
		pos=line.find_first_of(ch);
		tmp=line.substr(0,pos);
		stringstream sstr;
		sstr<<tmp;
		sstr>>id;
		sstr.clear();
		
		pos++;
		
		len=line.size();
		i=pos;
		
		//cout<<"\n====>"<<id<<" " ;
		//	bool find=false;
	
		while(i<len){
			while(i<len && line[i] != '|')i++;
			if(i<len &&line[i] == '|'){
				tmp=line.substr(pos,i-pos);
				stringstream sstr;
		    sstr<<tmp;
		    sstr>>tokenid;
		    vec[id].push_back(tokenid);
		    sstr.clear();
		   // queryIDvec[id].push_back(tokenid);
		    if(tokenid>max)max=tokenid;
		    //cout<<tokenid<<" ";
		    pos=++i;
			}
			else{
				tmp=line.substr(pos);
				stringstream sstr;
		    sstr<<tmp;
		    sstr>>tokenid;
		    vec[id].push_back(tokenid);
		    sstr.clear();
		    //queryIDvec[id].push_back(tokenid);
		    if(tokenid>max)max=tokenid;
		    //cout<<tokenid<<" ";
			}
			//system("pause");
				
		}	
		sort(vec[id].begin(),vec[id].end(),CMP);
		
		//if(id%100000==0)cout<<line<<endl;;
	}
//	cout<<" max word token id = "<<max<<endl;
	
	fin.close();
	//return max;
}
void Train::q_kFeature(){
  
  ifstream fin("./data/training.txt");
  if(!fin){
    cout<<"can not open the train.txt"<<endl;
    _exit(-1);
  }
  
  string query_file="./data/queryid_tokensid.txt";
  string key_file="./data/purchasedkeywordid_tokensid.txt";
  
  vector< vector<int> > vecQ,vecK;
  vecQ.resize(26243605+2);
  vecK.resize(1249784+2);
  
  ToVector(query_file,vecQ);
  ToVector(key_file,vecK);
  
  string line;
  Record rec;
  CTR array[qkArraySize];
  for(int i=0;i<qkArraySize;i++){
    array[i].click=0;
    array[i].impres=0;
  }
  int i,j,len_q,len_k,com,diff_q,diff_k;
  while(getline(fin,line)){
    parse_line(line,rec);
    i=0;j=0;
    len_q=vecQ[rec.queryID].size();
    len_k=vecK[rec.keywordID].size();
    com=0;
   
    while(i<len_q&&j<len_k&&(vecQ[rec.queryID][i] == vecK[rec.keywordID][j])){
        i++;j++;
        com++;
    }
    diff_q=len_q-com;
    diff_k=len_k-com;
    diff_q=diff_q>7?7:diff_q;
    diff_k=diff_k>7?7:diff_k;
    
    if(diff_k>diff_q){
      array[qkBase+diff_k].click += rec.click;
      array[qkBase+diff_k].impres += rec.impres;
    }
    else{
      array[diff_q].click +=rec.click;
      array[diff_k].impres +=rec.impres;
  }
  
  ofstream fout("./result/QKCTR.txt");
  if(!fout){
    cout<<"can not open the QKCTR "<<endl;
    _exit(-1);
  }
  for(int i=0;i<qkArraySize;i++){
    fout<<array[i].click*1.0/array[i].impres<<" ";
  }
  fout<<endl;
}
}
  
  
int Train::maxWordID(string &filename ){
	ifstream fin(filename.c_str());
	
	if(!fin){
		cerr << "can not open the file "<<filename<<endl;
		return -1;
	}
	
	string line,tmp;
	string::size_type pos;
	char ch=9;
	stringstream sstr;
	int id,tokenid,len,i;
	
	/*int cnt=0;
	while(getline(fin,line)){
		
		cnt++;
	}
	cout<<" queryid_tokenid.txt has line= "<<cnt<<endl;
	fin.close();
	return ;
*/	
	int max=-1;
	while(getline(fin,line)){
		pos=line.find_first_of(ch);
		tmp=line.substr(0,pos);
		
		sstr<<tmp;
		sstr>>id;
		sstr.clear();
		
		pos++;
		
		len=line.size();
		i=pos;
		
		//cout<<"\n====>"<<id<<" " ;
		//	bool find=false;
	
		while(i<len){
			while(i<len && line[i] != '|')i++;
			if(i<len &&line[i] == '|'){
				tmp=line.substr(pos,i-pos);
		    sstr<<tmp;
		    sstr>>tokenid;
		    sstr.clear();
		   // queryIDvec[id].push_back(tokenid);
		    if(tokenid>max)max=tokenid;
		    //cout<<tokenid<<" ";
		    pos=++i;
			}
			else{
				tmp=line.substr(pos);
		    sstr<<tmp;
		    sstr>>tokenid;
		    sstr.clear();
		    //queryIDvec[id].push_back(tokenid);
		    if(tokenid>max)max=tokenid;
		    //cout<<tokenid<<" ";
			}
			//system("pause");
				
		}	
		//if(id%100000==0)cout<<line<<endl;;
	}
	cout<<" max word token id = "<<max<<endl;
	
	fin.close();
	return max;
}

int Train::maxTokenID(string &filename){
	
	ifstream fin(filename.c_str());
	
	if(!fin){
		cerr << "can not open the file "<<filename<<endl;
		return -1;
	}
	
	string line,tmp;
	string::size_type pos;
	char ch=9;
	stringstream sstr;
	int id;
	
	/*int cnt=0;
	while(getline(fin,line)){
		
		cnt++;
	}
	cout<<" queryid_tokenid.txt has line= "<<cnt<<endl;
	fin.close();
	return ;
*/	
	int max=-1;
	while(getline(fin,line)){
		pos=line.find_first_of(ch);
		tmp=line.substr(0,pos);
		
		sstr<<tmp;
		sstr>>id;
		sstr.clear();
		
		if(id>max){
			max=id;
		}
	}
	return max;
}

void Train::initCTR(CTR *array,int num){
  for(int i=0;i<num;i++){
    array[i].click = 0;
    array[i].impres = 0;
  }
}
void Train::ctrCount(){
	
	ifstream fin("./data/training.txt");
	
	if(!fin){
		cerr << "can not open the training file "<<endl;
		return ;
	}
	
	string line;
	Record rec;
	//int click ,impres;
	CTR *queryCTR= new CTR[queryNum];
	CTR *descripCTR = new CTR[descripNum];
	CTR *keywordCTR = new CTR[keywordNum];
	CTR *titleCTR = new CTR[titleNum];
	
	for(int i=0;i<queryNum;i++){
		queryCTR[i].click=0;
		queryCTR[i].impres=0;
	}
	
	for(int i=0;i<descripNum; i++){
		descripCTR[i].click=0;
		descripCTR[i].impres=0;
	}
	
	for(int i=0; i<keywordNum;i++){
		keywordCTR[i].click=0;
		keywordCTR[i].impres=0;
	}
	
	for(int i=0;i<titleNum;i++){
		titleCTR[i].click=0;
		titleCTR[i].impres=0;
	}
	
	
	while(getline(fin,line)){
		
		parse_line(line,rec);		
		//click = rec.click;
		//impres = rec.impres;
		
		queryCTR[rec.queryID].click += rec.click;
		queryCTR[rec.queryID].impres += rec.impres;
		
		descripCTR[rec.descriptionID].click += rec.click;
		descripCTR[rec.descriptionID].impres +=rec.impres;
		
		keywordCTR[rec.keywordID].click += rec.click;
		keywordCTR[rec.keywordID].impres += rec.impres;
		
		titleCTR[rec.titleID].click += rec.click;
		titleCTR[rec.titleID].impres += rec.impres;
		
	}
	
	ofstream fout_q("./result/queryCTR.txt");
	if(!fout_q){	
		cerr<<"can not open the queryCTR.txt"<<endl;
		return;
	}
	
	ofstream fout_d("./result/descripCTR.txt");
	if(!fout_d){	
		cerr<<"can not open the descripCTR.txt"<<endl;
		return;
	}	
		
	ofstream fout_k("./result/keywordCTR.txt");
	if(!fout_k){
		cerr<<"can not open the keywordCTR.txt"<<endl;
		return;
	}	
	
	ofstream fout_t("./result/titleCTR.txt");
	if(!fout_t){
		cerr<<"can not open the titleCTR.txt"<<endl;
		return;
	}		
	
	float tmp;
	for(int i=0;i<queryNum;i++){
	  if(queryCTR[i].impres== 0){
	    tmp=0;
	  }else{
		tmp=queryCTR[i].click*1.0/queryCTR[i].impres;
	  }
	  fout_q<<tmp<<endl;
		
	}
	
	for(int i=0;i<descripNum;i++){
	  if(descripCTR[i].impres ==0){
	    tmp=0;
	  }else{
		tmp=descripCTR[i].click*1.0/descripCTR[i].impres;
	  }
	  fout_d<<tmp<<endl;
	}
	
	for(int i=0; i<keywordNum;i++){
	  if(0 == keywordCTR[i].impres){
	    tmp=0;
	  }else{
		tmp=keywordCTR[i].click*1.0/keywordCTR[i].impres;
	  }
	  fout_k<<tmp<<endl;
	}
	
	for(int i=0;i<titleNum;i++){
	  if(0 == titleCTR[i].impres){
	    tmp=0;
	  }else{
		tmp=titleCTR[i].click*1.0/titleCTR[i].impres;
	  }
    fout_t<<tmp<<endl;
	}
	
	delete [] queryCTR;
	delete [] descripCTR;
	delete [] keywordCTR;
	delete [] titleCTR;
	
	
	fout_q.close();
	fout_d.close();
	fout_k.close();
	fout_t.close();
	fin.close();
	
	
}
void Train::ctrCountAd(){

	ifstream fin("./data/training.txt");
	
	if(!fin){
		cerr << "can not open the training file "<<endl;
		return ;
	}
	
	string line;
	Record rec;
	CTR *adCTR = new CTR[adNum];
	CTR *advCTR = new CTR[advNum];
	CTR *userCTR = new CTR[userNum];

	initCTR(adCTR,adNum);
	initCTR(advCTR,advNum);
	initCTR(userCTR,userNum);

	//	Record rec;
	while(getline(fin,line)){
	  parse_line(line,rec);

	  adCTR[rec.adID].click += rec.click;
	  adCTR[rec.adID].impres += rec.impres;

	  advCTR[rec.advID].impres += rec.impres;
	  advCTR[rec.advID].click += rec.click;

	  userCTR[rec.userID].click += rec.click;
	  userCTR[rec.userID].impres += rec.impres;

	}
	ofstream fout_a("./result/adCTR.txt");
	ofstream fout_v("./result/advCTR.txt");
	ofstream fout_u("./result/userCTR.txt");

	if(!fout_a){
	  cout<<"can not open the adCTR.txt"<<endl;
	  return ;
	}
	if(!fout_v){
	  cout<<"can not open the advCTR.txt"<<endl;
	  return ;
	}
	if(!fout_u){
	  cout<<"can not open the userCTR.txt"<<endl;
	  return ;
	}


	float tmp;
	for(int i=0;i<adNum; i++){
	  if(0 == adCTR[i].impres){
	    tmp=0;
	  }else{
	    tmp = adCTR[i].click*1.0/adCTR[i].impres;
	  }
	  fout_a<<tmp<<endl;
	}

	for(int i=0;i < advNum;i++){
	  if(0 == advCTR[i].impres){
	    tmp=0;
	  }else{
	    tmp = advCTR[i].click*1.0/advCTR[i].impres;
	  }
	  fout_v<<tmp<<endl;
	}

	for(int i=0;i<userNum;i++){
	  if(0 == userCTR[i].impres){
	    tmp=0;
	  }else{
	    tmp = userCTR[i].click*1.0/userCTR[i].impres;
	  }
	  fout_u<<tmp<<endl;
	}

	delete [] adCTR;
	delete [] advCTR;
	delete [] userCTR;
	fin.close();
	fout_a.close();
	fout_v.close();
	fout_u.close();
}

void Train::ctrDepth(){
   ifstream fin("./data/training.txt");
	
	if(!fin){
		cerr << "can not open the training file "<<endl;
		return ;
	}
	
	string line;
	Record rec;	
  double depthCTR[6][6];
  int depthcnt[6][6];
  
  for(int i=0;i<6;i++){
		for(int j=0;j<6;j++){
			depthCTR[i][j]= 0;
			depthcnt[i][j]=0;
			
		}
	}
 unsigned  int pos,depth;
 double pctr=0;
	while(getline(fin,line)){
	  parse_line(line,rec);
	  pos= rec.position;
	  depth=rec.depth;
	  pos = pos>5?5:pos;
	  depth = depth > 5 ?5:depth;
    
    pctr=rec.click*1.0/rec.impres;
    
	  depthCTR[pos][depth] += pctr;
	  depthcnt[pos][depth]++; 
	  
	}
	ofstream fout_d("./result/depth_posCTR.txt");
	if(!fout_d){
		cout<<"can not open the depth_posCTR.txt"<<endl;
		return ;
	}
	double tmp;
	for(int i=0;i<6;i++){
		for(int j=0;j<6;j++){
      if(depthcnt[i][j]==0 )depthCTR[i][j]=0;
      else{
         depthCTR[i][j] = depthCTR[i][j]/depthcnt[i][j];
      }
      fout_d<<depthCTR[i][j]<<" ";
		}
		fout_d<<endl;
	}
	
	fin.close();
	fout_d.close();
}
	


/*		
int main(){
//	maxID();
  freopen("rmse.txt","w", stdout);
  Train tr;
  // tr.maxID();
  //tr.ctrCount();
  // tr.ctrCountAd();
  tr.ctrDepth();
  
  /* string filename;
  int max=0,tmp;
  filename="./data/queryid_tokensid.txt";
  tmp=tr.maxTokenID(filename);
  if(tmp>max){
		max=tmp;
	}
	cout<< filename <<" maxID= "<<tmp<<endl;
	
	//descriptionid_tokensid.txt
	filename="./data/descriptionid_tokensid.txt";
  tmp=tr.maxTokenID(filename);
  if(tmp>max){
		max=tmp;
	}
	cout<< filename <<" maxID= "<<tmp<<endl;

  //purchasedkeywordid_tokensid.txt
	filename="./data/purchasedkeywordid_tokensid.txt";
  tmp=tr.maxTokenID(filename);
  if(tmp>max){
		max=tmp;
	}
	cout<< filename <<" maxID= "<<tmp<<endl;

  //titleid_tokensid.txt
  filename="./data/titleid_tokensid.txt";
  tmp=tr.maxTokenID(filename);
  if(tmp>max){
		max=tmp;
	}
	cout<< filename <<" maxID= "<<tmp<<endl;


  cout<<"the final max num id =  "<<max<<endl;;
  
  	system("pause");
	return 0;
}
		*/			
		
		
		
		
		
		
		
		
		
		
		
		







	
