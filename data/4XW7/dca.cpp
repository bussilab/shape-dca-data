#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <time.h>
#include <cstdlib>
#include <mpi.h>  
#include <random>   

#define alfah 0.01
#define alfaj 0.01
#define tau 1000
#define exponent 1
#define lenMean 50000 //NUMBER OF STEPS OF THE STHOCASTIC GRADIENT DESCENT
#define lenWait 10000
#define lenObs 5000
using namespace std;

unsigned nt2index(char nucleotide){
  if(nucleotide=='A' || nucleotide=='a') return 0;
  if(nucleotide=='U' || nucleotide=='u') return 1;
  if(nucleotide=='C' || nucleotide=='c') return 2;
  if(nucleotide=='G' || nucleotide=='g') return 3;
  if(nucleotide=='-') return 4;
  assert(0);
}

class histo{
  public:
  void set(unsigned i, unsigned j,double x);
  double counts[5][5];
  double sums[5][5];
  double count[5];
  double sum[5];
  histo(){
    for(unsigned i=0;i<5;++i){ count[i]=0; sum[i]=0; for(unsigned j=0;j<5;++j){sums[i][j]=0.0; counts[i][j]=0.0;}}
  }
};
int main(int argc,char*argv[]){
	MPI_Init(&argc,&argv);
	int rank,np;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&np);
        std::vector<double> reduce_buffer1;
        std::vector<double> reduce_buffer2;
	double  l, sim_thresh; 
	unsigned iold, inew, jold;
   	std::ifstream input(argv[1]);
	l=(double)atof(argv[2]);
	//l=0.0; 
	//READING ALIGNMENTS
	std::string line;
	std::vector<string> seq;
	std::vector<string> names;
	int iseq=0, seqsize, len, seqid;
	while(getline(input, line)){
		seq.resize(iseq+1);
		seq[iseq]+=line;
		iseq++;
	}
	seqsize=seq.size();
	len=seq[0].length();
	for(unsigned i=0;i<seqsize;++i)
		assert(seq[i].length()==len);

	//REWEIGHTING OF SEQUENCES BASED ON SIMILARITY
	//sim_thresh=(double)atof(argv[3]);
	//double identity[seqsize], Meff=0.0;
	//for(unsigned is=0;is<seqsize;++is){
	//	identity[is]=1.0;
	//	for(unsigned it=0;it<seqsize;++it){
	//		seqid=0;
	//		for(unsigned in=0;in<len;++in){
	//			char nt1=seq[is][in];
	//			char nt2=seq[it][in];
	//			if(nt2index(nt1)==nt2index(nt2)) seqid++;
	//		}
	//		if((double)seqid/len>=sim_thresh) identity[is]++; 
	//	}
	//}
	//for(unsigned is=0;is<seqsize;++is) 
	//	Meff+=(double)(1.0/identity[is]);
        //std:: cerr << Meff << "\n";
        //double invnorm=1.0/(Meff); 
	double invnorm=1.0/(double)seqsize;
	double Jcum, Jcumold, deltaH, t;
	unsigned nucl[len];
	std::vector<histo> pair(len*len);
	std::vector<histo> sing(len);
	std::vector<histo> J(len*len);
	std::vector<histo> new_J(len*len);
	std::vector<histo> h(len);
	std::vector<histo> new_h(len);
	std::vector<histo> F(len*len);
	std::vector<histo> new_F(len*len);
	std::vector<histo> f(len);
	std::vector<histo> new_f(len);
	//RANDOM NUMBERS 
	//std::random_device rd {}; 
	std::mt19937 g{rank}; 
   	std::uniform_int_distribution<int> unif(0,seqsize-1);
	std::uniform_int_distribution<int> kind(0,4);
	std::uniform_real_distribution<double> rando(0.0,1.0);

	//INITIALIZING THE LEARNING PROCEDURE WITH A RANDOM SEQUENCE IN THE ALIGNMENT
	unsigned init=unif(g);
	for(unsigned in=0;in<len;in++){
		char nt1=seq[init][in];
		nucl[in]=nt2index(nt1);
	}	
	//FREQUENCY COUNTS ON ALIGNMENTS
	for(unsigned in=0;in<len;in++){ 
		for(unsigned jn=0;jn<len;jn++){
			for(unsigned is=0;is<seqsize;++is){
				char nt1=seq[is][in];
				char nt2=seq[is][jn];
				pair[in*len+jn].counts[nt2index(nt1)][nt2index(nt2)]+=1.0; //(1.0/identity[is]);
			}
		}
		for(unsigned is=0;is<seqsize;++is){
			char nt1=seq[is][in];
			sing[in].count[nt2index(nt1)]+=1.0; //(1.0/identity[is]);
		}
	}
	//LEARNING PROCEDURE
	t=0;
	while(t<lenMean+lenObs){
		//GENERATION OF RANDOM SEQUENCE VIA METROPOLIS ALGORITHM
		int track=0;
		while(track<20){
			unsigned in=0;
			while(in<len){
				iold=nucl[in];
				inew=iold;
				while(inew==iold) inew=kind(g);
				Jcum=0.0;	
				Jcumold=0.0;
				for(unsigned jn=0;jn<len;jn++){
					jold=nucl[jn];
					if(in!=jn){
						Jcumold += J[in*len+jn].counts[iold][jold];
						Jcum += J[in*len+jn].counts[inew][jold];
					}
				}	
				deltaH = h[in].count[inew]-h[in].count[iold] + Jcum-Jcumold;
				if(deltaH<0 || exp(-deltaH)>rando(g)) nucl[in]=inew;
				in++;
			}
		track++;
		}
		t++;
		//PARAMETERS UPDATE
		if(t<lenMean){	
			for(unsigned in=0;in<len;in++)	
				for(unsigned i=0;i<5;i++){
					if(i==4)h[in].count[i]=0;
					else
						h[in].count[i]=h[in].count[i]+(alfah/std::pow(1+t/(double)tau,exponent))*((nucl[in]==i?1:0)-sing[in].count[i]*invnorm-l*(h[in].count[i]));
					for(unsigned jn=0;jn<len;jn++)
						for(unsigned j=0;j<5;j++){
							if(i==4 || j==4)J[in*len+jn].counts[i][j]=0;
							else 
							J[in*len+jn].counts[i][j]=J[in*len+jn].counts[i][j]+(alfaj/std::pow(1+t/(double)tau,exponent))*((nucl[in]==i?1:0)*(nucl[jn]==j?1:0)-pair[in*len+jn].counts[i][j]*invnorm-l*(J[in*len+jn].counts[i][j]));
						}		
				}
			//AVERAGE OF PARAMETERS FROM ALL PROCESSORS
                        reduce_buffer1.resize(5*len);
			for(unsigned in=0;in<len;in++)
                                for(unsigned k=0;k<5;k++) reduce_buffer1[in*5+k]=h[in].count[k];

                        MPI_Allreduce(MPI_IN_PLACE,reduce_buffer1.data(),5*len,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			for(unsigned in=0;in<len;in++)
                                for(unsigned k=0;k<5;k++) new_h[in].count[k]=reduce_buffer1[in*5+k];
				
                        reduce_buffer2.resize(25*len*len);
			for(unsigned in=0;in<len;in++)for(unsigned jn=0;jn<len;jn++)
                                for(unsigned k=0;k<5;k++) for(unsigned l=0;l<5;l++)
                                  reduce_buffer2[in*25*len+jn*25+k*5+l]=J[in*len+jn].counts[k][l];

                        MPI_Allreduce(MPI_IN_PLACE,reduce_buffer2.data(),25*len*len,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			for(unsigned in=0;in<len;in++)for(unsigned jn=0;jn<len;jn++)
                                for(unsigned k=0;k<5;k++) for(unsigned l=0;l<5;l++)
                                  new_J[in*len+jn].counts[k][l]=reduce_buffer2[in*25*len+jn*25+k*5+l];

			
			for(unsigned in=0;in<len;in++)for(unsigned i=0;i<5;i++){
				h[in].count[i]=new_h[in].count[i]/(double)np;
				for(unsigned jn=0;jn<len;jn++)for(unsigned j=0;j<5;j++)
					J[in*len+jn].counts[i][j]=new_J[in*len+jn].counts[i][j]/(double)np;	
			}
			//STORING PARAMETERS 
			if(t>=lenWait){
				for(unsigned in=0;in<len;in++){
					for(unsigned i=0;i<5;i++){
						h[in].sum[i]+=h[in].count[i];
						for(unsigned jn=0;jn<len;jn++)
							for(unsigned j=0;j<5;j++)
								J[in*len+jn].sums[i][j]+=J[in*len+jn].counts[i][j];
					}
				}
			}	
		}
		//ASSIGNING AVERAGE OF STORED PARAMETERS 
		if((int)t==lenMean){
			for(unsigned in=0;in<len;in++)for(unsigned i=0;i<5;i++){h[in].count[i]=h[in].sum[i]/(double)(lenMean-lenWait);
				for(unsigned jn=0;jn<len;jn++)for(unsigned j=0;j<5;j++) {J[in*len+jn].counts[i][j]=J[in*len+jn].sums[i][j]/(double)(lenMean-lenWait);}}
		}
		//FREQUENCY COUNTS: SEQUENCES GENERATED FROM DISTRIBUTION WITH ASSIGNED PARAMETERS
		if(t>lenMean){
			for(unsigned in=0;in<len;in++)for(unsigned i=0;i<5;i++){
				if(nucl[in]==i)
					f[in].count[i]++;
				for(unsigned jn=0;jn<len;jn++)
					for(unsigned j=0;j<5;j++)
						if(nucl[in]==i && nucl[jn]==j)
							F[in*len+jn].counts[i][j]++;
			}
		}
	}
	//AVERAGE OF FREQUENCIES FROM ALL PROCESSORS
	for(unsigned in=0;in<len;in++)for(unsigned jn=0;jn<len;jn++)
		MPI_Allreduce(&F[in*len+jn].counts[0][0],&new_F[in*len+jn].counts[0][0],25,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	for(unsigned in=0;in<len;in++)
		MPI_Allreduce(&f[in].count[0],&new_f[in].count[0],5,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	
	//WRITING RESULTS
	if(!rank){
		double norm[len][len], Jsum[len][len], row_sum[len], all=0.0;
		//THE FOLLOWING TWO FILES CONTAIN THE EMPIRICAL FREQUENCY COUNTS VS ESTIMATED ONES
		std::stringstream filepi;
		filepi<<"conv_fi";
		std::ofstream pi(filepi.str().c_str());
		std::stringstream filepij;
		filepij<<"conv_fij";
		std::ofstream pij(filepij.str().c_str());
		//THE FOLLOWING TWO FILES CONTAIN THE ESTIMATED HEMILTONIAN PARAMETERS
		std::stringstream fileh;
		fileh<<"h_vals";
		std::ofstream hval(fileh.str().c_str()); 
		std::stringstream filej;
		filej<<"j_vals";
		std::ofstream jval(filej.str().c_str()); 
		//THE FOLLOWING FILE CONTAINS FINAL COVARIANCE SCORES (MINIMIZED FROBENIUS NORM OF COUPLINGS MATRIXES AFTER APC CORRECTION) AND CORRESPONDING PAIR INDEXES
		std::stringstream filenorm;
		filenorm<<"scores";
		std::ofstream frob(filenorm.str().c_str()); 
		
		for(unsigned in=0;in<len;in++){
			for(unsigned i=0;i<5;i++){
				f[in].count[i]=new_f[in].count[i]/(double)np;
				pi<<sing[in].count[i]*invnorm<<" "<<f[in].count[i]/(double)lenObs-l*h[in].count[i]<<"\n"; 
				hval<<h[in].count[i]<<"\n";	
			}
		}
		for(unsigned in=0;in<len;in++)	
			for(unsigned jn=0;jn<len;jn++){
				Jsum[in][jn]=0.0;
				for(unsigned i=0;i<5;i++){
					for(unsigned j=0;j<5;j++){
						F[in*len+jn].counts[i][j]=new_F[in*len+jn].counts[i][j]/(double)np;
						pij<<pair[in*len+jn].counts[i][j]*invnorm<<" "<<F[in*len+jn].counts[i][j]/(double)lenObs-l*J[in*len+jn].counts[i][j]<<"\n"; 
						jval<<J[in*len+jn].counts[i][j]<<"\n";	
						Jsum[in][jn]+=J[in*len+jn].counts[i][j];
						J[in*len+jn].count[i]+=J[in*len+jn].counts[i][j];
					}
				}
			}
		//MINIMIZATION OF FROBENIUS NORM AND APC CORRECTION
		for(unsigned in=0;in<len;in++){
			row_sum[in]=0.0;
			for(unsigned jn=0;jn<len;jn++){
				norm[in][jn]=0.0;
				for(unsigned i=0;i<4;i++)
					for(unsigned j=0;j<4;j++){
						J[in*len+jn].counts[i][j]=J[in*len+jn].counts[i][j]-J[in*len+jn].count[i]/5.0-J[jn*len+in].count[j]/5.0+Jsum[in][jn]/25.0;
						norm[in][jn]+=J[in*len+jn].counts[i][j]*J[in*len+jn].counts[i][j];
					}
				norm[in][jn]=sqrt(norm[in][jn]);
				all+=norm[in][jn];
				row_sum[in]+=norm[in][jn];
			}
		}
		for(unsigned in=0;in<len;in++)	
			for(unsigned jn=0;jn<len;jn++){
				norm[in][jn]=norm[in][jn]-(row_sum[in]*row_sum[jn])/all;
				if(in<jn) frob<<norm[in][jn]<<" "<<in<<"  "<<jn<<"\n";
			}
		
	}
	MPI_Finalize();	
	return 0;
}	
