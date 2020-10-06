#!/bin/bash

#FIRST ARGUMENT MUST BE THE INPUT FILE: MULTIPLE SEQUENCE ALIGNMENT OF HOMOLOGOUS RNA SEQUNCES IN FASTA FORMAT
input=$1

#SECOND ARGUMENT MUST BE THE NUMBER OF REQUIRED PROCESSES TO BE RUN IN PARALLEL
np=$2

awk 'BEGIN{FS="";}{for(i=1;i<=NF;i++)if($1!=">")printf $i;if($1==">" && NR>1)printf "\n";}' $input | sed "s/T/U/g" | sed "s/t/U/g" > temp
grep -v "[^ACUCGaucg-]" temp > clean
rm temp
awk 'BEGIN{FS="";}{if(NR==1){for(i=1;i<=NF;i++){targ[i]=$i;if($i!="-")printf $i;}printf "\n";}else{for(i=1;i<=NF;i++)if(targ[i]!="-")printf $i;printf "\n"}}' clean > seqs
rm clean
#RUNNING BOLTZMANN LEARNING 
mpicxx -std=c++0x dca.cpp -O2 -o dca.o 
#mpicxx -std=c++0x dca.cpp -o dca.o
mpirun -np $np ./dca.o seqs 0.0 0.9 
sort -grk1 scores -o scores
rm seqs
