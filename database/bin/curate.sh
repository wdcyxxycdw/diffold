#!/bin/bash
FILE=`readlink -e $0`
bindir=`dirname $FILE`
rootdir=`dirname $bindir`
cd $rootdir

echo "extract fasta"
if [ -s "rnacentral_species_specific_ids.fasta.gz" ];then
    echo "  step 1: decompressing..."
    zcat rnacentral_species_specific_ids.fasta.gz > temp_rnacentral_raw.fasta
    echo "  step 2: filtering headers..."
    grep -ohP "^\S+" temp_rnacentral_raw.fasta > temp_rnacentral_filtered.fasta
    echo "  step 3: processing fasta..."
    $bindir/fastaNA temp_rnacentral_filtered.fasta > temp_rnacentral_processed.fasta
    echo "  step 4: cataloging..."
    $bindir/catRNAcentral temp_rnacentral_processed.fasta rnacentral.fasta rnacentral.tsv
    rm rnacentral_species_specific_ids.fasta.gz temp_rnacentral_raw.fasta temp_rnacentral_filtered.fasta temp_rnacentral_processed.fasta
fi

echo "makeblastdb"
$bindir/makeblastdb -in rnacentral.fasta -parse_seqids -hash_index -dbtype nucl

echo "unzip Rfam"
if [ -s "Rfam.cm.gz" ];then
    gzip -d Rfam.cm.gz
fi
rm Rfam.cm.*
$bindir/cmpress Rfam.cm

echo "extract nt"
for filename in `ls nt*tar.gz`;do
    tar -xvf $filename
    rm $filename
done
if [ -s "nt.gz" ];then
    echo "  step 1: decompressing nt.gz..."
    zcat nt.gz > temp_nt_raw.fasta
    echo "  step 2: filtering headers..."
    grep -ohP "^\S+" temp_nt_raw.fasta > temp_nt_filtered.fasta
    echo "  step 3: processing fasta..."
    $bindir/fastaNA temp_nt_filtered.fasta > nt
    rm nt.gz temp_nt_raw.fasta temp_nt_filtered.fasta
fi
