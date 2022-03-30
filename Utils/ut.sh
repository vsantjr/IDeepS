#!/bin/bash

#######
# Script to connect and transfer file from/to SDumont

# We assume that the name of the user is name.user and the name of the project is proj. Replace these with the proper information.

# Connect: ./ut.sh -c 0

# File TO SD: ./ut.sh -t filename

# File FROM SD: ./ut.sh -f filename

# Dir FROM SD: ./ut.sh -d subdir
#######

# This is the dir in our SDumont account where you will receive files from and send files to.
dirinout="work"

while getopts c:t:f:d: flag
do
    case "${flag}" in
        c) 
        	echo "Connect!"
        	ssh name.user@login.sdumont.lncc.br
        	exit;;
        t) 
            	echo "File TO SD!"
            	f=${OPTARG}
            	scp $f name.user@login.sdumont.lncc.br:/scratch/proj/name.user/$dirinout
        	exit;;
        f) 
        	echo "File FROM SD!"
        	f=${OPTARG}
            	scp name.user@login.sdumont.lncc.br:/scratch/proj/name.user/$dirinout/$f .
        	exit;;
        d) 
        	echo "Dir FROM SD!"
        	subdir=${OPTARG}
            	scp -r name.user@login.sdumont.lncc.br:/scratch/proj/name.user/$dirinout/$subdir .
        	exit;;	
    esac
done
