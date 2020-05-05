#!/bin/sh  

ropt=0
while getopts ri:o:m:d: name  
do  
    case $name in  
        r)ropt=1;;
        i)iopt=$OPTARG;;
		o)oopt=$OPTARG;;
		m)mopt=$OPTARG;;
		d)dopt=$OPTARG;;
        *)echo "Invalid arg!";;  
    esac  
done  

shift $((OPTIND-1))
if [ $ropt = 0 ]
then
	python execute.py -i $iopt -o $oopt -m $mopt -d $dopt
else
	python execute.py -r -i $iopt -o $oopt -m $mopt -d $dopt
fi