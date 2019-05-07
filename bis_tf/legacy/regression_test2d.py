#!/bin/python

from __future__ import print_function
import os
import tempfile
import json
import sys
import argparse
import nibabel as nib
import numpy as np
import bis_tf_utils as bisutil

def load_flatten(fname):
    a=nib.load(fname).get_data()
    print('.....\t loaded '+fname+' dim=',a.shape)
    return a.flatten()


def run_test(outputfile="log.txt",temp=True,outdir=None,golddir="/home/xenios/testing",steps=100,mintest=0,maxtest=-1) :

    if (temp==True):
        output_directory=tempfile.mkdtemp()
        print("..... Beginning comparison mode gold=",golddir,"tmp=",output_directory)
    else:
        output_directory=outdir
        print("..... Beginning creation mode=",output_directory)
        
    steps=int(steps)


    mydir=os.path.dirname(os.path.abspath(__file__))
    scriptpath=os.path.join(mydir,'bis_tf_fcnmodel.py')
    dpath2  = os.path.abspath(os.path.join(mydir,'../sampledata'))
    dpath3  = os.path.abspath(os.path.join(mydir,'../../sampledata'))
    
    command = "python3 "+scriptpath
    optbase2d="--train -i "+dpath2+"/mni_training.txt  -p 32 -b 16  --fixed_random -s "+str(steps)+" --num_connected 32 --num_filters 16"
    optbase3d="--train -i "+dpath3+"/smallCT.txt -t "+dpath3+"/smallCTmask.txt  -p 16 -b 16  --fixed_random -s "+str(steps)+" --num_connected 32 --num_filters 16"

    mode = [ 0 ,0,0,0, 1,1,0,1 ]
    suffixname = [ "2dclass","2dclasssmooth", "2dregr", "2dregrsmooth","3dclass","3dclasssmooth","2dplustclass", "3dplustclass" ]
    options    = [ optbase2d+" -t "+dpath2+"/masks_training.txt",
                   optbase2d+" -t "+dpath2+"/masks_training.txt -l 10.0",
                   optbase2d+" -t "+dpath2+"/dmap_training.txt --metric l2",
                   optbase2d+" -t "+dpath2+"/dmap_training.txt --metric l2 -l 10.0",
                   optbase3d+" ",
                   optbase3d+" -l 10.0 ",
                   optbase2d+" --second_input "+dpath2+"/dmap_training.txt -t "+dpath2+"/masks_training.txt -l 2.0",
                   optbase3d+" --second_input "+dpath3+"/smallCT.txt -l 2.0",
               ]

    secondoptions = [ "",
                      "",
                      "",
                      "",
                      "",
                      "",
                      " --second_input "+dpath2+"/mni_dmap/MNI_T1_1mm_0020_0020_stripped_mask_0020_MNI_T1_1mm_0020_dmap.nii.gz",
                      " --second_input "+dpath3+"/smallct/smallCT_20.nii.gz" 
                  ]
    
    imagename= [ ""+dpath2+"/mni_images/MNI_T1_1mm_0020_MNI_T1_1mm_0020.nii.gz",
                 ""+dpath3+"/smallct/smallCT_20.nii.gz" ]

    outname ="20.nii.gz"
    outdict= { }

    mintest=bisutil.force_inrange(mintest,0,len(suffixname)-1)
    if (maxtest<0):
        maxtest=len(suffixname)-1
    maxtest=bisutil.force_inrange(maxtest,mintest,len(suffixname)-1)

    print("..... Will execute tests:"+str(mintest)+":"+str(maxtest))

    for i in range (mintest,maxtest+1):

        print("..............................................................................................")        
        print("..... Running test:"+str(i))
        print(".....")

        dirname = os.path.join(output_directory,suffixname[i])
        os.mkdir(dirname)
        
        compname=golddir+'/'+suffixname[i]+'/'+outname
        outfile=os.path.join(dirname,outname)
        logfile=os.path.join(dirname,'log.txt')
        tmpfile=os.path.join(dirname,'tmp.nii.gz')
        
        md=mode[i]
        cmd0=command +" "+options[i]+" -o "+dirname
        bisutil.execute_command(cmd0)
        
        cmd=command +" --fixed_random -i "+imagename[md] +" -m "+dirname+" -o "+outfile+" -b 256"+secondoptions[i]
        bisutil.execute_command(cmd)

        
        outdict[i]= { 
            'command' : cmd0,
            'command2' : cmd,
            'suffix'  : suffixname[i]
        }

        if (temp==True):

            print('.....\n.....')
            imagedata1 = load_flatten(compname)
            imagedata2 = load_flatten(outfile)
            cc=np.corrcoef(imagedata1, imagedata2)[0, 1]
            ccs=str(round(cc,4))
            print('.....\n.....\t\t cc='+ccs)
            outdict[i]['result']=ccs
            print("..............................................................................................")        
            print(".....")
            
    if (temp==True):
        for key in outdict:
            print("..... "+outdict[key]['suffix']+":"+outdict[key]['result'])

    with open(outputfile, 'w') as fp:
        json.dump(outdict, fp, sort_keys=True,indent=4)

    print("..............................................................................................")        
    print('..... Final Result saved in '+outputfile)
            
if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='Load an image for patch sampling.')
   parser.add_argument('-g','--golddir',  help='Comparison directory or generated directory')
   parser.add_argument('-c','--create', help='Create data',default=False,action='store_true')
   parser.add_argument('-o','--output',  help="Output text file",default="log.txt")
   parser.add_argument('-s','--steps', help='Number of steps (iterations)',default=100,type=int)
   parser.add_argument('--mintest', help='First test to run',default=0,type=int)
   parser.add_argument('--maxtest', help='Last test to run',default=-1,type=int)


   args = parser.parse_args()

   if (args.create):
       run_test(outputfile=args.output,
                temp=False,
                outdir=args.golddir,
                steps=args.steps,
                mintest=args.mintest,
                maxtest=args.maxtest,
                golddir="")
   else:
       run_test(outputfile=args.output,
                temp=True,
                outdir=None,
                steps=args.steps,
                mintest=args.mintest,
                maxtest=args.maxtest,
                golddir=args.golddir)

