import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r",  action="store_true",
   help="If specified, turn on the relevance feedback.")
parser.add_argument("-i", type=str,
                    help="input query file")
parser.add_argument("-o", type=str,
                    help="the output rank list file")
parser.add_argument("-m", type=str,help="model-dir")
parser.add_argument("-d", type=str,
                    help="NTCIR-dir")
args = parser.parse_args()
print(args.r)