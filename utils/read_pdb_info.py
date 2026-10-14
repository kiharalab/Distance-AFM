import os
import numpy as np

def read_pdb_info(pdb_file,seq_len):

    coords = np.zeros((seq_len,3))
    with open(pdb_file,"r") as f:
        for line in f.readlines():
            if(line.startswith('ATOM') and line[13:16] == 'CA '):
                line = line.strip()
                resn=int(line[22:26].replace(' ',''))-1
                x=float(line[30:38])
                y=float(line[38:46])
                z=float(line[46:55])
                coords[resn] = np.array([x,y,z])

    return coords

def get_seq(seq_file):
    with open(seq_file, 'r') as s:
        s.readline() #read the '>' line
        seq = ""
        for line in s:
            seq += line.strip()
    return seq

def get_all_sequences(seq_file):
    sequences = {} 
    with open(seq_file, 'r') as s:
        for line in s:
            line = line.strip() 
            if not line:
                continue
            if line.startswith(">"):
                seq_name = line[1:] 
                sequences[seq_name] = "" 
                current_seq_name = seq_name 
            else:
                sequences[current_seq_name] += line 
    return sequences

def get_total_sequence_length(sequences):
    total_length = 0 
    for seq in sequences.values():
        total_length += len(seq)
    return total_length 