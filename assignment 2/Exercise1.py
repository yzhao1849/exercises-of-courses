def transcribe(dna):
    rna=[]
    i=len(dna)-1
    while(i>=0):
        if dna[i]=='A':
            rna.append('U')
        elif dna[i]=='G':
            rna.append('C')
        elif dna[i]=='C':
            rna.append('G')
        elif dna[i]=='T':
            rna.append('A')
        i=i-1
    #RNA=''.join(str(i) for i in rna)
    RNA=''.join(rna)
    return RNA

"""The DNA fragments are chosen from the gene sequence of
chlamyopsin 4 light-gated ion channel of Chlamydomonas reinhardtii
"""
RNA=transcribe("GCAGCACCATACTTGACATCTGTCGCC")
print(RNA)

RNA=transcribe("TTGGGCGCGAGCTGCTATTTGTAAC")
print(RNA)

RNA=transcribe("GTGTTACTGCGCGGTGC")
print(RNA)

RNA=transcribe("TTTGCGTGTGTGTCTGGACTGAT")
print(RNA)

RNA=transcribe("AAGGGTTGACCAGGGGCAGAGAGCGCTGCCCCTCATGCAAGACCGCAAACTTCAGCGTGACTGTACCCGA")
print(RNA)