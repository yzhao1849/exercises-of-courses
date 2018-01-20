PrimeNumber=[0]*100
for i in range(2,101,1):
    k=2
    NotPrime=i
    while(NotPrime*k<=100):
        PrimeNumber[NotPrime*k-1]=1
        k+=1
Index=0
for number in PrimeNumber:
    Index+=1
    if number==0:
        print(Index)
    
    
