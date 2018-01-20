Each=0.90
for i in range(10):
    Price1=Each*(i+1)
    Price2=Each*(i+11)
    print("{:d} balls: {:.2f} euros    {:d} balls: {:.2f} euros".format(i+1,Price1,i+11,Price2))
