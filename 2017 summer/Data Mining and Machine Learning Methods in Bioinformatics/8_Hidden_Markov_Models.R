### Hidden Markov Models
library(HMM)
startProbs=c(.5,.5)
transProbs = matrix(c(.9,.1,.1,.9),byrow=TRUE,nrow = 2)
emissionProbs = matrix(c(.4,.1,.1,.4,.2,.3,.3,.2),byrow=TRUE,nrow = 2)
hmm1=initHMM(c("alpha","beta"),c("A","T","C","G"),startProbs,transProbs,emissionProbs)
# entry transProbs[X,Y] gives the probability of a transition from state X to state Y
#observation="ATGGGACTCG" doesn't work
observation=c("A","T","G","G","G","A","C","T","C","G") #"ATGGGACTCG"

### Use Baum-Welch algorithm to determine the new transition A and emission B probabilities
bw1=baumWelch(hmm1,observation, maxIterations=100, delta=1E-9, pseudoCount=0)

### Viterbi decoding algorithm: determine the most likely hidden state generating the observation
vt1=viterbi(bw1$hmm,observation) 

### Calculate the forward-backward probabilities given the new parameter set
logForward=forward(bw1$hmm,observation)
print(exp(logForward))
logBackward=backward(bw1$hmm,observation)
print(exp(logBackward))
posterior1=posterior(bw1$hmm,observation)
print(posterior1)

### Viterbi training algorithm: estimate transition and emission probabilities
vTraining=viterbiTraining(hmm1, observation, maxIterations=100, delta=1E-9, pseudoCount=0)
posterior2=posterior(vTraining$hmm,observation)
vt2=viterbi(vTraining$hmm,observation)
