import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

def generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif, saveOpt=True, displayOpt=False):

    # Generate thetas for Background and Motif with corresponding Dirichlet priors
    thetaBg = np.random.dirichlet(alphaListBg)
    thetaMw = np.random.dirichlet(alphaListMw, size=lenMotif) # Generates Theta_j for each motif position

    seqList = np.zeros((numSeq, lenSeq))
    startList = np.zeros(numSeq)

    for s in range(numSeq):
        # Get the starting point of motif M-W+1
        r = np.random.randint(lenSeq-lenMotif+1)
        startList[s] = r

        for pos in range(lenSeq):
            # Sample from Background
            if pos < r or pos >= r+lenMotif:
                value = np.where(np.random.multinomial(1,thetaBg)==1)[0][0]
            # Sample from Motif
            else:
                j = pos - r # index of motif letter
                value = np.where(np.random.multinomial(1,thetaMw[j])==1)[0][0]

            seqList[s,pos] = value

    seqList = seqList.astype(int)
    startList = startList.astype(int)

    # Store the motifs in the sequences into a multidimensional array for debugging.
    motifList = np.zeros((numSeq,lenMotif))
    for i in range(numSeq):
        r = startList[i]
        motifList[i] = seqList[i,r:r+lenMotif]
    motifList = motifList.astype(int)

    if displayOpt:
        print("Background Parameters")
        print("Alpha")
        print(alphaListBg)
        print("Theta")
        print(thetaBg)

        print("\nSequence List")
        print(seqList)
        print("\nStarting Positions of Motifs")
        print(startList)

        print("\nMotifs")
        print(motifList)

        print("\nMotif Parameters")
        print("Alpha")
        print(alphaListMw)
        print("Theta")
        print(thetaMw)

    if saveOpt:
        filename = "data/alphaBg.txt"
        np.savetxt(filename, alphaListBg, fmt='%.5f')

        filename = "data/alphaMw.txt"
        np.savetxt(filename, alphaListMw, fmt='%.5f')

        filename = "data/thetaBg.txt"
        np.savetxt(filename, thetaBg, fmt='%.5f')

        filename = "data/thetaMw.txt"
        np.savetxt(filename, thetaMw, fmt='%.5f')

        filename = "data/sequenceList.txt"
        np.savetxt(filename, seqList, fmt='%d')

        filename = "data/startList.txt"
        np.savetxt(filename, startList, fmt='%d')

        filename = "data/motifsInSequenceList.txt"
        np.savetxt(filename, motifList, fmt='%d')

    return seqList, startList, motifList, thetaBg, thetaMw
def margLikeMagic(alpha, totalcount, n_counts):
    K = n_counts.shape[0]
    J = n_counts.shape[1]
    margLike = np.zeros(J)
    #division becomes subtraction with log. Log is used to not get overflow/underflow
    math1 = math.lgamma(np.sum(alpha)) - math.lgamma(totalcount + np.sum(alpha))
    for j in range(J):
        math2 = 0
        for k in range(K):
            math2 += math.lgamma(n_counts[k,j] + alpha[k]) - math.lgamma(alpha[k])
        margLike[j] = math2 + math1
    return margLike

def margLikeBg(alpha, totalcount, b_counts):
    K = b_counts.shape[0]
    margLike = 1
    #division becomes subtraktion with log. Log is used to not get overflow
    math1 = math.lgamma(np.sum(alpha)) - math.lgamma(totalcount + np.sum(alpha))
    math2 = 0
    for k in range(K):
        math2 += math.lgamma(b_counts[k] + alpha[k]) - math.lgamma(alpha[k])
    #multiplication becomes addition (sum) with log
    margLike = math2 + math1

    return margLike

def gibbs(seqList, alphaListBg, alphaListMw, iterations, lenMotif):
    N = seqList.shape[0] #len(segList) number of sequences
    M = seqList.shape[1] #len(segList[0]) length of sequences
    K = alphaListBg.shape[0]
    R = np.zeros((iterations, N))
    for i in range(N):
        R[0][i] = np.random.randint(0, (M-lenMotif+1))

    nTot = N*lenMotif
    bTot = N*(M-lenMotif)

    nCounts = np.zeros((K, lenMotif))
    bCounts = np.zeros(K)
    for it in range(1, iterations):
        R_temp = R[it-1]
        for n in range(N):
            full_conditional_vec = np.zeros(M-lenMotif+1)
            for position in range(M-lenMotif+1):
                R_temp[n] = position
                nCounts = np.zeros((K, lenMotif))
                bCounts = np.zeros(K)

                for sequence in range(N):
                    j = 0
                    for m in range(M):
                        symbol = seqList[sequence,m]
                        if m in range(int(R_temp[sequence]), int(R_temp[sequence])+lenMotif):

                            nCounts[int(symbol)-1, j] += 1
                            j += 1
                        else:
                            bCounts[int(symbol)-1] += 1

                margLike_magic = margLikeMagic(alphaListMw, nTot, nCounts)
                margLike_bg = margLikeBg(alphaListBg, bTot, bCounts)


                term = 0
                for j in range(margLike_magic.shape[0]):
                    term = margLike_magic[j] + term
                product = term + margLike_bg
                full_conditional_vec[position] = product

            prob = np.exp(full_conditional_vec - np.max(full_conditional_vec))
            prob = prob/ np.sum(prob)
            ind_random = np.random.multinomial(1, prob)
            R_temp[n] = np.argmax(ind_random)

        R[it] = R_temp
    return R

def plot_convergence(sequenceVec, truth, show=True, chain=True, sequence=True):
    truth_vec = [truth]*len(sequenceVec)
    if chain:
        plt.figure(sequence)
        plt.plot(sequenceVec, label='start position samples, chain =' +str(int(chain)))
        plt.legend(loc='upper left')
        if show:
            plt.title(" ")
            plt.plot(truth_vec, label='truth = ' +str(int(truth)), alpha =0.7)
            plt.legend(loc='upper left')
            plt.ylim(-1, 6)
            plt.savefig('sequence_'+str(sequence)+'.png')

    else:
        plt.plot(sequenceVec, label='start position samples')
        plt.plot(truth_vec, label='truth = ' +str(int(truth)), alpha =0.7)
        plt.legend(loc='upper left')
        plt.title(" ")
        if show:
            plt.show()


def convergence(numSeq, lenSeq, lenMotif, iterations, alphaListBg, alphaListMw):
    runs = 3
    seqList, startList, motifList, thetaBg, thetaMw = generateSequences(alphaListBg,\
    alphaListMw, numSeq=30, lenSeq=20, lenMotif=5, saveOpt=False, displayOpt=False)
    rList=list()
    for i in range(runs):
        R = gibbs(seqList, alphaListBg, alphaListMw, iterations, lenMotif)
        rList.append(R)

    for m in range(runs):
        for j in range(numSeq):
            if m != (runs-1):
                plot_convergence(rList[m][:,j], startList[j], run=(m+1), sequence=j)
            else:
                plot_convergence(rList[m][:,j], startList[j], show=True, run=(m+1), sequence=j) #only show the final plot


def multiple_accuracy(numSeq, lenSeq, lenMotif, iterations, alphaListBg, alphaListMw):
    """seqList, startList, motifList, thetaBg, thetaMw = generateSequences(alphaListBg, alphaListMw,\
    numSeq=30, lenSeq=20, lenMotif=5, saveOpt=True, displayOpt=False) """
    runs = 3
    lag = 5

    for i in range(runs):
        #R = gibbs(seqList, alphaListBg, alphaListMw, iterations, lenMotif)
        accuracy(R, startList, lag)

def accuracy(R, startList, lag=False):
    sequences=len(startList)
    correct=0
    freqV=list()

    for i in range(sequences):
        if lag!=False:
            data = Counter(R[70::lag,i])
        else:
            data =Counter(R[70:,i])

        most_common=int(data.most_common(1)[0][0])
        freqV.append(most_common)
        if most_common == int(startList[i]):
            correct+=1
            print("Correct was ", data.most_common(1))
        else:
            print("Wrong", data.most_common(1), "Correct was", startList[i] )

    print("Accuracy was " + str(100*correct/sequences) +"%")

def question1():
    alphaListBg = np.ones(4) * 1
    #alphaListMw = np.ones(4) * 0.9
    alphaListMw = [0.8, 0.8, 0.8, 0.8]
    numSeq = 5
    lenSeq = 10
    lenMotif = 10
    iterations = 200
    seqList, startList, motifList, thetaBg, thetaMw = generateSequences(alphaListBg, alphaListMw,\
        numSeq=30, lenSeq=20, lenMotif=5, saveOpt=True, displayOpt=False)
    R = gibbs(seqList, alphaListBg, alphaListMw, iterations, lenMotif)
    #multiple_accuracy(numSeq, lenSeq, lenMotif, iterations, alphaListBg, alphaListMw)
    #convergence(numSeq,lenSeq, lenMotif, iterations, alphaListBg, alphaListMw)

def main():
    question1()
    #question2()


if __name__ == "__main__": main()