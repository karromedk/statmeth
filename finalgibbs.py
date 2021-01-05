import numpy as np
import math as math
from matplotlib import pyplot as plt
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import copy

def generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif, saveOpt=False, displayOpt=False):
    # Generate thetas for Background and Motif with corresponding Dirichlet priors
    thetaBg = np.random.dirichlet(alphaListBg)
    thetaMw = np.random.dirichlet(alphaListMw, size=lenMotif)  # Generates Theta_j for each motif position

    seqList = np.zeros((numSeq, lenSeq))
    startList = np.zeros(numSeq)

    for s in range(numSeq):
        # Get the starting point of motif
        r = np.random.randint(lenSeq - lenMotif + 1)
        startList[s] = r

        for pos in range(lenSeq):
            # Sample from Background
            if pos < r or pos >= r + lenMotif:
                value = np.where(np.random.multinomial(1, thetaBg) == 1)[0][0]
            # Sample from Motif
            else:
                j = pos - r  # index of motif letter
                value = np.where(np.random.multinomial(1, thetaMw[j]) == 1)[0][0]

            seqList[s, pos] = value

    seqList = seqList.astype(int)
    startList = startList.astype(int)

    # Store the motifs in the sequences into a multidimensional array for debugging.
    motifList = np.zeros((numSeq, lenMotif))
    for i in range(numSeq):
        r = startList[i]
        motifList[i] = seqList[i, r:r + lenMotif]
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

def Gibbs(iterations, seqList,alphaListBg,alphaListMw,numSeq,lenSeq,lenMotif):
    Rlist=np.zeros(shape=(iterations+1,numSeq))
    Rlist[0]=np.random.randint(0, lenSeq-lenMotif+1, size=numSeq)
    N = numSeq * lenMotif
    B=numSeq*(lenSeq-lenMotif)

    #----marginal likelihoods first half background-----#
    margLikeB = math.lgamma(np.sum(alphaListBg)) - math.lgamma(B + np.sum(alphaListBg))
    # ------marginal likelihoods first half magic-------#
    margLikeM = math.lgamma(np.sum(alphaListMw)) - math.lgamma(N + np.sum(alphaListMw))
    #------------------total counts in seqList for every letter------------------------#
    totalCounts=dict(Counter(item for sequence in seqList for item in sequence))
    #----------------------------------------------------------------------------------#
    for it in tqdm(range(iterations)):
        R=list(Rlist[it])
        for sq in range(numSeq):
            p=np.zeros(lenSeq-lenMotif)
            BkOld = {0:0,1:0,2:0,3:0}
            NkjOld = [{0:0,1:0,2:0,3:0} for j in range(lenMotif)]
            Nkj=[{0:0,1:0,2:0,3:0} for j in range(lenMotif)]
            #------- /gamma(alpha(k) (last)----#
            margLike_bg=margLikeB
            margLike_mw = margLikeM * lenMotif
            for k in range(len(alphaListBg)):
                margLike_bg-=math.lgamma(alphaListBg[k])
            for j in range(lenMotif):
                for k in range(len(alphaListBg)):
                    margLike_mw-=math.lgamma(alphaListMw[k])
            #---------------------------#
            for s in range(lenSeq-lenMotif):
                R[sq]=s
                #-----compute Nkj--#
                w = [seqList[i][int(R[i]):int(R[i] + lenMotif)] for i in range(numSeq)]
                NkjNew = [dict(Counter([w[i][j] for i in range(numSeq)])) for j in range(lenMotif)]
                for j in range(lenMotif):
	                Nkj[j] = {key: NkjNew[j][key] - NkjOld[j].get(key, 0) for key in NkjNew[j].keys()}
                #-----computing Bk---------#
                x = {0: 0, 1: 0, 2: 0, 3: 0}
                for j in range(lenMotif):
                    for k in range(len(alphaListMw)):
                        try:
                            x[k]=x[k]+NkjNew[j][k]
                        except KeyError:
                            pass
                BkNew={key: totalCounts[key] - x.get(key,0) for key in totalCounts.keys()}
                Bk = {key: BkNew[key] - BkOld.get(key,0) for key in BkNew.keys()}
                #computing margLike for B

                for k in range(len(alphaListBg)):
                    try:
                        if s == 0:
                            margLike_bg += math.lgamma(Bk[k]+alphaListBg[k])
                        elif Bk[k] > 0:
                            for i in range(math.floor(BkOld[k] + alphaListBg[k] - 1),
                                           math.floor(BkNew[k] + alphaListBg[k] - 1)):
                                margLike_bg += float(np.log(i))

                        elif Bk[k] < 0:
                            for i in range(math.floor(BkNew[k] + alphaListBg[k] - 1),
                                           math.floor(BkOld[k] + alphaListBg[k] - 1)):
                                margLike_bg -= float(np.log(i))
                        else:
                            pass
                    except KeyError:
                        pass
                #computing margLike  for magic
                for j in range(lenMotif):
                    for k in range(len(alphaListMw)):
                        try:
                            if s==0:
                                margLike_mw+=math.lgamma(Nkj[j][k]+alphaListMw[k])
                            elif Nkj[j][k]>0 or Nkj[j][k]<0:
                                margLike_mw-=math.lgamma(NkjOld[j][k]+alphaListMw[k])
                                margLike_mw+=math.lgamma(NkjNew[j][k]+alphaListMw[k])
                            else:
                                pass
                        except KeyError:
                            pass
            #-----------Full cond-------------#
                p[s]=margLike_bg+margLike_mw # marglike for b and sq
                if (margLike_bg+margLike_mw)==-float('inf') or (margLike_bg+margLike_mw)==float('inf'):
                    return None
                BkOld=BkNew.copy()
                NkjOld=NkjNew.copy()

            p=np.exp(p-np.max(p))
            p=p/sum(p)
            Rtemp=np.argmax(np.random.multinomial(1,p))
            R[sq]=Rtemp
        Rlist[it+1]=list(R)
    return Rlist
#---estimated potential scale reduction-----#
def epsc(plot, chain):
    n = plot.shape[1]
    sequences=[]
    W1=[]
    for chains in range(chain):
       for item in plot[chains][0][50:]:
           sequences.append(item)
       W1.append(np.var(plot[chains][0][50:]))
    B=np.var(sequences)
    W=np.mean(W1)
    V_theta=(1 - 1. / n) * W + 1. / n * B
    R=np.sqrt(V_theta/W)
    print("R is: ", R)

def question1():
    alphaListBg=[1,1,1,1]
    #alphaListMw=[0.9,0.9,0.9,0.9]
    alphaListMw=[0.9, 0.9, 0.9 ,0.9]
    numSeq=5
    lenSeq=30
    lenMotif=10
    chain=1
    iterations=200
    Rlist = []
    seqList, startList, motifList, thetaBg, thetaMw = generateSequences(alphaListBg, alphaListMw, numSeq, lenSeq, lenMotif,saveOpt=False , displayOpt=False)
    print("SEQLIST", seqList)
    for chains in range(chain):
        print("chain1: ", chains)
        R=Gibbs(iterations, seqList,alphaListBg,alphaListMw,numSeq,lenSeq,lenMotif)
        if R is None:
            return None
        Rlist.append(R)
    plot=np.zeros(shape=(chain,numSeq,iterations))
    for chains in range(chain):
        print("chain2:", chains)
        for i in range(iterations):
            for j in range(numSeq):
                plot[chains][j][i]=Rlist[chains][i][j]
    xplot=np.linspace(1,iterations,iterations)
    for chains in range(chain):
        print("chain3:", chains)
        for j in range(numSeq):
            plt.plot(xplot,plot[chains][j])
    #------estimated potential scale reduction---------#
    # source: https://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
    epsc(plot,chain)
    #------plot-----#
    plt.title("Right start positions: " + str(startList))
    plt.ylabel('Start positions')
    plt.xlabel('Iterations')
    #plt.show()
    #-----medians----#
    corr=0
    medians=[] 
    for j in range(numSeq):
        medians.append(math.floor(np.median(plot[chains][j])))
        if math.floor(np.median(plot[chains][j]))==startList[j]:
            corr+=1
    print("Median: ", medians)
    accuracy=corr/numSeq
    print("Accuracy with medians was: ",str(100*accuracy))
    seq=len(startList)
  #----most common----#
    correct=0
    freqV=list()
    for i in range(numSeq):
        data =Counter(R[:,i])
        most_common=int(data.most_common(1)[0][0])
        freqV.append(most_common)
        if most_common == int(startList[i]):
            correct+=1
            print("Correct was ", data.most_common(1))
        else:
            print("Wrong", data.most_common(1), "Correct was", startList[i] )
    print("Accuracy with most common was " + str(100*correct/seq) +"%")
    print("\nRight start positions: ")
    print(startList)
    print("most guessed start positions")
    print(freqV)
    print("Median: ", medians)
    #histogram(numSeq, plot)
    ####--- plot histogram---####
def histogram(numSeq, plot):
    # lists with a dict for each sequence
    guesses = []

    for x in range(numSeq):
        # create empty dict where the keys are the numbers that been guessed and the values are number of quesses for that specific number
        this_seq = defaultdict(lambda: 0)

        # read all guesses for specific sequence.
        for y in plot:
            for z in y[x]:
                # +1  for guess in variable z
                this_seq[z] += 1

        guesses.append(this_seq)

    # make histogram
    seq_number = 0

    # for each sequence
    for seq in guesses:
        # find lowest and highest guesses, to make limits in x
        min_guess = min(seq.keys())
        max_guess = max(seq.keys())

        # lists for data in x- and y-led
        x_axis = []
        y_axis = []

        i = min_guess

        while i <= max_guess:
            x_axis.append(i)
            y_axis.append(seq[i])
            i += 1

        # draw diagram
        plt.bar(x_axis, y_axis, align='center', alpha=1)
        plt.ylabel('Number of guesses')
        plt.xlabel('Guessed start position')
        plt.title('Sequence ' + str(seq_number))

        plt.show()

        seq_number += 1



def question2():
    alphaListBg=[1.0,1.0,1.0,1.0] #given
    alphaListMw=[0.8,0.8,0.8,0.8] #given
    alphabet=['a','c','g','t']
    i=0
    seqList=[]
    with open('../data/2_1_data.txt', 'r') as f:
            for row in f:
                if i<25:
                    seqList.append([])
                    row=row.rstrip()
                    row=row.split(' ')
                    i+=1
                    for letter in row:
                        seqList[-1].append(alphabet.index(letter))

    numSeq=len(seqList)
    lenSeq=len(seqList[0])
    lenMotif=10
    iterations=100
    chain=1
    Rlist=[]
    for chains in range(chain):
        R=Gibbs(iterations, seqList,alphaListBg,alphaListMw,numSeq,lenSeq,lenMotif)
        if R is None:
            return None
        Rlist.append(R)
    plot=np.zeros(shape=(chain,numSeq,iterations))
    for chains in range(chain):
        for i in range(iterations):
            for j in range(numSeq):
                plot[chains][j][i]=Rlist[chains][i][j]
    xplot=np.linspace(1,iterations,iterations)
    median=[]
    for chains in range(chain):
        for j in range(numSeq):
            median.append(math.floor(np.median(plot[chains][j][50:])))
    for chains in range(chain):
        for j in range(1):
            plt.plot(xplot,plot[chains][j])
    medi =[]
    for med in range(1):
        medi.append(median[med])
    plt.xlabel('Iterations')
    plt.ylabel('Positions')

    print("guessed start positions medians:")
    print(median)
    #----epsc------------#
    epsc(plot, chain)
    #-----------------------#
     #----most common----#
    freqV=list()
    for i in range(0,numSeq):
        data =Counter(R[50:,i])
        most_common=int(data.most_common(1)[0][0])
        freqV.append(most_common)
    print("most guessed start positions")
    print(freqV)
    avg = []
    for a in range(1):
        avg.append(freqV[a])
    plt.title('Median: ' + str(medi) + ', Most Common: ' + str(avg))
    plt.show()

    ####--- plot histogram---####
    histogram(numSeq,plot)

def main():
    question2()

if __name__ == "__main__":
    main()

