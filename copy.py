import random
import matplotlib.pyplot as plt
import math
import numpy as np
import time

plt.style.use('ggplot')

mutation = True
elitism = False
mutationPorcentage = .90
tournamentPercentage = 0.02
generations = 100
nPopulation = 400


fuzzyNetworks = 7

chromosomeSize = fuzzyNetworks*4

weight = 5


pMain = [8,25,4,45,10,17,35]



population = []

fa = []

yMain = []
xMain = []



plotDistanceX = []
plotDistanceY = []

plotGraphX = []
plotGraphY = []

line1 = []
line2 = []
line3 = []

def createPopulation():
    for i in range(0, nPopulation):
        chromosome=random.sample(range(0,256), chromosomeSize)
        population.append(chromosome)
    
    # population[0] = [193, 34, 248, 32, 99, 155, 0, 255, 58, 6 , 73, 71, 77, 8, 46, 237, 148, 32, 49, 157, 169, 159, 0, 255, 66, 16, 6, 242]

def populateFA(ei):
    fa.clear()
    global line1
    global line2
    global plotGraphX
    global plotGraphX

    plotGraphX = []
    plotGraphY = []
    sDistance = 999999999999999999999999

    for chromosome in population:
        fa_v = 0
        plotGraphXTemp = []
        plotGraphYTemp = []

        for i in range(0,1000):

            x = i/10
            y = getYFuzzy(chromosome, x)
            # y = getYpoint(chromosome[0]/weight,chromosome[1]/weight,chromosome[2]/weight,chromosome[3]/weight,chromosome[4]/weight,chromosome[5]/weight,chromosome[6]/weight, x)
            
            plotGraphXTemp.append(x)
            plotGraphYTemp.append(y)
            
            fa_v += abs(yMain[i] - y)
        
        fa.append(fa_v)

        if sDistance > fa_v:
            sDistance = fa_v
            plotGraphX = plotGraphXTemp
            plotGraphY = plotGraphYTemp

    
    plotDistanceX.append(ei)
    plotDistanceY.append(sDistance)


    line1, line2 = live_plotter(plotDistanceX,plotDistanceY, plotGraphX,plotGraphY,line1, line2, sDistance, chromosome)

def getFA(po):
    poFA = []
    for chromosome in po:
        fa_v = 0
        for i in range(0,1000):
            x = i/10
            y = getYFuzzy(chromosome, x)
            # y =  getYpoint(chromosome[0]/weight,chromosome[1]/weight,chromosome[2]/weight,chromosome[3]/weight,chromosome[4]/weight,chromosome[5]/weight,chromosome[6]/weight, x)
            
            fa_v += abs(yMain[i] - y)
        
        poFA.append(fa_v)
    return poFA

def getYpoint(a,b,c,d,e,f,g,xi):
    if c == 0 or e == 0:
        return 0
    else:
        return  a * (b * math.sin(xi/c) + d * math.cos(xi/e)) + f * xi - g

def creatMainGraph():
    for i in range(0,1000):
        xMain.append(i/10)
        yMain.append( getYpoint(pMain[0],pMain[1],pMain[2],pMain[3],pMain[4],pMain[5],pMain[6], xMain[i]) )
        
def betterOptions(participants):
    betterOption = participants[0]
    for i in participants:
        if(fa[betterOption] > fa[i]):
            betterOption = i

    return betterOption

def reproduction(f,m):
    cutPoint = random.randint(0, chromosomeSize * 8)
    cutIndex = int(cutPoint / 8)

    n = cutPoint - (cutIndex * 8)

    if(n == 0):
        fc = f[0:cutIndex] + m[cutIndex:chromosomeSize]
        mc = m[0:cutIndex] + f[cutIndex:chromosomeSize]
    else:

        lowMask = (2**n)-1
        highMask = ((2**chromosomeSize)-1)-((2**n)-1)

        lowPar1 = f[cutIndex] & lowMask
        lowPar2 = m[cutIndex] & lowMask

        highPar1 = f[cutIndex] & highMask
        highPar2 = m[cutIndex] & highMask

        child1 = lowPar1 | highPar2
        child2 = lowPar2 | highPar1

        fc = f[0:cutIndex] + [child2] + m[cutIndex+1:chromosomeSize]
        mc = m[0:cutIndex] + [child1] + f[cutIndex+1:chromosomeSize]

    return fc,mc

def mutation(childList):
    participants = random.sample(range(0,nPopulation), int(nPopulation * mutationPorcentage))
    
    for i in participants:
        cutPoint = random.randint(0, (chromosomeSize * 8) - 1)

        numberIndex = int(cutPoint / 8)

        n = cutPoint - (numberIndex * 8)

        numberToMutate = childList[i][numberIndex]

        bindata = '{0:08b}'.format(numberToMutate)

        bitNot = '1' if bindata[n] == '0' else '0'
        new = bindata[0:n] + bitNot + bindata[n+1:8]
        
        newInt = int(new, 2)

        childList[i][numberIndex] = newInt


def tournament():
    global population

    childList = []
    for i in range(0, int(nPopulation/2)):
            participants = random.sample(range(0,nPopulation), int(nPopulation * tournamentPercentage))
            f = betterOptions(participants)

            participants = random.sample(range(0,nPopulation), int(nPopulation * tournamentPercentage))
            m = betterOptions(participants)

            a,b = reproduction(population[f],population[m])

            childList.append(a)
            childList.append(b)
    
    mutation(childList)
    mutation(childList)
    # mutation(childList)
    # mutation(childList)
    # mutation(childList)
    # mutation(childList)
    # mutation(childList)
    # mutation(childList)

    if(elitism):

        completeList = population + childList
        faList = fa + getFA(childList)

        listIndex = {v: k for v, k in enumerate(faList)}
        sortedListIndex = list(dict(sorted(listIndex.items(), key=lambda item: item[1])).items())
        populationIndex = sortedListIndex[0:nPopulation]
        
        betterList = [v for v, k in populationIndex]

        population = [completeList[v] for v in betterList]

        pass
    else:
        population = childList


def live_plotter(x_vec,y1_data, x_vec2, y1_data2,line1, line2,sDistance, identifier='',pause_time=0.0001):
    global ax1
    global ax2
    global line3
    global plt

    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax1.plot(x_vec,y1_data,'-o',alpha=0.8)      

        fig2 = plt.figure(figsize=(10,5))
        ax2 = fig2.add_subplot(111)
        # create a variable for the line so we can later update it
        line2, = ax2.plot(x_vec2,y1_data2,'-o',alpha=0.8)
        line3, = ax2.plot(xMain,yMain,'-o',alpha=0.8)   

    #update plot label/title
    plt.ylabel('Y Label')

    plt.title( str(sDistance) + ' - ' + str(identifier))
    plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line1.set_xdata(x_vec)

    line2.set_ydata(y1_data2)
    line2.set_xdata(x_vec2)

    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        # plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
        ax1.set_ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

    if np.min(y1_data2)<=line2.axes.get_ylim()[0] or np.max(y1_data2)>=line2.axes.get_ylim()[1]:
        ax2.set_ylim([np.min(yMain)-np.std(yMain),np.max(yMain)+np.std(yMain)])

      # adjust limits if new data goes beyond bounds
    if np.min(x_vec)<=line1.axes.get_xlim()[0] or np.max(x_vec)>=line1.axes.get_xlim()[1]:
        # plt.xlim([np.min(x_vec)-np.std(x_vec),np.max(x_vec)+np.std(x_vec)])
        ax1.set_xlim([np.min(x_vec)-np.std(x_vec),np.max(x_vec)+np.std(x_vec)])

    if np.min(x_vec2)<=line2.axes.get_xlim()[0] or np.max(x_vec2)>=line2.axes.get_xlim()[1]:
        ax2.set_xlim([np.min(x_vec2)-np.std(x_vec2),np.max(x_vec2)+np.std(x_vec2)])

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1, line2


weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
def getYFuzzy(data_fuzzy, x):

    fuzzyNetworks

    bf = 0
    af = 0
    for i in range(fuzzyNetworks):
        m = data_fuzzy[i*4]
        de = data_fuzzy[(i*4)+1] 
        p = data_fuzzy[(i*4)+2] 
        q = data_fuzzy[(i*4)+3] 
        
        if de == 0:
            mf = 0
        else:
            mf = math.exp((-math.pow((x-m), 2))/(2*math.pow(de, 2)))

        a = mf*(p*x+q)

        bf += mf
        af += a


    if bf == 0:
        y = 0
    else:
        y = af/bf

    return y  


def getYFuzzyBackup(data_fuzzy, x):

    m1 = data_fuzzy[0]
    de1 = data_fuzzy[1]
    p1 = data_fuzzy[2]
    q1 = data_fuzzy[3]

    m2 = data_fuzzy[4]
    de2 = data_fuzzy[5]
    p2 = data_fuzzy[6]
    q2 = data_fuzzy[7]

    m3 = data_fuzzy[8]
    de3 = data_fuzzy[9]
    p3 = data_fuzzy[10]
    q3 = data_fuzzy[11]

    m4 = data_fuzzy[12]
    de4 = data_fuzzy[13]
    p4 = data_fuzzy[14]
    q4 = data_fuzzy[15]

    if de1 == 0 or de2 == 0 or de3 == 0 or de4 == 0:
        return 0
        
    mf1 = math.exp((-math.pow((x-m1), 2))/(2*math.pow(de1, 2)))
    mf2 = math.exp((-math.pow((x-m2), 2))/(2*math.pow(de2, 2)))
    mf3 = math.exp((-math.pow((x-m3), 2))/(2*math.pow(de3, 2)))
    mf4 = math.exp((-math.pow((x-m4), 2))/(2*math.pow(de4, 2)))

    bf = mf1+mf2+mf3+mf4

    a1 = mf1*(p1*x+q1)
    a2 = mf2*(p2*x+q2)
    a3 = mf3*(p3*x+q3)
    a4 = mf4*(p4*x+q4)

    af = a1+a2+a3+a4

    y = af/bf

    return y   

if __name__ == "__main__":
    
    creatMainGraph()

    line1, line2 = live_plotter([0],[0], [0],[0],line1, line2, 0)
    # time.sleep(3)
    
    createPopulation()

    for i in range(0, generations):
        populateFA(i)

        tournament()
    input("Press Enter to continue...")