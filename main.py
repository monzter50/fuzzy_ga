
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import lib

tournament_percentage = 0.02
mutation_percentage = 0.90
populations = 400
fuzzy_network = 7
chromosome_size = fuzzy_network * 4
mutation = True
elit = False
weight = 5
main = [8,25,4,45,10,17,35] 
generations = 100
population =[]

fa = []

y_main = []
x_main = []
plot_distance_x = []
plot_distance_y = []
plot_graph_x = []

line1 = []
line2 = []
line3 = []


def create_populations():
    for i in range(0, populations):
        chromosome=random.sample(range(0,256), chromosome_size)
        population.append(chromosome)


def populate_fa(ei):
    fa.clear()
    global line1
    global line2
    global plot_graph_x
    global plot_graph_x

    plot_graph_x = []
    plot_graph_y = []
    sDistance = 999999999999999999999999

    for chromosome in population:
        fa_v = 0
        plot_graph_x_temp = []
        plot_graph_y_temp = []

        for i in range(0,1000):

            x = i/10
            y = lib.get_y_fuzzy(chromosome, x)
            
            plot_graph_x_temp.append(x)
            plot_graph_y_temp.append(y)
            
            fa_v += abs(yMain[i] - y)
        
        fa.append(fa_v)

        if sDistance > fa_v:
            sDistance = fa_v
            plot_graph_x = plot_graph_x_temp
            plot_graph_y = plot_graph_y_temp

    
    plot_distance_x.append(ei)
    plot_distance_y.append(sDistance)


    line1, line2 = plot(plot_distance_x,plotDistanceY, plot_graph_x,plot_graph_y,line1, line2, sDistance, chromosome)


def create_main_graph():
    for i in range(0,1000):
        x_main.append(i/10)
        y_main.append(lib.get_y_population(main[0],main[1],main[2],main[3],main[4],main[5],main[6], x_main[i]) )


def get_fa(po):
    po_fa = []
    for chromosome in po:
        fa_v = 0
        for i in range(0,1000):
            x = i/10
            y = lib.get_y_fuzzy(chromosome, x,fuzzy_network)
            # y =  getYpoint(chromosome[0]/weight,chromosome[1]/weight,chromosome[2]/weight,chromosome[3]/weight,chromosome[4]/weight,chromosome[5]/weight,chromosome[6]/weight, x)
            
            fa_v += abs(y_main[i] - y)
        
        po_fa.append(fa_v)
    return po_fa



def reproduction(f,m):
    cut_point = random.randint(0, chromosome_size * 8)
    cut_index = int(cut_point / 8)

    n = cut_point - (cut_index * 8)

    if(n == 0):
        fc = f[0:cut_index] + m[cut_index:chromosome_size]
        mc = m[0:cut_index] + f[cut_index:chromosome_size]
    else:
        low_mask = (2**n)-1
        hight_mask = ((2**chromosome_size)-1)-((2**n)-1)

        low_parcial_1 = f[cut_index] & low_mask
        low_parcial_2 = m[cut_index] & low_mask

        hight_parcial_1 = f[cut_index] & hight_mask
        hight_parcial_2 = m[cut_index] & hight_mask

        child_1 = low_parcial_1 | hight_parcial_2
        child_2 = low_parcial_2 | hight_parcial_1

        fc = f[0:cut_index] + [child_2] + m[cut_index+1:chromosome_size]
        mc = m[0:cut_index] + [child_1] + f[cut_index+1:chromosome_size]

    return fc,mc


def mutation(child_list):
    participants = random.sample(range(0,nPopulation), int(nPopulation * mutationPorcentage))
    
    for i in participants:
        cut_point = random.randint(0, (chromosome_size * 8) - 1)

        index = int(cut_point / 8)

        n = cut_point - (index * 8)

        mutate = child_list[i][index]

        bin_data = '{0:08b}'.format(mutate)

        bit_not = '1' if bin_data[n] == '0' else '0'
        new = bin_data[0:n] + bit_not + bin_data[n+1:8]
        
        new_int = int(new, 2)

        child_list[i][index] = new_int
    
    return child_list



def tournament():
    global population

    child_list = []
    for i in range(0, int(populations/2)):
        participants = random.sample(range(0,populations), int(populations * tournament_percentage))
        f = lib.better_options(participants)

        participants = random.sample(range(0,populations), int(populations * tournament_percentage))
        m = lib.better_options(participants)

        a,b = reproduction(population[f],population[m])

        child_list.append(a)
        child_list.append(b)
    
    mutation(child_list)
    mutation(child_list)

    if(elit):

        complete_list = population + child_list
        faList = fa + get_fa(child_list)

        list_index = {v: k for v, k in enumerate(faList)}
        sorted_list_index = list(dict(sorted(list_index.items(), key=lambda item: item[1])).items())
        population_index = sorted_list_index[0:populations]

        better_list = [v for v, k in population_index]

        population = [complete_list[v] for v in better_list]

        pass
    else:
        population = child_list


def plot(x_vec,y1_data, x_vec2, y1_data2,line1, line2,distances, identifier='',pause_time=0.0001):
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
        line3, = ax2.plot(x_main,y_main,'-o',alpha=0.8)   

    #update plot label/title
    plt.ylabel('Y Label')

    plt.title( str(distances) + ' - ' + str(identifier))
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
        ax2.set_ylim([np.min(y_main)-np.std(y_main),np.max(y_main)+np.std(y_main)])

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


def get_y_fuzzy_backup(data_fuzzy, x):

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
    
    create_main_graph()

    line1, line2 = plot([0],[0], [0],[0],line1, line2, 0)
    # time.sleep(3)
    
    create_populations()

    for i in range(0, generations):
        populate_fa(i)

        tournament()
    input("Press Enter to continue...")