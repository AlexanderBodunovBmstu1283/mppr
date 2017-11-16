import math
import time
import tread
import random



class Genetic():
    def __init__(self):
        self.Initial_population = 100
        self.Mating_population = int(self.Initial_population / 2)
        self.Favoured_population = int(self.Mating_population / 2)
        self.Number_cities = 10
        self.cut_length = self.Number_cities / 5
        self.Mutation_probability = 0.10

        self.loops = 1.5
        self.xlow = 0.0
        self.ylow = 0.0
        self.xhigh = 100.0
        self.yhigh = 100.0
        self.min_cost = 5.0
        #Thread_T = null
        self.costPerfect = 0.
        self.started = False
        #self.T=Null
        self.sities=[]
        self.chromosomes = []


    def start(self):
        self.cities=[]
        self.xrange = self.xhigh - self.xlow
        self.yrange = self.yhigh - self.ylow
        self.phi = random.randint(0,1) * 2. * math.pi
        self.dphi = self.loops * math.pi /self.Number_cities
        for i in range (self.Number_cities):
            r = 0.5 * self.xrange * (i + 1) / self.Number_cities
            self.phi += self.dphi
            xpos = self.xrange / 2 + r * math.cos(self.phi)
            ypos = self.yrange / 2 + r * math.sin(self.phi)
            ypos = self.yrange / 2 + 2 * i
            xpos = self.xrange / 2 + i * i / 7
            self.cities.append(City(xpos, ypos, self.xrange/2, self.yrange/2))

        Chromosome_perfect=Chromosome(self.Number_cities, self.cities)
        cl=[]
        for i in range(self.Number_cities):
            cl.append(i)
        Chromosome_perfect.set_cities(cl)
        Chromosome_perfect.calculate_cost(self.cities)
        print("Cost perfect: ",Chromosome_perfect.get_cost())
        for i in range(self.Number_cities):
            index1 = (int)(0.999999 * random.randint(0,1) * self.Number_cities)
            index2 = (int)(0.999999 * random.randint(0,1) * self.Number_cities)

            temp=self.cities[index2]
            self.cities[index2] = self.cities[index1]
            self.cities[index1] = temp

        self.chromosomes=[]
        for i in range(self.Initial_population):
            self.chromosomes.append(Chromosome(self.Initial_population,self.sities))
        for i in range(self.Number_cities):
            self.chromosomes[i] = Chromosome(self.Number_cities, self.cities)
            self.chromosomes[i].cut_length=self.cut_length
            self.chromosomes[i].Mutation_probability=self.Mutation_probability
        self.timestart=time.time()
        self.timend = self.timestart
        self.started=True

        #C.update(C.getGraphics());
        #// while (timend - timestart < 5000.0) {timend = System.currentTimeMillis();}
        self.Sort_chromosomes(self.Initial_population)
        self.Epoch = 0
        #if (self.T != Null):
            #self.T=Null
        #T=tread
        #T.setPriority(thread.MIN_PRIORITY)
        #T.start()





    def Sort_chromosomes(self,num):
        ctemp=Chromosome(self.Number_cities, self.cities)
        swapped = True
        while (swapped):
            swapped = False
            for i in range(num-1):
                if (self.chromosomes[i].get_cost() > self.chromosomes[i + 1].get_cost()):
                    ctemp = self.chromosomes[i]
                    self.chromosomes[i] = self.chromosomes[i + 1]
                    self.chromosomes[i + 1] = ctemp
                    swapped = True
            #print(1)

    def run(self):
        this_cost = 500.0
        old_cost = 0.0
        dcost = 500.0
        count_same = 0

        #C.update(C.getGraphics());
        while (this_cost > self.costPerfect and count_same < 100):
            self.Epoch+=1
            self.ioffset = self.Mating_population
            self.mutated = 0
            for i in range(self.Favoured_population):
                cmother = self.chromosomes[i]
                father = (int)(0.999999 * random.randint(0,1) * self.Mating_population)
                self.mutated += cmother.mate(father, self.chromosomes[self.ioffset], self.chromosomes[self.ioffset + 1])
                self.ioffset += 2
            for i in range(self.Mating_population):
                self.chromosomes[i] = self.chromosomes[i + self.Mating_population]
                self.chromosomes[i].calculate_cost(self.cities)
            self.Sort_chromosomes(self.Mating_population)
            cost = self.chromosomes[0].get_cost()
            dcost = math.modf(cost - this_cost)
            this_cost = cost
            mutation_rate = 100.0 * self.mutated / self.Mating_population
            print("Epoch ",self.Epoch," Cost ",int(this_cost)," Mutated ",mutation_rate," Count ",count_same)
            if (int(this_cost) == int(old_cost)):
                count_same+=1
            else:
                count_same = 0
                old_cost = this_cost
            #print(2)

            #C.update(C.getGraphics())
            print("A solution found after ",self.Epoch," epochs!")
            self.timestart=time.time()
            while (self.timend - self.timestart < 500000.0):
                self.timend=time.time()
                #print(3)
            self.start() ##################################

class City:
    def __init__(self,xpos, ypos, xrange, yrange):
        self.xpos=xpos
        self.ypos=ypos
        self.xrange=xrange
        self.yrange=yrange
    def proximity(self,cother):
        xdiff = self.xpos - cother.get_xpos()
        ydiff = self.ypos - cother.get_ypos()
        return math.sqrt(xdiff * xdiff + ydiff * ydiff)
    def proximity1(self,x,y):
        xdiff = self.xpos - x
        ydiff = self.ypos - y
        return math.sqrt(xdiff * xdiff + ydiff * ydiff)
    def proximity2(self):
        return self.distance_to_centre



class Chromosome:
    def __init__(self,sequence_length, cities):
        self.Mutation_probability = 0.10
        self.length = sequence_length
        self.cities=cities
        self.taken=[]
        self.city_list=[]
        #for i in range(self.length):
            #self.taken.append(i)
        city_list=[]
        for i in range(self.length):
            city_list.append(i)
        self.cost = 0.0
        for i in range(self.length):
            self.taken.append(random.choice([True, False]))
        for i in range(self.length):
            pass
            #print(self.taken[i])
        for i in range(self.length-1):
            icandidate=0
            while (self.taken[icandidate]):
                icandidate = int(0.999999 *random.random() * self.length )
                #print(4,icandidate,self.taken[icandidate])
            city_list[i] = icandidate
            self.taken[icandidate] = False#random.choice([True, False])
            if (i==self.length-2):
                icandidate = 0
                while (self.taken[icandidate]):
                    try:
                        a=self.taken[icandidate+1]
                        icandidate+=1
                    except:
                        break
                    #print(5)
                city_list[i + 1] = icandidate
        self.calculate_cost(self.cities)
        self.cut_length = 1

    def rand_bool(self):
        a=round(random.randint(0,1))
        if a==0:
            return False
        return True

    def calculate_cost(self,cities):
        try:
            self.cost = cities[self.city_list[0]].proximity()
            for i in range(self.length-1):
                dist = cities[self.city_list[i]].proximity(cities[self.city_list[i + 1]])
                self.cost += dist
        except:
            pass
            #print(len(cities),"uuuuuuuuuuuuuuuuu")
    def set_cities(self,list):
        for i in range (self.length):
            self.city_list.append(list[i])

    def _print(self):
        print("chromosome: ")
        for i in range(self.length):
            print(" ",self.city_list[i])
        print("\n")
    def check(self):
        self.taken=[]
        #for i in range(self.length):
            #taken.append
        for i in range(self.length):
            self.taken[i]=False
        for i in range(self.length):
            self.taken[self.city_list[i]] = True
        for i in range(self.length):
            if (not(self.taken[i])):
                print("Bad !")
                #print()
                return False
        return True
    def mate(self,father,offspring1,offspring2):
        cutpoint1 = int(0.999999 * random.randint(0,1) * self.length - self.cut_length)
        cutpoint2 = cutpoint1 + self.cut_length
        taken1=[]
        taken2=[]
        off1=[]
        off2=[]
        for i in range(self.length):
            taken1[i] = False
            taken2[i] = False
        for i in range(self.length):
            if (i < cutpoint1 or i >= cutpoint2):
                off1[i] = -1
                off2[i] = -1
            else:
                imother = self.city_list[i]
                ifather = father.sity[i]
                off1[i] = ifather
                off2[i] = imother
                taken1[ifather] = True
                taken2[imother] = True
        for i in range(cutpoint1):
            if (off1[i] == -1):
                for j in range(self.length):
                    imother = self.city_list[j]
                    if (not(taken1[imother])):
                        off1[i] = imother
                        taken1[imother] = True
                        break
            if (off2[i] == -1):
                for j in range(self.length):
                    ifather=father.sity[j]
                    if (not(taken2[ifather])):
                        off2[i] = ifather
                        taken2[ifather] = True
                        break
        for i in range(cutpoint2,self.length-1):
            if (off1[i] == -1):
                for j in range(self.length-1):
                    imother = self.city_list[j]
                    if (not(taken1[imother])):
                        off1[i] = imother
                        taken1[imother] = True
                        break
            if (off2[i] == -1):
                for j in range (0,self.length-1):
                    ifather = father.sity[j]
                    if (not(taken2[ifather])):
                        off2[i] = ifather
                        taken2[ifather] = True
                        break
        offspring1.set_cities(off1)
        offspring2.set_cities(off2)
        mutate = 0
        if (random.randint(0,1) < self.Mutation_probability):
            iswap1 = int(0.999999 * random.randint(0,1) * self.length)
            iswap2 = int(0.999999 * random.randint(0,1) * self.length)
            i = off1[iswap1]
            off1[iswap1] = off1[iswap2]
            off1[iswap2] = i
            mutate+=1
        if (random.randint(0,1) < self.Mutation_probability):
            iswap1 = int(0.999999 * random.randint(0,1) * self.length)
            iswap2 = int(0.999999 * random.randint(0,1) * self.length)
            i = off2[iswap1]
            off2[iswap1] = off1[iswap2]
            off2[iswap2] = i
            mutate+=1
        return mutate
    def oldmate(self,father,offspring1,offspring2):
        cutpoint1 = int(0.999999 * random.randint(0,1) * (self.length - self.cut_length))
        cutpoint2 = cutpoint1 + self.cut_length
        taken1=[]
        taken2=[]
        for i in range(self.length):
            imother = self.city_list[i]
            ifather = father.sity[i]
            icand = imother
            if (i >= cutpoint1 and i < cutpoint2):
                icand = ifather
            if (taken1[icand]):
                icand = 0
                while (taken1[icand]):
                    icand+=1
                    #print(6)
            offspring1.set_city(i, icand)
            taken1[icand] = True
            icand = ifather
            if (i >= cutpoint1 and i < cutpoint2):
                icand = imother
            if (taken2[icand]):
                icand = 0
                while (taken2[icand]):
                    icand+=1
                    #print(7)

            offspring2.set_city(i, icand)
            taken2[icand] = True
        mutate = 0
        if (random.randint(0,1) < self.Mutation_probability):
            iswap1 = int (0.999999 * random.randint(0,1) * self.length)
            iswap2 = int (0.999999 * random.randint(0,1) * self.length)
            i = offspring1.get_city(iswap1)
            offspring1.set_city(iswap1, offspring1.get_city(iswap2))
            offspring1.set_city(iswap2, i)
            mutate+=1
        if (random.randint(0,1) < self.Mutation_probability):
            iswap1 = (int) (0.999999 * random.randint(0,1) * self.length)
            iswap2 = (int) (0.999999 * random.randint(0,1) * self.length)
            i = offspring2.get_city(iswap1)
            offspring2.set_city(iswap1, offspring2.get_city(iswap2))
            offspring2.set_city(iswap2, i)
            mutate+=1
        return mutate

    def Cromosome(self):
        pass

    def set_city(self):
        pass
    def set_cut(fcut):
        pass
    def get_cost(self):
        return self.cost


a=Genetic()
a.start()