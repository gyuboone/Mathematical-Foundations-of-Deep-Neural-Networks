import random

random.seed(17)

N = 3000
K = 600
p = 18/37

count_200 = 0

for times in range(N):

    games = 1
    money = 100
    while (games<=600 and money>0 and money <200):
        if random.random()<=p:
            money+=1
        else:
            money-=1

        games+=1

    if money==200:
        count_200+=1

print(count_200,'/ 3000')


q = 0.55
result = 0

for times in range(N):
    weight = 1
    games = 1
    money = 100
    while (games<=600 and money>0 and money <200):
        if random.random()<=q:
            money+=1
            weight*=p/q
        else:
            money-=1
            weight*=(1-p)/(1-q)

        games+=1

    if money>=200:
        result += weight

print(result/N)