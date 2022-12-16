import itertools

l = ['a', 'b', 'c', 'd',"e"]

c = itertools.combinations(l, 2)

print(type(c))
# <class 'itertools.combinations'>

c_list = []
for i in range(len(l)):
    c = itertools.combinations(l, i+1)
    for v in c:
        c_list += ["".join(v)]
print(len(c_list))
print(",".join(c_list))
