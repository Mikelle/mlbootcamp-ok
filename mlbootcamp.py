import csv, os
import numpy as np
from scipy.sparse import csr_matrix

dataPath = "data/"
graphPath = os.path.join(dataPath, "graph")
demographyPath = os.path.join(dataPath, "trainDemography")
testUsersPath = os.path.join(dataPath, "users")
resultPath = "result/"

maxTotal = 47289241
linksCount = 27261623

testUsers = set()
for line in csv.reader(open(testUsersPath)):
    testUsers.add(int(line[0]))
print("loaded test Users")


def load_csr(path):
    loaded = np.load(path + ".npz")
    return csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])


testGraph = load_csr(os.path.join(resultPath, "testGraph"))
print("loaded testGraph")
birthDates = np.load(os.path.join(resultPath, "birthDates.npy"))
print("loaded dates")

with open(os.path.join(resultPath, "prediction.csv"), "w") as output:
    writer = csv.writer(output, delimiter=',')

    for user in testUsers:
        ptr = testGraph.indptr[user - 1]
        ptrNext = testGraph.indptr[user]

        friendsDates = np.fromiter(map(lambda x: birthDates[x], testGraph.indices[ptr:ptrNext]), dtype=np.int)

        meanDate = np.mean(friendsDates)

        schoolmatesAgeList = []
        collegeAgeList = []
        closeFriendsList = []

        for i in range(ptr, ptrNext):
            mask = testGraph.data[i]
            age = birthDates[testGraph.indices[i]]
            if mask & (1 << 8):
                closeFriendsList.append(age)
            if mask & (1 << 10):
                schoolmatesAgeList.append(age)
            if mask & (1 << 14):
                collegeAgeList.append(age)

        dates = [float(meanDate)]

        if len(collegeAgeList) > 0:
            dates.append(float(np.median(collegeAgeList)))

        if len(schoolmatesAgeList) > 0:
            dates.append(float(np.median(schoolmatesAgeList)))

        if len(closeFriendsList) > 0:
            dates.append(float(np.median(closeFriendsList)))

        date = np.median(dates)

        writer.writerow([user, date])