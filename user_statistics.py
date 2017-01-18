#! /usr/bin/env python
import csv
import matplotlib.pyplot as plt

statistic = {2012:0, 2013:0, 2014:0, 2015:0, 2016:0, 2017:0}
with open("galaxy_stats.csv") as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    for row in file:
        statistic[int(row[2])] += int(row[1])

del statistic[2017]
list_years = []
list_jobs = []

for key in sorted(statistic.iterkeys()):
    list_years.append(key)
    list_jobs.append(statistic[key])
    # print "%s: %s" % (key, mydict[key])
print list_jobs
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

plt.bar(xrange(len(statistic)), list_jobs, align='center')
plt.xticks(xrange(len(statistic)), list_years)
plt.xlabel("Year")
plt.ylabel("Number of Jobs")
plt.title("Number of jobs per year")
plt.savefig("jobs_galaxy.svg", format="svg")
plt.savefig("jobs_galaxy.pdf", format="pdf")

plt.yscale('log')
plt.savefig("jobs_galaxy_log.svg", format="svg")
plt.savefig("jobs_galaxy_log.pdf", format="pdf")

