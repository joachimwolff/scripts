#!/usr/bin/env python
import psycopg2
import sys
import pprint
 
def main():
	conn_string = "host='localhost' dbname='galaxydb' user='postgres' password=''"
	# print the connection string we will use to connect
	# print "Connecting to database\n	->%s" % (conn_string)
 
	# get a connection, if a connect cannot be made an exception will be raised here
	conn = psycopg2.connect(conn_string)
 
	# conn.cursor will return a cursor object, you can use this cursor to perform queries
	cursor = conn.cursor()
 
	
	# get academia / insdustry users
	
	# cursor.execute("SELECT name FROM galaxy.galaxy_group")
    


	## General parameters
	# Registrations/active users
	cursor.execute("SELECT COUNT(id) FROM galaxy.galaxy_user")
	records = cursor.fetchall()
	print("Registered users: %i" % records[0])
	# Recurring users since 2017/01/01
	cursor.execute("SELECT COUNT(update_time) FROM galaxy.galaxy_user WHERE update_time BETWEEN '2017-01-01 00:00:00' AND NOW()")
	records = cursor.fetchall()
	print("Active users in 2017: %i" % records[0])	
	cursor.execute("SELECT COUNT(update_time) FROM galaxy.galaxy_user WHERE update_time BETWEEN '1977-01-01 00:00:00' AND '2015-12-31 24:00:00'")
	records = cursor.fetchall()
	print("Users last active in 2015: %i" % records[0])	

	cursor.execute("SELECT COUNT(update_time) FROM galaxy.galaxy_user WHERE update_time BETWEEN '2016-01-01 00:00:00' AND '2016-12-31 24:00:00'")
	records = cursor.fetchall()
	print("Number of users active in 2016: %i" % records[0])	
	# Academia / Industry
	list_of_academic_mail_adresses = ["uni","mpg", "dkfz", "charite", "edu", "ac.uk", ]
	list_of_commercial_mail = [ "gmail", "gmx", "yahoo", "hotmail", "googlemail"]

	cursor.execute("SELECT email FROM galaxy.galaxy_user")
	user_mail = cursor.fetchall()
	
	academic_count = 0
	commercial_mail_count = 0
	unknown_mail_count = 0
	for mail_adress in user_mail:
		# print mail_adress[0].split("@")[1]
		if not any(candidates in  mail_adress[0].split("@")[1] for candidates in list_of_academic_mail_adresses):
			academic_count += 1
		elif not any(candidates in  mail_adress[0].split("@")[1] for candidates in list_of_commercial_mail):
			commercial_mail_count += 1
		else:
			unknown_mail_count += 1
	print("Academic users: %i" % academic_count)
	print("Commercial mail adresses: %i" % commercial_mail_count)
	print("Unknown mail: %i" % unknown_mail_count)


	## Tools/Workflows for download (users take them away)
	# Number of downloads
	cursor.execute("SELECT COUNT(id) FROM galaxy.workflow_invocation") #maybe workflow_output?
	records = cursor.fetchall()
	print("Number of times all workflows were used: %i" % records[0])
	# Person hours

	## Applications/Pipelines/Workflows (users bring data/samples)
	# Number of projects/samples --> number of workflows
	cursor.execute("SELECT COUNT(id) FROM galaxy.workflow")
	records = cursor.fetchall()
	print("Number of workflows total: %i" % records[0])
	# Datavolume (Storage)
	# CPU hours
	# Person hours
	# Broker service (e.g. submission to archives) 

	## Webapplications (users use webpage)
	# Number of projects --> number of histories
	cursor.execute("SELECT COUNT(id) FROM galaxy.history")
	records = cursor.fetchall()
	print("Number of histories total: %i" % records[0])
	cursor.execute("SELECT COUNT(id) FROM galaxy.history WHERE update_time BETWEEN '2017-01-01 00:00:00' AND NOW()")
	records = cursor.fetchall()
	print("Number of histories in 2017: %i" % records[0])
	cursor.execute("SELECT COUNT(id) FROM galaxy.history WHERE update_time BETWEEN '2016-01-01 00:00:00' AND '2016-12-31 24:00:00'")
	records = cursor.fetchall()
	print("Number of histories in 2016: %i" % records[0])	
	# Number of jobs processed
	cursor.execute("SELECT COUNT(id) FROM galaxy.job")
	records = cursor.fetchall()
	print("Number of jobs total: %i" % records[0])
	cursor.execute("SELECT COUNT(id) FROM galaxy.job WHERE update_time BETWEEN '2017-01-01 00:00:00' AND NOW()")
	records = cursor.fetchall()
	print("Number of jobs in 2017: %i" % records[0])
	cursor.execute("SELECT COUNT(id) FROM galaxy.job WHERE update_time BETWEEN '2016-01-01 00:00:00' AND '2016-12-31 24:00:00'")
	records = cursor.fetchall()
	print("Number of jobs in 2016: %i" % records[0])	
	# Datavolume (Storage)
	cursor.execute("SELECT SUM(file_size) FROM galaxy.dataset")
	records = cursor.fetchall()
	print records[0][0]
	size = int(records[0][0]) / (2**30)
	print("Total storage used: %i GB" % size)
	# cursor.execute("SELECT SUM(total_size) FROM galaxy.dataset")
	# records = cursor.fetchall()
	# print("Total storage used: %i" % records[0])
	# print("Number of histories total: %i" % records[0])

	# CPU hours
	# cursor.execute("SELECT (metric_name, metric_value, job_id) FROM galaxy.job_metric_numeric JOIN galaxy.job ON galaxy.job_metric_numeric.job_id = galaxy.job.id WHERE galaxy.job.update_time BETWEEN '2016-01-08 12:00:00' AND '2016-01-08 24:00:00' ")
	# records = cursor.fetchall()
	# pprint.pprint(records)
	# Hits/visits/unique users
	# Broker service (e.g. submission to archives) 


	## Databases (users use database or take them away)
	# Number of primary resources parsed -> curated entries (mobilisation) (monitors the internal progress)
	# Number of downloads
	# Datavolume (Storage)
	# Webservices (Rest, Soap)
	# Hits/visits/unique users



 
if __name__ == "__main__":
	main()
