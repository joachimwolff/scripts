#!/usr/bin/env python
import psycopg2
import sys
import pprint
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

import plotly
from plotly.graph_objs import Scatter, Layout
import xlsxwriter

def get_active_users(pCursor, pDate1, pDate2):
    """Date needs to be in the format: YY-MM-DD HH:MM:SS e.g. 2012-01-01 00:00:00"""
    query =  "SELECT COUNT( DISTINCT galaxy.job.user_id) FROM galaxy.job \
            INNER JOIN galaxy.galaxy_user \
             ON galaxy.galaxy_user.id = galaxy.job.user_id \
             WHERE galaxy.job.update_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    # print query
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]
def get_recurring_users(pCursor, pDate1, pDate2):
    '''A recurring is someone who runs at least once a month a job.'''
    query =  "SELECT COUNT( DISTINCT galaxy.job.user_id) FROM galaxy.job \
            INNER JOIN galaxy.galaxy_user \
             ON galaxy.galaxy_user.id = galaxy.job.user_id \
             WHERE galaxy.job.update_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    # print query
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]

def get_jobs(pCursor, pDate1, pDate2):
    query = "SELECT COUNT(id) FROM galaxy.job WHERE update_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]

def get_number_of_histories(pCursor, pDate1, pDate2):
    query = "SELECT COUNT(id) FROM galaxy.history WHERE update_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]

def get_registered_users(pCursor):
    query = "SELECT COUNT(id) FROM galaxy.galaxy_user WHERE deleted = False"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]
def get_registered_users_at_date(pCursor, pDate):
    query = "SELECT COUNT(id) FROM galaxy.galaxy_user \
                WHERE galaxy.galaxy_user.deleted = False \
                AND galaxy.galaxy_user.create_time BETWEEN '1977-01-01 00:00:00' AND '"+pDate + "'"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]
def get_registered_users_since_date(pCursor, pDate1, pDate2):
    query = "SELECT COUNT(id) FROM galaxy.galaxy_user \
                WHERE galaxy.galaxy_user.deleted = False \
                AND galaxy.galaxy_user.create_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]
def get_number_of_workflows(pCursor):
    query = "SELECT COUNT(id) FROM galaxy.workflow"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]

def get_number_of_workflows_usage(pCursor, pDate1, pDate2):
    query = "SELECT COUNT(id) FROM galaxy.workflow_invocation WHERE galaxy.workflow_invocation.update_time BETWEEN '"+ pDate1 +"' AND '"+pDate2+"'"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records[0][0]

def get_storage_usage_total(pCursor):
    pCursor.execute("SELECT SUM(file_size) FROM galaxy.dataset WHERE deleted = False")
    records = pCursor.fetchall()
    size = int(records[0][0]) / (2**40)
    return size

def get_storage_usage_at_date(pCursor, pDate):
    pCursor.execute("SELECT SUM(file_size) FROM galaxy.dataset WHERE deleted = False AND galaxy.dataset.update_time BETWEEN '1977-01-01 00:00:00' AND '"+pDate + "'")
    records = pCursor.fetchall()
    if records[0][0] != None:
        size = float(records[0][0]) / float(2**40)
        return size
    return 0
def get_jobs_start_time(pCursor):
    query = "SELECT galaxy.job.create_time FROM galaxy.job"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records

def get_jobs_finish_time(pCursor):
    query = "SELECT galaxy.job.create_time FROM galaxy.job"
    pCursor.execute(query)
    records = pCursor.fetchall()
    return records

def plot_plotly(pX, pY, pTitle, pYaxisTitle, pFileName, pX2=None, pY2=None, pYaxisTitle2=None, pMode=None, pMode2=None, pFigureTitle=None):
    if pMode is None:
        pMode = 'lines+markers'
    if pFigureTitle is None:
        pFigureTitle = 'Month / Year'
    trace0 = Scatter(x=pX, y=pY, mode=pMode)
    data = [trace0]
    if pX2 is not None:
        trace1 = Scatter(x=pX2, y=pY2, mode=pMode2)
        data.append(trace1)
    plotly.offline.plot({
    "data": data,
    "layout": Layout(title=pTitle, xaxis = dict(title= pFigureTitle,
                                    ticklen= 5,
                                    zeroline= False,
                                    gridwidth= 2,
                                ),
                                yaxis=dict(
                                    title= pYaxisTitle,
                                    ticklen= 5,
                                    gridwidth= 2,
                                )
                    )
    }, filename='images/'+pFileName+'_'+pTitle+'.html', auto_open=False)
    return 'images/'+pFileName+'_'+pTitle+'.html'

def create_html(pFile_names, pHeaders, pDiv, pReportTitle):
    html_string = '''
    <html> 
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <style>body{ margin:0 100; background:whitesmoke; } </style>
        </head>
        <body>
            <h1>Galaxy Freiburg Statistics Report</h1>'''

            

    for image, header, div in zip(pFile_names, pHeaders, pDiv):
        html_string += '''<h2>'''+header + '''</h2>'''
        
        html_string += '''<iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
                            src="''' + image + '''?width=800&height=550"></iframe>'''
        html_string += '''<div>'''+div + '''</div>'''
        
    html_string += ''' </body> \
                        </html>'''
       
    f = open(pReportTitle + '.html','w')
    f.write(html_string)
    f.close()
def write_month_statistic_to_excel(pFile, pRow, pCol, pData, pHeader):
    pFile.write(pRow, pCol, pHeader)
    pRow += 1
    for data, value in pData:
        pFile.write(pRow, pCol, data)
        pFile.write(pRow+1, pCol, value)
        pCol += 1

def create_report(pCursor, pStartDate, pEndDate, pReportTitle, pExcel=False, pExcelTitle=""):
    date_start = pd.date_range(start=pStartDate, end=pEndDate, freq='MS')
    date_end = pd.date_range(start=pStartDate, end=pEndDate, freq='M')

    registered_users = get_registered_users(pCursor)
    number_of_workflows = get_number_of_workflows(pCursor)
    storage = get_storage_usage_total(pCursor)

    active_users_per_month = []
    jobs_per_month = []
    number_of_histories_per_month = []
    registered_users_in_year_per_month = []
    storage_usage_per_month = []
    number_of_workflows_usage_per_month = []

    for i in xrange(len(date_start) - 1):
        active_users_per_month.append(get_active_users(pCursor, str(date_start[i]), str(date_end[i])))
        jobs_per_month.append(get_jobs(pCursor, str(date_start[i]), str(date_end[i])))
        number_of_histories_per_month.append(get_number_of_histories(pCursor, str(date_start[i]), str(date_end[i])))
        registered_users_in_year_per_month.append(get_registered_users_at_date(pCursor, str(date_end[i])))
        storage_usage_per_month.append(get_storage_usage_at_date(pCursor, str(date_end[i])))
        number_of_workflows_usage_per_month.append(get_number_of_workflows_usage(pCursor, str(date_start[i]), str(date_end[i])))
    
    

    # start_data = get_jobs_start_time(pCursor)
    # # print start_data
    # for data in start_data:
    #     data = data[0].time()
    # finish_data = get_jobs_finish_time(pCursor)
    # for data in finish_data:
    #     print data[0].time()
    #     data = data[0].time()
    #     print data
    dates_for_plotting = date_end.format()
    # hours = pd.date_range(start='00:00:00', end='23:59:59', freq='H')
    filenames = []
    filenames.append(plot_plotly(dates_for_plotting, active_users_per_month, "Active users per month", "Unique users", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, jobs_per_month, "Jobs per month", "Unique jobs", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, number_of_histories_per_month, "Number of histories per month" , "Number of histories", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, storage_usage_per_month, "Storage usage per month", "Used storage in TB", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, number_of_workflows_usage_per_month, "Used workflows per month", "Used workflows", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, registered_users_in_year_per_month, "Registered users per month", "Total registered users", pReportTitle))
    filenames.append(plot_plotly(dates_for_plotting, registered_users_in_year_per_month, "Registered users per month", "Total registered users", pReportTitle))
    # filenames.append(plot_plotly(hours.format(), start_data, "Times of start and finish of a job", "Time start job", pX2=hours.format(), pY2=finish_data, pYaxisTitle2="Time end job", pMode="markers", pMode2="markers", pFigureTitle="Hours"))

    headers = ["Active users per month", "Jobs per month", "Used histories", "Development of storage usage", "Workflows", "Development of registered users"]
    div = ["An unique active user in a month is someone who runs a job a least once.", "", "", "Storage usage", "Number of workflow runs in a month.", "", ""]
    create_html(filenames, headers, div, pReportTitle)

    if pExcel:
        workbook = xlsxwriter.Workbook(pExcelTitle + ".xlsx")
        worksheet = workbook.add_worksheet()

        active_users_excel = zip(dates_for_plotting, active_users_per_month)
        jobs_per_month_excel = zip(dates_for_plotting, jobs_per_month)
        number_of_histories_per_month_excel = zip(dates_for_plotting, number_of_histories_per_month)
        storage_usage_per_month_excel = zip(dates_for_plotting, storage_usage_per_month)
        number_of_workflows_usage_per_month_excel = zip(dates_for_plotting, number_of_workflows_usage_per_month)
        registered_users_in_year_per_month_excel = zip(dates_for_plotting, registered_users_in_year_per_month)

        worksheet.write(0, 0, 'Registrations')
        worksheet.write(0, 1, registered_users)
        worksheet.write(1, 0, "Number of workflows")      
        worksheet.write(1, 1, number_of_workflows)
        worksheet.write(2, 0, "Total used storage in TB")
        worksheet.write(2, 1, storage)
        
        worksheet.write(4, 0, "Statistics per month")

        write_month_statistic_to_excel(worksheet, pRow=6, pCol=0, pData=active_users_excel, pHeader=headers[0])
        write_month_statistic_to_excel(worksheet, pRow=9, pCol=0, pData=jobs_per_month_excel, pHeader=headers[1])
        write_month_statistic_to_excel(worksheet, pRow=12, pCol=0, pData=number_of_histories_per_month_excel, pHeader=headers[2])
        write_month_statistic_to_excel(worksheet, pRow=15, pCol=0, pData=storage_usage_per_month_excel, pHeader=headers[3])
        write_month_statistic_to_excel(worksheet, pRow=18, pCol=0, pData=number_of_workflows_usage_per_month_excel, pHeader=headers[4])
        write_month_statistic_to_excel(worksheet, pRow=21, pCol=0, pData=registered_users_in_year_per_month_excel, pHeader=headers[5])
        
        
        workbook.close()


def main():
    if not os.path.exists("images"):
        os.makedirs("images")
    # connection string to database
    conn_string = "host='localhost' dbname='galaxydb' user='postgres' password=''"
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)
    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    
    create_report(cursor, pd.datetime(2011, 1, 1), datetime.date.today() , "Galaxy Freiburg Usage Statistics", pExcel=False, pExcelTitle="")
    # for denbi
    create_report(cursor, pd.datetime(2015, 3, 1), datetime.date.today(), "Galaxy Freiburg Usage Statistics for de.NBI", pExcel=True, pExcelTitle="Galaxy Freiburg Usage Statistics for de.NBI")
    
    

    
    
if __name__ == "__main__":
    main()
