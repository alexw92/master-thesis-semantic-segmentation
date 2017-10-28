
import numpy as np
import tensorflow as tf
import random as ran
from datetime import datetime

# Example split date into month, year etc.
# dtobject = datetime.strptime('2008-09-26T01:51:42.000Z', date_format)
# access a la 'dtobject.hour', 'dtobject.second'

# ReImport module in interactive mode
# import imp
# imp.reload(module)

# Clicks data set (SessionID, Timestamp, ItemId, Category)
# Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer
# Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
# Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer
# Category – the context of the click. The value "S" indicates a special offer, "0" indicates  a missing value,
# a number between 1 to 12 indicates a real category identifier, any other number a special brand

# Buys data set
# Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer
# Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
# Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer
# Price – the price of the item. Could be represented as an integer number.
# Quantity – the quantity in this buying.  Could be represented as an integer number.


# X-Y Format
# Approach 1:
# X: Sequences of clicks (SessionID, Timestamp, ItemId, Category)...
# Y: Did he buy at least 1 item? Yes/No
#
# Raw data buys:
# '420374,2014-04-06T18:44:58.314Z,214537888,12462,1'
# '420374,2014-04-06T18:44:58.325Z,214537850,10471,1'
# '281626,2014-04-06T09:40:13.032Z,214535653,1883,1'
# '420368,2014-04-04T06:13:28.848Z,214530572,6073,1'
# '420368,2014-04-04T06:13:28.858Z,214835025,2617,1'
# '140806,2014-04-07T09:22:28.132Z,214668193,523,1'
# '140806,2014-04-07T09:22:28.176Z,214587399,1046,1'
# '140806,2014-04-07T09:22:28.268Z,214774667,1151,1'
# '140806,2014-04-07T09:22:28.280Z,214578823,1046,1'

# Raw data clicks
# 1,2014-04-07T10:51:09.277Z,214536502,0
# 1,2014-04-07T10:54:09.868Z,214536500,0
# 1,2014-04-07T10:54:46.998Z,214536506,0
# 1,2014-04-07T10:57:00.306Z,214577561,0
# 2,2014-04-07T13:56:37.614Z,214662742,0
# 2,2014-04-07T13:57:19.373Z,214662742,0
# 2,2014-04-07T13:58:37.446Z,214825110,0
# 2,2014-04-07T13:59:50.710Z,214757390,0
# 2,2014-04-07T14:00:38.247Z,214757407,0
# 2,2014-04-07T14:02:36.889Z,214551617,0
# 3,2014-04-02T13:17:46.940Z,214716935,0
# 3,2014-04-02T13:26:02.515Z,214774687,0
# 3,2014-04-02T13:30:12.318Z,214832672,0
# 4,2014-04-07T12:09:10.948Z,214836765,0
# 4,2014-04-07T12:26:25.416Z,214706482,0
# 6,2014-04-06T16:58:20.848Z,214701242,0
# 6,2014-04-06T17:02:26.976Z,214826623,0
# 7,2014-04-02T06:38:53.104Z,214826835,0
# 7,2014-04-02T06:39:05.854Z,214826715,0
# 8,2014-04-06T08:49:58.728Z,214838855,0
# read data lines
data_item_to_item = '../../ANN_DATA/RecSys15/clicks_item_to_item.txt'
data_write_mixed = '../../ANN_DATA/RecSys15/clicks_changed_items_mixed.txt'
data_write = '../../ANN_DATA/RecSys15/clicks_changed_items.txt'
data_clicks = '../../ANN_DATA/RecSys15/yoochoose-clicks.dat'
data_buys = '../../ANN_DATA/RecSys15/yoochoose-buys.dat'
date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
max_lines = 34000000  # found lines in click.data: 33003944 (in buy.data: 1150753)
# found different item ids: 52739 -> dim for 1-Hot feature encoding
# ascending labels could cause problems for training...should be mixed somehow

# creates new dataset with itemids mapped to numbers from 1 to ~50000
# in order to get a lower number of dimensions for nn processing


def read_clicks_and_buys():
    itemset = set()
    max = 0
    print('Reading buys data...')
    datalinesbuy = []
    with open(data_buys) as f:
        for i in range(max_lines):
            if i % 10000000 == 0:
                print(((i/max_lines)*100), '%')
            line = f.readline()
            if not line:
                break;
            datalinesbuy.append(line)
        print('100 %')
        datalinesbuy = [line.rstrip('\n') for line in datalinesbuy]

    print('Reading clicks data...')
    datalinesclick = []
    id_counter = 0
    itemdict = {}
    with open(data_clicks) as f:
        writefile = open(data_write, 'w')
        for i in range(max_lines):
            if i % 1000000 == 0:
                writefile.flush()
                print(((i/max_lines)*100), '%')
            line = f.readline()
            if not line:
                break;                   # 1,2014-04-07T10:51:09.277Z,214536502,0
            datalinesclick.append(line)
            split = line.split(',')
            item = split[-2]
            if not (item in itemdict):
                id_counter = id_counter + 1
                itemdict[item] = id_counter
            itemset.add(item)
            writefile.write(split[0]+','+split[1]+','+str(itemdict[split[2]])+','+split[3])
            nitem = int(item)
            if nitem > max:
                max = nitem
        print('100 %')
        writefile.close()
        # datalinesclick = [line.rstrip('\n') for line in datalinesclick]
        # print(f.readline())
        # print(f.readlines(20))
    print('data_buys ', len(datalinesbuy))
    print('data_clicks ', len(datalinesclick))
    return max, itemset,itemdict


def shuffleItemIds(itemdict, infile=data_clicks, outfile=data_write_mixed):
    """
    Load  previously written files and mix item id mapping in order to eliminate
    ascending order of ids
    :param infile: clickfile
    :param outfile: file with ascending item ids eliminated
    :return:
    """
    # Mix item id mapping
    print('Mixing item ids')
    ilist = list(itemdict.keys())
    print(len(ilist))
    for i in range(1, len(itemdict)):
        a = ran.randint(0, len(itemdict)-1)
        b = ran.randint(0, len(itemdict)-1)
        itemdict[ilist[a]], itemdict[ilist[b]] = itemdict[ilist[b]], itemdict[ilist[a]]
    with open(infile,'rt') as f:
        writef = open(outfile,'wt')
        for i in range(34000000):
            if i % 1000000 == 0:
                print(((i/max_lines)*100), '%')
            line = f.readline()
            if line==None:
                break
            split = line.split(sep=',')
            if len(split)<4:
                break;
            newline = split[0]+','+split[1]+','+str(itemdict[split[2]])+','+split[3]
            writef.write(newline)
        print('100 %')
        print('mixed item ids saved to '+outfile)
        writef.close()


def create_item_to_item_relation(infile=data_write_mixed, outfile=data_item_to_item):
    """
    :param infile: the file written byshuffleItemIds
    :param outfile: A file contaning a set of tuples (itemid1,itemid2) with itemid2 having been
            clicked immediately after itemid1 in a session.
    """
    lastsession = ''
    lastitem = ''
    line = ''
    with open(infile, 'rt') as read:
        write = open(outfile, 'wt')
        i = 0
        while line is not None:
            i = i+1
            if (i % 3400000) == 0:
                print(str(100*(i/max_lines))+' %')
            line = read.readline()
            split = line.split(',')
            if len(split) < 4:
                print('format error '+str(len(split))+' '+str(i)+' of '+str(max_lines)+' line '+line)
                break
            session = split[0]
            item = split[2]
            if lastsession != session:
                lastsession = session
                lastitem = item
                continue
            else:
                write.write(lastitem+','+item+'\n')
                lastsession = session
                lastitem = item
        write.close()


# max, itemset, itemdict = read_clicks_and_buys()
# shuffleItemIds(itemdict)
# create_item_to_item_relation()
