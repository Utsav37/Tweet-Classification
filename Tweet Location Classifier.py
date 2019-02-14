#!/usr/bin/env python3
#LOGIC: Overall logic is simple implementation of Naive Bayes and smartly using stopwords. 
# P(label/Words)  proportional to P(Word1/label) *P(Word2/label)* P(Word3/label) * P(label)
# Work done by @Utsav 
# CITATION:
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://thispointer.com/python-how-to-find-keys-by-value-in-dictionary/
# Output is like below: 
# CITY                   TOP 5 WORDS
# San_Francisco,_CA         ['CA', '#SanFrancisco,', 'San', 'Francisco,', '#CareerArc']
# San_Diego,_CA         ['San', 'Diego,', 'CA', 'CA)', '#SanDiego,']
# Washington,_DC         ['DC)', 'DC', 'Washington,', '#Washington,', '#CareerArc']
# Toronto,_Ontario         ['ON', 'b/w', 'Trucks)', 'Toronto', 'Toronto,']
# Atlanta,_GA         ['#Atlanta,', 'GA', '#CareerArc', 'Atlanta,', 'Georgia']
# Manhattan,_NY         ['NY', 'New', 'York,', 'NY)', 'York']
# Philadelphia,_PA         ['Philadelphia,', 'PA', 'Philadelphia', '#Philadelphia,', 'PA)']
# Boston,_MA         ['#Boston,', '#CareerArc', 'MA', 'Boston,', 'Boston']
# Chicago,_IL         ['Chicago,', 'IL)', '#Chicago,', 'IL', 'Chicago']
# Houston,_TX         ['Houston', '#Houston,', 'TX', 'Houston,', '#CareerArc']
# Los_Angeles,_CA         ['Los', 'Angeles,', '#LosAngeles,', 'CA', '#CareerArc']
# Orlando,_FL         ['#orlpol', '#opd', 'FL', 'S', '#Orlando,']
# length of y original is :  32000
# correctly classified are :  31367
# length of y original is :  500
# correctly classified are :  324
# Training Accuracy is :  0.98021875
# Testing_accuracy is :  0.648
import sys
train=sys.argv[1]
test=sys.argv[2]
output=sys.argv[3]

def remove_non_ascii_1(text):
	smallline=[]
	for i in text :
		if(ord(i)<128):
			smallline.append(i)
		else:
			smallline.append("")
	return smallline
stopwords=set(['#Job', '#job','with','#Jobs','#jobs','&amp;','&amp','on','I\'m','The','our','a','after','all','am','an','and','any','are','as','at','because','before','being','below','between','both','but','by','can\'t','could','did','didn\'t','do','doesn\'t'
,'doing','don\'t','down','during','each','few','for','further','hadn\'t','hasn\'t','have','having','here','hers','herself','himself','i\'ll'
,'i\'ve','if','in','into','is','it','its','itself','let\'s','more','my','nor','off','only','ought','ourselves','own','she','she\'s','so','than','the','them','then'
,'there','there\'s','these','they','this','through','under','up','very','wasn\'t','we\'ll','were','what\'s','when\'s','where','where\'s','which'
,'while','who','who\'s','whom','would','you\'d','you\'ve','__','(@','(2','be','St','from','This', 'We\'re','request','great','#job?','report'
,'out','opened','via','See','___','____','that','opened','Opened','St.','here:','opening','latest','#Hiring','/','yourself','yourselves','about','above','aren\'t','again','against','been',':','!','@','#','$','%','^'
,'&','*','(',')','_','+','|','~','`','=','-','{','}','[',']','?','>','<','.','"',';','\'','in','at','the','to','a','and','of','for','I','you'
,'work', 'Click','me', 'Can', 'anyone', 'apply:','was','Want','#job','#Job:','____','______'])

# cleandata does the work of returning list cleanlevel1data that has list of tweets (which infact is list of words) which is fully cleaned , 
# which does not contain stopwords, nor does it contain any type of blank spaces
# It also returns citytweetcount which contains total count of tweets for each cities : dictionary with key as cityname and value as count of tweets
# cityset contains list of unique cities: It is a list of 12 cities. 
def cleandata(cleanlines):

	cityjunk=[]
	for i in range(0,len(cleanlines)):
		cityjunk.append(cleanlines[i][0])
	# print(cityjunk)
	sub=",_"
	citysets=[s for s in cityjunk if sub in s]

	citytweetcount={}
	cityset=set(citysets)
	for i in cityset:
		citytweetcount[i]=0

	cleanlevel1data=[]
	for i in range(0,len(cleanlines)):
		if(cleanlines[i][0] in cityset):
			citytweetcount[cleanlines[i][0]]+=1
			cleanlevel1data.append(cleanlines[i].copy())
			toberemoved=[]
			for s in range(0,len(cleanlines[i])):
				if(cleanlines[i][s] in stopwords):
					toberemoved.append(cleanlines[i][s])
			if(len(toberemoved)>0):
				cleanlevel1data[-1]=[x for x in cleanlevel1data[-1] if x not in toberemoved]
		else:
			data=cleanlines[i]
			toberemoved=[]
			for s in range(0,len(cleanlines[i])):
				if(cleanlines[i][s] in stopwords):
					toberemoved.append(cleanlines[i][s])
			if(len(toberemoved)>0):
				cleanlevel1data[-1].append(x for x in data if x not in toberemoved)

	return(cleanlevel1data,citytweetcount,cityset)


# FOR classifytweets:
# cleanlines received here as argument simply does not contain NON ASCII characters and no blank spaces. It is a list of tweets.
# Returned parameters: 

# dictoftweets2 is dictionary with key as city and further the value of that key is
 # itself  a dictionary which has word as key and value as probability of that word in all tweets for a particular city
 # eg: {'Chicago':{'#DeepPizza':0.00450,'ChicagoBulls':0.00341,....},'San_Fransisco':{'Word1':prob1,'Word2':prob2,......},''}
 #  cityset contains list of unique cities: It is a list of 12 cities.
 # cleanlevel1data  has list of tweets (which infact is list of words) which is fully cleaned , 
# which does not contain stopwords, nor does it contain any type of blank spaces
# citytweetcount returned contains total probability of tweets for each cities : dictionary with key as cityname and value as probability of tweets


def classifytweets(cleanlines):
	cleanlevel1data,citytweetcount,cityset=cleandata(cleanlines)          
	totalcitydata=0
	for key,value in citytweetcount.items() :
		totalcitydata=totalcitydata+value
	for key,value in citytweetcount.items() :
		citytweetcount[key]=value/totalcitydata
# Initializing dictionaries that are required in latter part of the program
	dictoftweets={}
	totalwords={}
	# max5citywords={}
	for i in cityset:
		dictoftweets[i]={}
		totalwords[i]=0
		# max5citywords[i]={}

	for i,line in enumerate(cleanlevel1data):
		cityname=line[0]
		tweet=line[1:] ########
		for word in tweet:
			# print("dictoftweets[cityname] is :    ",dictoftweets[cityname].keys())
			if word not in dictoftweets[cityname].keys():
				dictoftweets[cityname][word]=1
			else:
				dictoftweets[cityname][word]+=1
	values=0
	# for key,value in dictoftweets['San_Francisco,_CA'].items() :
		# values=values+value
	# print(dictoftweets,"\n\n\n\n\n\n\n\n\n\n\n")
	dictoftweets2=dictoftweets.copy()
	for cityname in cityset:
		for key,value in dictoftweets[cityname].items():
			totalwords[cityname]=totalwords[cityname]+value
	for cityname in cityset:	
		for key,value in dictoftweets[cityname].items():
			# print(value/totalwords[cityname])
			# print(dictoftweets2[cityname][key])
			dictoftweets2[cityname][key]= (value/totalwords[cityname])
			# print(dictoftweets2[cityname][key])
	# below code for test to be mended
	
	max5citywords= {}
	for cityname in cityset:
		cityvaluelist=list(dictoftweets2[cityname].values())
		listOfValues=[]
		for i in range(0,5):
			# print(i)
			tempmax=max(cityvaluelist)
			listOfValues.append(tempmax)
			cityvaluelist.remove(tempmax)
		listOfKeys = list()
		# listOfValues=templist
		# listOfItems = dictoftweets2[cityname].items()
		for key,val  in dictoftweets2[cityname].items():
			if val in listOfValues:
				listOfKeys.append(key)
		max5citywords[cityname]=listOfKeys
	print("CITY                   TOP 5 WORDS ")
	for key,value in max5citywords.items():
		print(key,"       ",value)
	return (cityset,cleanlevel1data,citytweetcount,dictoftweets2)


# returns accuracy and predicted y 
def getaccuracy(cityset,cleanlevel1data,citytweetcount,dictoftweets2):
	yoriginal=[]
	ypred=[]
	probadict={}
	for i in cityset:
		probadict[i]=1
	# print(probadict)
	productofall=1
	# print("DICT OF TWEETS 2 KEYS : ",dictoftweets2)
	import operator
	# if(datafrom=="test"):
		# for line in cleanlevel1data:
			# print("TEST DATA IS HERE: ",cleanlevel1data[line])
	for tweet in cleanlevel1data:
		# if(datafrom=="test"):
			# print(tweet)
		yoriginal.append(tweet[0])
		ytweet=tweet[1:]
		# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",ytweet)
		for city in cityset:
			# if(datafrom=="test"):
				# print("CITYTWEETCOUNT IS ",citytweetcount)
			productofall=citytweetcount[city]
			# print("PRODUCT OF ALL IS : ",productofall)
			for word in ytweet:
				if(word in dictoftweets2[city].keys()):
					productofall=productofall*dictoftweets2[city][word]
					# print("PRODUCT OF ALL IS : ",productofall)
				else:
					productofall=productofall*float(pow(10,-6))
					# print("PRODUCT :",productofall)
			probadict[city]=productofall
			# print("PROBADICT OF CITY IS : ",probadict[city])
		max_val=max(probadict.values())
			# print("PROBA DICT ITEMS IS : ",probadict.items())
			# print("MAX VAL IS :     ",max_val)
		for k,v in probadict.items():
			if v==max_val:
				max_key=k
		ypred.append(max_key)
	# print(ypred)
	count=0
	print("length of y original is : ",len(yoriginal))

	for i in range(len(yoriginal)):
		# if(len(yoriginal)==len(ypred)):
			# print("PARTYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYy")
		if(yoriginal[i]==ypred[i]):
			count+=1
	print("correctly classified are : ",count)
	accuracy=count/len(yoriginal)
		
	return (accuracy,ypred)



trainlines=[]
traincleanlines=[]
with open(train, "r", encoding = "ISO-8859-1") as file:
    for line in file.readlines():
    	trainlines.append(''.join(list(remove_non_ascii_1(line))).split())
traincleanlines= [x for x in trainlines if x != []]
(cityset,cleanlevel1data,citytweetcount,dictoftweets2)=classifytweets(traincleanlines)
training_accuracy,ypred=getaccuracy(cityset,cleanlevel1data,citytweetcount,dictoftweets2)


testlines=[]
testcleanlines=[]
with open(test, "r", encoding = "ISO-8859-1") as file:
    for line in file.readlines():
    	testlines.append(''.join(list(remove_non_ascii_1(line))).split())
testcleanlines= [x for x in testlines if x != []]
cleanlevel1data,citytweetcount,cityset=cleandata(testcleanlines)

testing_accuracy,ypred=getaccuracy(cityset,cleanlevel1data,citytweetcount,dictoftweets2)
print("Training Accuracy is : ",training_accuracy)
print("Testing_accuracy is : ",testing_accuracy)
fileobj = open(output,'w')
for i,line in enumerate(cleanlevel1data):
	finallist=[]
	templist=line
	templist.insert(0, ypred[i])
	fileobj.write(" ".join(str(x) for x in templist))
	fileobj.write("\n")
fileobj.close()
