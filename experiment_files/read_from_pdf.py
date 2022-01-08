#%%
import PyPDF4
import numpy as np
path = "/Volumes/Lexar/DataScience/Disease_Diagnosis_NLP_Project/experiment_files/Data/Ch13PDF.pdf"
ff = open( path , "rb")
pdfReader = PyPDF4.PdfFileReader(ff)
print(pdfReader.numPages)
total_page = pdfReader.numPages
#%%
list_from_pdf =[]
for page_num  in np.arange(0 , total_page):
    pageObj = pdfReader.getPage(page_num)
    list_from_pdf = list_from_pdf + [word.replace('\n', '') for word in pageObj.extractText().split(" ")]
print(list_from_pdf)
print(len(list_from_pdf))
#%%
title_list1 = ["Introduction", "EAR", 'ASSESSMENT', 'AUDIOMETRIC', 'NOSE', 'THROAT', 'OTITIS' ,  ]
title_list2 = [ 'Causes' , 'Complications' , 'Pathophysiology' , 'Signs' , 'Diagnosis' ]
str = ""
for index , word in enumerate(list_from_pdf):
    if word in title_list1:
        str = str + "@@@" + word + "***"
    elif word in title_list2:
        if word == "Causes" and list_from_pdf[index+1] == "and" and list_from_pdf[index+2] == 'Incidence':
            str = str + "@@@" + word + "***"
        elif word == "Signs" and list_from_pdf[index+1] == "and" and list_from_pdf[index+2] == 'Symptoms':
            str = str + "@@@" + word + "***"
        elif word == "Complications":
            try:
                if list_from_pdf[index+1][0].isupper():
                    str = str + "@@@" + word + "***"
            except:
                print("NO Complications")
        elif word == "Pathophysiology":
            try:
                if list_from_pdf[index+1][0].isupper():
                    str = str + "@@@" + word + "***"
            except:
                print('No Pathophysiology')
        elif word == "Diagnosis" :
            try:
                if list_from_pdf[index+1][0].isupper():
                    str = str + "@@@" + word + "***"
            except:
                print("No Diagnosis")
    else:
        str = str + word + " "
final_text_list = [word_list.split("***") for word_list in str.split("@@@")]
#%%
for num in np.arange(len(final_text_list)):
    print("==================")
    print(final_text_list[num])
ff.close()

#%%
signs_and_diagnosis =[]
for data in final_text_list:
    tem_list = []
    if data[0] == "Signs":
        signs_and_diagnosis.append(data)
    if data[0] == "Diagnosis":
        signs_and_diagnosis.append(data)
print(signs_and_diagnosis)
#%%
import pandas as pd
df_signs_and_diagnosis = pd.DataFrame(np.array([signs_and_diagnosis]))
# df_signs_and_diagnosis.to_csv("df_signs_and_diagnosis.csv" , index=False)

