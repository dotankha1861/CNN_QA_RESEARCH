import pyodbc
from predict_label import predict_final
from similarity import tfidf_cosineSimilarity
from import_data_sql import connstr

num_class = 2

if __name__ == '__main__':
 
    conx = pyodbc.connect(connstr)
    cursor = conx.cursor()
    
    while True:

        question = input()
        question, label = predict_final(question, num_class)
        
        if len(question) == 0:
            print("Hệ thống không hiểu câu hỏi, vui lòng nhập câu hỏi rõ ràng hơn!")
            continue

        cursor.execute("select text from data where label2 = '" + label + "'")
        data = cursor.fetchall()

        answer = tfidf_cosineSimilarity(question,[row[0] for row in data])

        for row in answer:
            print("* ",row)

