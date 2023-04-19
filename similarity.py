import os
import math
from string import punctuation
from underthesea import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer

n_gram = 3
min_df = 2
epsilon_sim = 0.05

f = open(os.path.join('Data','stop_words.txt'), 'r', encoding='utf-8')
data_stop_words = f.read()
stop_words = data_stop_words.split('\n') 

# tích vô hướng 2 vector
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2) )

# độ dài vector 
def length(v):
     return math.sqrt(dotproduct(v, v))

# độ tương đồng 2 vector 
def cosineSimilarity(v1, v2):
    if length(v1) == 0 or length(v2) == 0:
        return 0
    return dotproduct(v1, v2) / (length(v1) * length(v2))

# xử lý văn bản trước khi vector hóa bằng tf-idf
def user_defined_preprocessor(document):
    document = document.translate(str.maketrans('', '', punctuation))
    token_words = word_tokenize(document.lower())
    token_words = [word for word in token_words if word not in stop_words]
    return ','.join(token_words) 

# token các từ tạo n_gram
def user_defined_tokenizer(document):
    return document.lower().split(',')

# tính độ tương đồng văn bản giữa orgDoc với các otherDocs bằng độ đo Cosine sau khi vector hóa bằng tf-idf
def tfidf_cosineSimilarity(orgDoc, otherDocs):

    # tạo tập văn bản
    list_documents = [orgDoc]
    list_documents.extend(otherDocs)
    
    # vector hóa các văn bản bằng tf-idf với tfidfVectorizer    
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=True, ngram_range=(1,n_gram), min_df = min_df,
                                       preprocessor=user_defined_preprocessor, tokenizer=user_defined_tokenizer)  
    document_vectors = tfidf_vectorizer.fit_transform(list_documents).toarray()

    # giảm số chiều vector: chỉ tính độ tương đồng với những từ xuất hiện ở văn bản hỏi   
    orgDoc_vector = [x for x in document_vectors[0] if x!= 0]
    otherDocs_vector = []

    for vector in document_vectors[1:]:
        doc_vector = []
        for index in range(len(document_vectors[0])):
            if(document_vectors[0][index] != 0):
                doc_vector.append(vector[index])
        otherDocs_vector.append(doc_vector)

    # tính độ tương đồng giữa văn bản hỏi với các văn bản trả lời
    list_cosSim = [(cosineSimilarity(orgDoc_vector,doc_vector)) for doc_vector in otherDocs_vector]
   
    # chọn ra văn bản trả lời có độ tương đồng gần với văn bản hỏi.
    # nếu có nhiều vector, chọn ra 3
    max_sim = max(list_cosSim)
    index_max = []
    for index in range(len(list_cosSim)):
        if list_cosSim[index] >= max_sim - epsilon_sim:
            index_max.append(index)
            if len(index_max) == 3:
                break
   
    return [otherDocs[i] for i in index_max]

if __name__ == '__main__':
    print(user_defined_tokenizer(user_defined_preprocessor('Mục đích của trí tuệ nhân tạo là gì?')))