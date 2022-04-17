import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import mlflow 

from datetime import datetime as dt 

mlflow.set_tracking_uri("http://192.168.15.77:8080")
mlflow.set_experiment("clf-atp-pucpr")

def load_data():
    df = pd.read_csv("base_clean.csv")
    label, corpus = df['label'].tolist(),df['corpus'].tolist()
    corpus_treinamento, corpus_teste, rotulos_treinamento, rotulos_teste = train_test_split(corpus, label, test_size=0.10, random_state=42)
    return corpus_treinamento, corpus_teste, rotulos_treinamento, rotulos_teste
    

def train(corpus_treinamento, rotulos_treinamento):
    #sent_clf = Pipeline([('vect', CountVectorizer()),('clf', SVC(kernel='linear', C=1))])
    #sent_clf = Pipeline([('vect', CountVectorizer(lowercase=False)),('clf', SVC(kernel='poly', C=1))])
    #sent_clf = Pipeline([('vect', CountVectorizer(lowercase=False)),('clf', SVC(kernel='linear', C=1))])
    sent_clf = Pipeline([('vect', CountVectorizer()),("tfidf", TfidfTransformer()),("clf", SGDClassifier())])
     
    sent_clf = sent_clf.fit(corpus_treinamento, rotulos_treinamento)
    return sent_clf

def evalue_metrics(sent_clf,rotulos_teste,corpus_teste):
    rotulos_preditos = sent_clf.predict(corpus_teste)
    metrics = classification_report(rotulos_teste, rotulos_preditos,output_dict=True)
    generate_matrix(rotulos_teste, rotulos_preditos)
    
    return metrics

def generate_matrix(rotulos_teste, rotulos_preditos):
    # Podemos imprimir a matriz de confusão para tentar entender melhor os resultados
    mat = confusion_matrix(rotulos_teste, rotulos_preditos)

    rotulos_nomes = ['positivo', 'negativo', 'neutro']

    fig, ax = plt.subplots(figsize=(10,10)) 
    
    sns.heatmap(mat.T, 
                square=True, 
                annot=True, 
                fmt='d', 
                cbar=False, 
                xticklabels=rotulos_nomes, 
                yticklabels=rotulos_nomes )
    
    plt.xlabel('Categoria verdadeira')
    plt.ylabel('Categoria predita');
    img = 'output_{}.png'.format(str(dt.now()))
    print("Matriz de Confução : ",img)
    plt.savefig(img)

def main():
    with mlflow.start_run(run_name="pipe_clf_linear"):  
      mlflow.log_param("full", "pipeline_linear_clf")

      corpus_treinamento, corpus_teste, rotulos_treinamento, rotulos_teste = load_data()
      sent_clf = train(corpus_treinamento, rotulos_treinamento)
      metricas = evalue_metrics(sent_clf,rotulos_teste,corpus_teste)
      
      for k,v in metricas.items():
          try:
            for x,y in  v.items():
               metrica = {"{}_{}".format(k, x):y}
               mlflow.log_metric("{}_{}".format(k, x), float(y))      
          except AttributeError:  
            pass
      print(pd.DataFrame(metricas).T)  
      mlflow.sklearn.log_model(sent_clf,"model_linear")
    
      
        
if __name__=="__main__":
    main()