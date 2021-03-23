from simpletransformers.language_representation import RepresentationModel
from numpy import savetxt

def setence_representation(df,path):
    print('load SETENCE representation bert')
    model = RepresentationModel(model_type="bert",model_name="dbmdz/bert-base-italian-xxl-cased",use_cuda=False)
    setence_vectors = model.encode_sentences(df['full_text'], combine_strategy="mean")
    #word_vectors = model.encode_sentences(test_document_text, combine_strategy=None)
    savetxt(path + '/setence_repr_bert.csv',setence_vectors, delimiter=',')
    return  setence_vectors

def word_representation(df,path):
    print('load WORD representation bert')
    model = RepresentationModel(model_type="bert",model_name="dbmdz/bert-base-italian-xxl-cased",use_cuda=False)
    word_vectors = model.encode_sentences(df['full_text'], combine_strategy=None)
    savetxt(path + '/word_repr_bert.csv', word_vectors, delimiter=',')
    return word_vectors