import DataCreateHelper
import ModelsHelper

def example_cnnlstm():

def example_vanillialstm():

    traincount = 10000
    evaluatecount = 200
    n_patterns=1
    length = 5
    n_feature = 10
    out_index = 3
    LSTMmemorycount = 25

    X_train, y_train, X_predict, y_preidct = mydatacreate.preparedata(traincount, evaluatecount,
                                                                      mydatacreate.generateexample_vanillaLSTM,
                                                                      n_patterns=n_patterns,length=length, n_features=n_feature,
                                                                      out_index=out_index)

    showsequence = mydatacreate.decode_vanillaLSTM(X_predict[0], tuple(range(0, n_feature)))
    mydatacreate.plotonesequence(showsequence, "X_predict[0]")
    showresult = mydatacreate.decode_vanillaLSTM(y_preidct[0].reshape(1,n_feature), tuple(range(0, n_feature)))
    mydatacreate.plotonesequence(showresult, "y_preidct[0]")
    mymodel.VanillaLSTM(LSTMmemorycount, (length, n_feature), X_train, y_train, X_predict, y_preidct)

def example_stacklstm():
    trainfitcount=10000
    evaluate_patterns=100
    n_patterns = 10
    length = 50
    n_features = 1
    out_count = 5
    LSTMmemorycount=20

    X_train, y_train, X_predict, y_preidct=mydatacreate.preparedata(trainfitcount, evaluate_patterns,
                                                                    mydatacreate.generateexample_stacklstm,
                                                                    n_patterns=n_patterns,length=length,n_features=n_features ,out_count=out_count)

    mydatacreate.plotonesequence(X_train[0][0],"X_train[0][0]")
    mydatacreate.plotonesequence(y_train[0][0],"y_train[0][0]")
    mymodel.StackedLSTM(LSTMmemorycount,(length,n_features),out_count,X_train,y_train,X_predict,y_preidct)

if __name__=="__main__":
    mydatacreate = DataCreateHelper.DataCreate()
    mymodel = ModelsHelper.Models()

    #example_vanillialstm()
    #example_stacklstm()
    example_cnnlstm()