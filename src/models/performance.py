# Source: AdvDSI-Lab2-Exercise1-Solutions.ipynb
# Author: Anthony So

def score_model(X, y, set_name=None, target_type='regression', model=None):
    """Print regular performance statistics for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    import pandas as pd
    import numpy as np
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
           
    model_scores = []

    y_preds = model.predict(X)
    perf_accuracy  = accuracy_score(y, y_preds)

    if target_type == "regression":
        perf_mse       = mse(y, y_preds, squared=False)
        perf_mae       = mae(y, y_preds)

        model_scores.append([set_name, perf_accuracy, perf_mse, perf_mae])
    
        df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','MSE','MAE'])
    else:
        if target_type == "binary":
            average = 'binary'

            y_predict_proba = model.predict_proba(X)[:, 1]
            perf_AUC        = roc_auc_score(y, y_predict_proba)

        if target_type == "multiclass":
            average = 'macro'

        perf_precision  = precision_score(y, y_preds, average=average, zero_division=1)
        perf_recall     = recall_score(y, y_preds, average=average, zero_division=1)
        perf_F1         = f1_score(y, y_preds, average=average, zero_division=1)

        if target_type == "binary":
            model_scores.append([set_name, perf_accuracy, perf_precision, perf_recall, perf_F1, perf_AUC])
    
            df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','PREC','RECALL','F1','AUC'])

        if target_type == "multiclass":
            model_scores.append([set_name, perf_accuracy, perf_precision, perf_recall, perf_F1])
    
            df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','PREC','RECALL','F1'])

    return df_model_scores

# New NULL Model

def score_null_model(y_train, y_base, set_name=None, target_type='regression'):
    """Print regular performance statistics for the provided data

    Parameters
    ----------
    y_train : Numpy Array
        Predicted target
    y_base : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    model : str
        Model to be used

    Returns
    -------
    """
    import pandas as pd
    import numpy as np
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    model_scores = []

    perf_accuracy  = accuracy_score(y_base, y_train)

    if target_type == "regression":
        perf_mse       = mse(y_base, y_train, squared=False)
        perf_mae       = mae(y_base, y_train)

        model_scores.append([set_name, perf_accuracy, perf_mse, perf_mae])
    
        df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','MSE','MAE'])
    else:
        if target_type == "binary":
            average = 'binary'
            perf_AUC       = None #roc_auc_score(y_base, model.predict_proba(y_preds)[:, 1])}')

        if target_type == "multiclass":
            average = 'macro'

        perf_precision = precision_score(y_base, y_train, average=average, zero_division=1)
        perf_recall    = recall_score(y_base, y_train, average=average, zero_division=1)
        perf_F1        = f1_score(y_base, y_train, average=average, zero_division=1)

        if target_type == "binary":
            model_scores.append([set_name, perf_accuracy, perf_precision, perf_recall, perf_F1, perf_AUC])
    
            df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','PREC','RECALL','F1','AUC'])


        if target_type == "multiclass":
            model_scores.append([set_name, perf_accuracy, perf_precision, perf_recall, perf_F1])
    
            df_model_scores = pd.DataFrame (model_scores, columns = ['Set Name','ACC','PREC','RECALL','F1'])

    return df_model_scores


def score_models(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, y_base=None, includeBase=False, target_type='regression', model=None):
    
    """Score Models and return results as a dataframe

    Parameters
    ----------
    X_train : Numpy Array
        X_train data
    y_train : Numpy Array
        Train target
    X_val : Numpy Array
         X_val data
    y_val : Numpy Array
        Val target
    X_test : Numpy Array
         X_test data
    y_test : Numpy Array
        Test target
    includeBase: Boolean
        Calculate and display baseline
    model: model
        Model passed into function

    Returns
    -------
    """

    import pandas as pd
    import numpy as np

    df_model_scores = pd.DataFrame()

    if includeBase == True:
        df_model_scores_base = score_null_model(y_train = y_train, y_base = y_base, set_name='Base', target_type=target_type)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_base],ignore_index = True, axis=0)

    if X_train.size > 0:
        df_model_scores_train = score_model(X_train, y_train, set_name='Train', target_type=target_type, model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_train],ignore_index = True, axis=0)

    if X_val.size > 0:
        df_model_scores_val = score_model(X_val, y_val, set_name='Validate', target_type=target_type, model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_val],ignore_index = True, axis=0)

    if X_test.size > 0:
        df_model_scores_test = score_model(X_test, y_test, set_name='Test', target_type=target_type, model=model)

        df_model_scores = pd.concat([df_model_scores,df_model_scores_test],ignore_index = True, axis=0)

    display(df_model_scores)

    return

def fit_score_models(models, X_t, y_t, X_v, y_v, target_type="regression", dump_model="NO"):
    import pandas as pd
    import numpy as np
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    # Time related modules
    from datetime import datetime
    import pytz

    # Declare variables to store name of timezone
    tz_SYD = pytz.timezone('Australia/Sydney')

    model_scores = []
    for name, model in models.items():
        print("*******************************")
        print(datetime.now(tz_SYD), "- Start fit and score for model: ", name)
        clf = model
        clf.fit(X_t, y_t)
        print(datetime.now(tz_SYD), "- End fit for model: ", name)

        if dump_model == "YES":
            job.dump(clf, "../models/williams_sean-assignment2_" + name + ".joblib", compress=3)

        print(datetime.now(tz_SYD), "- Make train preds for model: ", name)
        t_preds = clf.predict(X_t)
        t_acc = accuracy_score(y_t, t_preds)

        print(datetime.now(tz_SYD), "- Make val preds for model: ", name)
        v_preds = clf.predict(X_v)
        v_acc = accuracy_score(y_v, v_preds)

        if target_type == "regression":
            t_mse = mean_squared_error(y_t, t_preds, squared=False)
            t_mae = mean_absolute_error(y_t, t_preds)

            v_mse = mean_squared_error(y_v, v_preds, squared=False)
            v_mae = mean_absolute_error(y_v_t, v_preds)

            model_scores.append([name, t_acc, v_acc, t_mse, v_mse, t_mae, v_mae])            
        else:
            if target_type == "binary":
                average = 'binary'

                print(datetime.now(tz_SYD), "- Calc train probs for model: ", name)
                t_probs = clf.predict_proba(X_t)[:, 1]
                t_auc = roc_auc_score(y_t, t_probs)

                print(datetime.now(tz_SYD), "- Calc val probs for model: ", name)
                v_probs = clf.predict_proba(X_v)[:, 1]
                v_auc = roc_auc_score(y_v, v_probs)
             
            if target_type == "multiclass":
                average = 'macro'

            t_prec = precision_score(y_t, t_preds, average=average, zero_division=1)
            t_rec = recall_score(y_t, t_preds, average=average, zero_division=1)
            t_f1 = f1_score(y_t, t_preds, average=average, zero_division=1)

            v_prec = precision_score(y_v, v_preds, average=average, zero_division=1)
            v_rec = recall_score(y_v, v_preds, average=average, zero_division=1)
            v_f1 = f1_score(y_v, v_preds, average=average, zero_division=1)

            if target_type == "binary":
               model_scores.append([name, t_acc, v_acc, t_prec, v_prec, t_rec, v_rec, t_f1, v_f1, t_auc, v_auc])

            if target_type == "multiclass":
               model_scores.append([name, t_acc, v_acc, t_prec, v_prec, t_rec, v_rec, t_f1, v_f1])

        print(datetime.now(tz_SYD), "- End fit and score for model: ", name)
        print("*******************************")
        print("                               ")

    if target_type == "regression":
        df_model_scores = pd.DataFrame (model_scores, columns = ['model','t_ACC','v_ACC','t_MSE','v_MSE','t_MAE','v_MAE'])

    if target_type == "binary":
        df_model_scores = pd.DataFrame (model_scores, columns = ['model','t_ACC','v_ACC','t_PREC','v_PREC','t_RECALL','v_RECALL','t_F1','v_F1','t_AUC','v_AUC'])

    if target_type == "multiclass":
        df_model_scores = pd.DataFrame (model_scores, columns = ['model','t_ACC','v_ACC','t_PREC','v_PREC','t_RECALL','v_RECALL','t_F1','v_F1'])

    display(df_model_scores)