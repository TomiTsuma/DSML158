import azure.functions as func
import logging
import pandas as pd
from scipy.spatial import distance
import pickle
import numpy as np
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="water-verification")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        _ = pd.DataFrame(req_body)
        unit_decision = pd.read_csv("water_unit_per_chemical_decision.csv")
        analysis_dict = pickle.load(open("analysis.dict","rb"))
        mahalanobis_thresholds = pickle.load(open("mahalanobis_thresholds.dict","rb"))

        result = {}
        _df = pd.DataFrame()
        for index, row in _.iterrows():
            analyses = row['analysis_id']
            for analysis_ in analyses:
                df_analysis_ = pd.DataFrame(row).T
                df_analysis_['analysis_name'] = df_analysis_['analysis_id'].apply(lambda x: analysis_dict[x] if x in [ i for i in analysis_dict.keys()] else None)
                _df = pd.concat([_df, df_analysis_])
        for index,row in _df.iterrows():
       
            sample_code = row['sample_code']
            analysis = row['analysis_name']
            analysis_id = row['analysis_id']
            if sample_code not in result.keys():
                result[sample_code] = []
            if analysis not in mahalanobis_thresholds.keys():
                result[sample_code].append({"sample_code": sample_code,"status":"fail", "message": f"Analysis not in models", "details": f"Analysis: {analysis} is not in the list of defined models" })   
                continue     
            scaler = pickle.load(open(f"scalers/{analysis}.pkl","rb"))
            pca = pickle.load(open(f"pca/{analysis}.pkl","rb"))

            pca_df = pd.read_csv(f"pca_df/{analysis}.csv",index_col=0)

            

            try :
                tmp_df = pd.DataFrame(row).T[pca_df.columns]   
                print(tmp_df)
            except:
                result[sample_code].append({"sample_code": sample_code,"status":"fail", "message": f"Missing parameters for analysis_id: {analysis_id}", "details": f"Expected parameters are {','.join(pca_df.columns)} for analysis: {analysis}" })
                continue
            failed_units_comparison = {}    
            for col in pca_df.columns:
                expected_units = unit_decision.loc[(unit_decision['crop'] == analysis) & (unit_decision['chemical_name'] == col)]
                print(expected_units[['crop','chemical_name','unit_name']].to_dict())
                if row[col]['units'] !=   expected_units['unit_name'].values[0] :
                    failed_units_comparison[col] = expected_units[['crop','chemical_name','unit_name']].to_dict()
                else:
                    row[col] = row[col]['result']
            if len(failed_units_comparison.keys()) > 0:
                result[sample_code].append({"sample_code": sample_code,"status":"fail", "message": f"Wrong units provided", "details": f"Expected units are {str(failed_units_comparison)} for analysis: {analysis}" })
                continue

            tmp_df = pd.DataFrame(row).T[pca_df.columns]    
            
            # tmp_df = 
            df_scaled = scaler.transform(tmp_df)
            df_pca = pd.DataFrame(pca.transform(df_scaled))

            mu = np.mean(pca_df, axis=0)
            sigma = np.cov(pca_df.T)

            mahalanobis_distance = distance.mahalanobis(df_pca.iloc[0], mu, np.linalg.inv(sigma))

            print(mahalanobis_distance)

            expected_md = mahalanobis_thresholds[analysis]
            print(expected_md)

            if mahalanobis_distance > expected_md:
                result[sample_code].append({"sample_code": sample_code,"status":"fail", "message": "Mahalanobis distance exceeds threshold", "description":f"Mahalanobis distance of {mahalanobis_distance} exceeds threshold of {expected_md} for analysis: {analysis}" })
            else:
                result[sample_code].append({"sample_code": sample_code,"status":"pass","message": "Mahalanobis distance within threshold", "description":f"Mahalanobis distance of {mahalanobis_distance} is within threshold of {expected_md} for analysis: {analysis}" })
            

        return func.HttpResponse(json.dumps(result),status_code=200)
    except Exception as e:
        print("Issue here", e)
        return func.HttpResponse(
             json.dumps(str(e)),
             status_code=500
        )