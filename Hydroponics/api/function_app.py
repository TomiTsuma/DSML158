import azure.functions as func
import logging
import pandas as pd
from scipy.spatial import distance
import pickle
import numpy as np
import json
import math

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="hydroponicsVerification")
def hydroponicsVerification(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        _ = pd.DataFrame(req_body)

        unit_decision = pd.read_csv("hydroponics_unit_per_chemical_decision.csv")
        unit_decision.analysis_name = [ i.replace(":","") for i in unit_decision.analysis_name ]
        analysis_dict = pickle.load(open("analysis.dict","rb"))
        mahalanobis_thresholds = pickle.load(open("mahalanobis_thresholds.dict","rb"))

        result = {}
        _df = _.explode('analysis_id', ignore_index=True)

        import math
        for index,row in _df.iterrows():
            sample_code = row['sample_code']
            if sample_code not in result.keys():
                result[sample_code] = []
            if math.isnan(row['analysis_id']):
                result[sample_code].append({"sample_code": sample_code,"status":"warning", "message": f"Analysis not in specified", "details": f"Analysis id not provided", "analysis": None  })   
                continue
            row['analysis_name'] = analysis_dict[row['analysis_id']]
            analysis = row['analysis_name'].replace(":","")
            analysis_id = row['analysis_id']
            if analysis not in mahalanobis_thresholds.keys():
                result[sample_code].append({"sample_code": sample_code,"status":"warning", "message": f"Analysis not in models", "details": f"Analysis: {analysis} is not in the list of defined models", "analysis": row['analysis_name'] })   
                continue  
            row = row.dropna()
            scaler = pickle.load(open(f"models/scalers/{analysis}.pkl","rb"))
            pca = pickle.load(open(f"models/pca/{analysis}.pkl","rb"))
            imputer = pickle.load(open(f"models/imputers/{analysis}.pkl","rb"))
            analysis_df = pd.read_csv(f"analysis/{analysis}.csv",index_col=0)

            analysis_df = analysis_df.loc[:, ~analysis_df.columns.duplicated()]

            mahalanobis_distance_df = pd.read_csv(f"mahalanobis_distance/{analysis}.csv",index_col=0).drop("mahalanobis_distance",axis=1)

            

            try :
                tmp_df = pd.DataFrame(row).T[analysis_df.columns]
            except Exception as e:
                result[sample_code].append({"sample_code": sample_code,"status":"warning", "message": f"Missing parameters for analysis: {analysis}", "details": f"Missing parameters: {','.join([ i for i in analysis_df.columns if i not in row.dropna().index])} for analysis: {analysis}", "analysis": row['analysis_name']})
                continue
            failed_units_comparison = {}    
            for col in tmp_df.columns:
                expected_units = unit_decision.loc[(unit_decision['analysis_name'] == analysis) & (unit_decision['chemical_name'] == col)]
            
                import math
                if type(row[col]) != dict and math.isnan(row[col]):
                    failed_units_comparison[col] = {}
                    failed_units_comparison[col]['expected_units'] = expected_units['unit_name'].tolist()[0]
                    failed_units_comparison[col]['units_provided'] = None
                elif row[col]['units'] !=   expected_units['unit_name'].values[0] :
                    failed_units_comparison[col] = {}
                    failed_units_comparison[col]['expected_units'] = expected_units['unit_name'].tolist()[0]
                    failed_units_comparison[col]['units_provided'] = row[col]['units']
                else:
                    continue    
            if len(failed_units_comparison.keys()) > 0:
                result[sample_code].append({"sample_code": sample_code,"status":"warning", "message": f"Wrong units provided", "details": f"Expected units are {str(failed_units_comparison)} for analysis: {analysis}" , "analysis": row['analysis_name']})
                continue
            for col in analysis_df.columns:
                if type(row[col]) == dict:
                    row[col] = row[col]['result']
            
            out_of_bounds_chems = [ ]
            for col in tmp_df.columns:
                if col == "sample_code":
                    continue
                if col == "ec_salts":
                    if row[col] > 105 or row[col] < 95:
                        out_of_bounds_chems.append("ec_salts out of bounds. Allowed bounds are 95 - 105")
                if col == "Charge Balance":
                    if row[col] < -1:
                        out_of_bounds_chems.append("Charge Balance out of bounds. Allowed lower boundary is -10. ")       
                elif col.lower() == "total suspended solids" and row['analysis_name'].lower() == "total suspended solids":
                    if row[col] > 1:
                        out_of_bounds_chems.append("total suspended solids out of bounds. If analysis is total suspended solids is <1 then total suspended solids check should be <1  . ")         
            if len(out_of_bounds_chems) > 0:
                result[sample_code].append({"sample_code": sample_code,"status":"warning", "message": f"Out of bounds", "details": f"{'.'.join(out_of_bounds_chems)}", "analysis": row['analysis_name'] })
                continue    

            tmp_df = pd.DataFrame(row).T[analysis_df.columns]
            tmp_df = imputer.transform(tmp_df)
            df_scaled = scaler.transform(tmp_df)
            df_pca = pd.DataFrame(pca.transform(df_scaled))
            
            df_pca.columns = [ i for i in df_pca.columns]
            mahalanobis_distance_df.columns = [ i for i in df_pca.columns]
            df_pca = pd.concat([df_pca, mahalanobis_distance_df])
            mu = np.mean(df_pca, axis=0)
            sigma = np.cov(df_pca.T)

            mahalanobis_distance = distance.mahalanobis(df_pca.iloc[0], mu, np.linalg.inv(sigma))

            expected_md = mahalanobis_thresholds[analysis]

            if mahalanobis_distance > expected_md:
                result[sample_code].append({"sample_code": sample_code,"status":"fail", "message": "Mahalanobis distance exceeds threshold", "details":f"Mahalanobis distance of {round(mahalanobis_distance,2)} exceeds threshold of {round(expected_md,2)} for analysis: {analysis}", "analysis": row['analysis_name'] })
            else:
                result[sample_code].append({"sample_code": sample_code,"status":"pass","message": "Mahalanobis distance within threshold", "details":f"Mahalanobis distance of {round(mahalanobis_distance,2)} is within threshold of {round(expected_md,2)} for analysis: {analysis}", "analysis": row['analysis_name'] })
        return func.HttpResponse(json.dumps(result),status_code=200)
    except Exception as e:
        print("Issue here", e)
        return func.HttpResponse(
             json.dumps(str(e)),
             status_code=500
        )