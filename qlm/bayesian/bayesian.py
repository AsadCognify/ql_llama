import secrets
import requests
from typing import Dict
from datetime import datetime
from bayes_opt import UtilityFunction
# from control_plane.billing import Billing
from bayes_opt import BayesianOptimization
# from control_plane.utils.qlerrors import QLError
# from control_plane.plane import logger, update_mongo
from qlm.utils.logger import logger
from qlm.utils.pod_errors import PodError
# from concurrent.futures import ThreadPoolExecutor
# from control_plane.utils.cp_runpod import get_pod_public_ip_and_port_5000_mapping


def bayesian_finetune(data, ip_port:str=None, pod_id:Dict=None):
    try:
        logger.info(f"Entered bayesian mode with data {data}")
        # biller = Billing('billing.db')
        count = 1
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=data["p_bounds"],
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        for i in range(data["init_points"]):
            next_point = {}
            next_point["epochs"] = data[f"fin_{i+1}"]["general_ft_params"]["epochs"]
            next_point["learning_rate"] = data[f"fin_{i+1}"]["general_ft_params"]["lr"]
            next_point["rank"] = data[f"fin_{i+1}"]["qlora_params"]["Rank"]
            next_point["alpha_rank_ratio"] = data[f"fin_{i+1}"]["qlora_params"]["Alpha"]
            next_point["batch_size"] = data[f"fin_{i+1}"]["general_ft_params"][
                "batch_size"
            ]
            # Add job entry
            job_data = {
                "id": secrets.token_urlsafe(6), #jdid,
                "job_id": secrets.token_hex(12),
                "start_time": datetime.now(),
                "end_time": None,
                "pod_id" : pod_id,
                "duration": None,
                "cost": None
            }
            # biller.add_entry(table_name='jobs', data=job_data)
            
            request_action = requests.post(
                # f"https://{pod_id}-5000.proxy.runpod.net/finetune/bayesian",
                # url=f"http://{ip_port}/finetune/bayesian",
                url=f"http://localhost:5000/finetune/bayesian",
                json=data[f"fin_{i+1}"],
            )
        
            # Update job entry
            # job_cost = biller.update_job_entry(token_id=job_data['id'], end_time=datetime.now())
            # bot_id = data[f"fin_{i+1}"]["definition"]["bot_id"]
            # combination_id = data[f"fin_{i+1}"]["definition"]["combination_id"]
            # update_mongo(finetune_cost=job_cost, bot_endpoint=f'{bot_id}/{combination_id}')

            logger.debug(f"Point: {count}\nResponse: {request_action.status_code} Response Text: {request_action.text}")
            count += 1

            optimizer.register(
                params=next_point, target=-1 * request_action.json()["eval_loss"]
            )
        for i in range(data["total_iterations"] - data["init_points"]):
            next_point = optimizer.suggest(utility)
            # send data and next_point to main backend and get a data object data = {"data":data, "next_point":next_point}
            bot_id = data["fin_1"]["definition"]["bot_id"]
            response = requests.post(
                # url=f"http://localhost:5000/api/eval_bot/create/combination/{bot_id}",
                url=f"https://stage.queryloop-ai.com/api/eval_bot/create/combination/{bot_id}",
                json={
                    "data": data["fin_1"],
                    "next_point": next_point,
                    "optimization_parameters": data["optimization_parameters"],
                    "optimization": data["optimization"],
                },
            )
            new_comb = response.json()['combination']

            # Add job entry
            job_data = {
                "id": secrets.token_urlsafe(6), #jdid,
                "job_id": secrets.token_hex(12),
                "start_time": datetime.now(),
                "end_time": None,
                "pod_id" : pod_id,
                "duration": None,
                "cost": None
            }
            # biller.add_entry(table_name='jobs', data=job_data)

            request_action = requests.post(
                # f"https://{pod_id}-5000.proxy.runpod.net/finetune/bayesian",
                # url=f"http://{ip_port}/finetune/bayesian",
                url=f"http://localhost:5000/finetune/bayesian",
                json=new_comb,
            )
            # Update job entry
            # job_cost = biller.update_job_entry(token_id=job_data['id'], end_time=datetime.now())
            # bot_id, combination_id = new_comb["definition"]["bot_id"], new_comb["definition"]["combination_id"]
            # update_mongo(finetune_cost=job_cost, bot_endpoint=f'{bot_id}/{combination_id}')

            logger.debug(
                f"Point: {count}\nrequest_action: {request_action.status_code} {request_action.text}"
            )
            count += 1
            optimizer.register(
                params=next_point, target=-1 * request_action.json()["eval_loss"]
            )

        # Delete biller SQLite object
        # del biller
        # Terminate pod since all the points ran without raising an exception
        try:
            if data['testing']['state'] == True:
                    request_terminate = requests.post(
                    url=f"{data['testing']['url']}/request/notify",
                    json={'token': data["token"], 'code': 36, 'bot_endpoint': ""}
                )
            

        except:
            request_terminate = requests.post(
                # url="http://localhost:8000/request/delete_model",
                url="https://cp.queryloop-ai.com/request/delete_model",
                json={
                    'token': data["token"],
            })
            if request_terminate.status_code == 200:
                return [optimizer.max, request_terminate.json()['cost']]
            else:
                # raise QLError("Error in pod termination", 5026)
                raise PodError("Error in pod termination", 5026)
        
    except Exception as e:
        # print(f"Error in bayesian_finetune: {e}")
        logger.error(f"Error in bayesian finetune: {e}", exc_info=True)
        # return {"status": f"Failed to start finetune because {e}"}
        # raise QLError("Failed Bayesian finetune because {e}", 5016)
        raise PodError("Failed Bayesian finetune because {e.args[0]}", 5016) from e
    
        # Pod shutdown handled at pod
