from airflow.models.dag import DAG
from airflow.decorators import task
import pendulum
from projects.image_processing.loading_cifar10 import LoadCifar10
from projects.image_processing.training import Train
from projects.image_processing.testing import Test
from projects.image_processing.classes_performance import ClassesPerformance


with DAG(
    dag_id="image_processing_nn",
    schedule=None,
    start_date=pendulum.datetime(2024, 10, 19, tz="Asia/Tehran"),
    catchup=False,
    tags=["1st project"],
    ) as dag:
    
    @task
    def load_data():
        LoadCifar10.load_cifar10()
        
    @task
    def train_data():
        Train.train()
        
    @task
    def test_data():
        Test.test()
        
    @task
    def performance():
        ClassesPerformance.classes_performance()
        
    load_data() >> train_data() >> test_data() >> performance()
        
        
        
    
        
        
