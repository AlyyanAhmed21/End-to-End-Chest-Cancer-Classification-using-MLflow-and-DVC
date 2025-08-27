from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger
from dotenv import load_dotenv

# Load environment variables from .env file for MLflow credentials
load_dotenv()

STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        # The save_score() method is called inside evaluation()
        evaluation.log_into_mlflow()


# --- THIS IS THE CRITICAL BLOCK THAT TELLS THE SCRIPT TO RUN ---
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an object of the class and call its main method
        pipeline = EvaluationPipeline()
        pipeline.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e