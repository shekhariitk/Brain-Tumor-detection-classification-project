from src.logger import logging
from src.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline




def main():
    STAGE_NAME = "Data Ingestion stage"
    try:
      logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = DataIngestionTrainingPipeline()
      data_ingestion.main()
      logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\nx==========x")
    except Exception as e:
            logging.exception(e)
            raise e
    
if __name__ == "__main__":
    main()