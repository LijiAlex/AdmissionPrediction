import logging
import os



class Utils:
    """
    Contains common utility methods
    """
    @staticmethod
    def init_logs():
        """
        Initializes the log file and default logging format
        :return: None
        """
        try:
            log_dir_path = os.getcwd()
            os.makedirs(log_dir_path, exist_ok=True)
            log_file = os.path.join(log_dir_path, "log.txt")
            logging.basicConfig(filename=log_file, level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s {%(pathname)s:%(lineno)d} %(message)s')
            logging.info("\n\n\n**********New execution begins************\n\n\n")
        except FileNotFoundError as e:
            print("Error: ", e)
        except Exception as e:
            logging.error("Error: ", e)

    # functions

    @staticmethod
    def standardization(data, scaler):
        """
        Standardizes the data

        data: dataset
        scaler: Standardization model

        Returns: First five records after modifying the dataset
        """
        arr = scaler.fit_transform(data)  # get standarised data
        return list(arr)


