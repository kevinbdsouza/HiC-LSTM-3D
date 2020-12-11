""""This file updates app.config with info from config.ini."""
import platform
import threading
import traceback
import yaml
import logging

from common.constants import Constants

logger = logging.getLogger(__name__)


class Config:
    # Here will be the instance stored.
    __singleton_lock = threading.Lock()
    __instance = None
    __config = None
    __run_time = None

    @classmethod
    def get_instance(cls):
        """ Static access method. """
        if cls.__instance is None:
            with cls.__singleton_lock:
                if not cls.__instance:
                    cls.__instance = Config()
        return cls.__instance

    def __init__(self):
        Config.__config = Config.load_config()

    @classmethod
    def get_config(cls):
        return cls.__config

    @classmethod
    def set_config(cls, k, v):
        cls.__config[k] = v

    @staticmethod
    def load_config():
        """
            Setup logging configuration
        """
        config = None
        path = Constants.PIPELINE_CONFIGS
        try:
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
        except:
            logger.error(traceback.format_exc())

        return config

    @staticmethod
    def get_remote_folder():
        ins = Config.get_instance().get_config()
        return ins["remote_folder"]

    @staticmethod
    def get_destination_folder():
        ins = Config.get_instance().get_config()
        if platform.system() == 'Windows':
            return ins["destination_folder_win"]
        else:
            return ins["destination_folder_linux"]

    @staticmethod
    def get_handler_config():
        ins = Config.get_instance().get_config()
        return ins.get("handler_config")

    @staticmethod
    def get_loss_config():
        ins = Config.get_instance().get_config()
        return ins.get("loss_config")

    @staticmethod
    def get_optimizer_config():
        ins = Config.get_instance().get_config()
        return ins.get('optimizer_config')

    @staticmethod
    def get_model_config():
        ins = Config.get_instance().get_config()
        return ins.get('model_config')

    @staticmethod
    def get_model_mode():
        ins = Config.get_instance().get_config()
        return ins.get('model_mode')

    @staticmethod
    def get_training_mode():
        ins = Config.get_instance().get_config()
        return ins.get('training_mode')

    @staticmethod
    def get_env_name():
        ins = Config.get_instance().get_config()
        return ins.get('env_name')

    @staticmethod
    def get_dataset():
        ins = Config.get_instance().get_config()
        return ins.get('dataset_config')
