# training_config.py
# Serves as a template for setting finetuning variables
class LLAMA3TrainingConfig:
    def __init__(self):
        self.data_path = None
        self.model_dir = None
        self.out_path = None
        self.start_epoch = None
        self.end_epoch = None
        self.lora_r = None
        self.lora_alpha = None
        self.learning_rate = None
        self.batch_size = None
        self.save_steps = None
        self.logging_steps = None

    def set_variables(self, data_path: str, model_dir: str, out_path: str, start_epoch: int, end_epoch: int, lora_r: int, lora_alpha: float, learning_rate: float, batch_size: int, save_steps: int, logging_steps: int):
        self.data_path = data_path
        self.model_dir = model_dir
        self.out_path = out_path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.logging_steps = logging_steps

    def set_variable(self, variable_name: str, variable_value):
        setattr(self, variable_name, variable_value)

    def get_variable(self, variable_name: str):
        return getattr(self, variable_name)

    def get_model_params(self):
        return {
            "model_dir": self.model_dir,
            "out_path": self.out_path,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps
        }

    def get_all_variables(self):
        return {
            "data_path": self.data_path,
            "model_dir": self.model_dir,
            "out_path": self.out_path,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps
        }
